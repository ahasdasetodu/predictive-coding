from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, TensorDataset

from pc_core.balance import BalanceConfig
from pc_core.byproduct import ByproductRegime, ConditionWeights
from pc_core.byproduct_eval import evaluate_hub
from pc_core.dataset import load_dataset
from pc_core.experiment.spec import (
    Kind,
    parse_reference,
    parse_sampler,
    reference_document,
    register,
    runner_argv,
    sampler_document,
)
from pc_core.experiment.yamlio import (
    as_bool,
    as_float,
    as_int,
    at_least,
    optional_float,
    reject_unknown,
    require,
    run_name,
    string_key_mapping,
    write_yaml,
)
from pc_core.graph import PCGraph
from pc_core.messages import ClueSampler, Degradation, SamplerConfig, label_prior
from pc_core.models import GraphTopology, consensus_graph, hub_graph
from pc_core.reference import ReferenceConfig, clean_reference, reference_accuracy, train_reference
from pc_core.regimes import PCConfig
from pc_core.trainer import EpochReport, Trainer

KindName = Literal["mnist", "clue-bits", "byproduct"]

_BYPRODUCT_REQUIRED = {"kind", "name", "seed", "graph", "epochs", "infer_steps", "eval_steps", "degrade"}
_BYPRODUCT_OPTIONAL = {
    "conditions", "sampler", "reference", "balance", "eval_curve", "thoughts",
    "infer_lr", "eval_lr", "weight_lr", "min_lr",
}
_GRAPH_KEYS = {"topology", "z", "h_img", "h_msg", "h_lab", "share_init"}
_GRAPH_REQUIRED = {"z", "h_img"}
_DEGRADE_KEYS = {"kind", "level"}
_CONDITION_KEYS = {"image", "msg", "both"}
_DEFAULT_EVAL_CURVE = (1, 5, 10, 30)
_DEFAULT_THOUGHTS = 24


@dataclass(frozen=True)
class ByproductSpec:
    """Train a hub or consensus graph with random modality presence, then read it out per clue type."""

    name: str
    seed: int
    z_dim: int
    h_dim: int
    h_msg: int
    h_lab: int
    epochs: int
    infer_steps: int
    eval_steps: int
    infer_lr: float
    eval_lr: float
    weight_lr: float
    min_lr: float
    degradation: Degradation
    conditions: ConditionWeights
    sampler: SamplerConfig
    reference: ReferenceConfig
    balance: bool
    eval_curve: tuple[int, ...]
    thoughts: int
    topology: GraphTopology = "hub"
    share_init: bool = True

    kind: KindName = "byproduct"

    def pc_config(self) -> PCConfig:
        """Settling steps and learning rates for this run."""
        return PCConfig(
            infer_steps=self.infer_steps,
            eval_steps=self.eval_steps,
            infer_lr=self.infer_lr,
            eval_lr=self.eval_lr,
            weight_lr=self.weight_lr,
            min_lr=self.min_lr,
        )


def parse_byproduct(raw: Mapping[str, object]) -> ByproductSpec:
    """Parse a hub training spec, with defaults for the read-out settings."""
    reject_unknown(raw, _BYPRODUCT_REQUIRED | _BYPRODUCT_OPTIONAL, "spec")
    require(raw, _BYPRODUCT_REQUIRED, "spec")
    defaults = PCConfig()
    graph = string_key_mapping(raw["graph"], "graph")
    reject_unknown(graph, _GRAPH_KEYS, "graph")
    require(graph, _GRAPH_REQUIRED, "graph")
    z_dim = at_least(as_int(graph["z"], "z"), "z", 1)
    return ByproductSpec(
        name=run_name(raw["name"]),
        seed=as_int(raw["seed"], "seed"),
        z_dim=z_dim,
        h_dim=at_least(as_int(graph["h_img"], "h_img"), "h_img", 1),
        h_msg=at_least(as_int(graph.get("h_msg", z_dim), "h_msg"), "h_msg", 1),
        h_lab=at_least(as_int(graph.get("h_lab", z_dim), "h_lab"), "h_lab", 1),
        topology=_topology(graph.get("topology", "hub")),
        share_init=as_bool(graph.get("share_init", True), "share_init"),
        epochs=at_least(as_int(raw["epochs"], "epochs"), "epochs", 1),
        infer_steps=at_least(as_int(raw["infer_steps"], "infer_steps"), "infer_steps", 0),
        eval_steps=at_least(as_int(raw["eval_steps"], "eval_steps"), "eval_steps", 0),
        infer_lr=optional_float(raw, "infer_lr", defaults.infer_lr),
        eval_lr=optional_float(raw, "eval_lr", defaults.eval_lr),
        weight_lr=optional_float(raw, "weight_lr", defaults.weight_lr),
        min_lr=optional_float(raw, "min_lr", defaults.min_lr),
        degradation=_degradation(raw["degrade"]),
        conditions=_conditions(raw.get("conditions", {})),
        sampler=parse_sampler(raw.get("sampler", {})),
        reference=parse_reference(raw.get("reference", {})),
        balance=as_bool(raw.get("balance", True), "balance"),
        eval_curve=_eval_curve(raw.get("eval_curve", list(_DEFAULT_EVAL_CURVE))),
        thoughts=at_least(as_int(raw.get("thoughts", _DEFAULT_THOUGHTS), "thoughts"), "thoughts", 0),
    )


def dump_byproduct(spec: object) -> dict[str, object]:
    """YAML document for a byproduct spec, including resolved defaults."""
    if not isinstance(spec, ByproductSpec):
        raise TypeError("dump_byproduct expects a ByproductSpec")
    return {
        "kind": spec.kind,
        "name": spec.name,
        "seed": spec.seed,
        "graph": {
            "topology": spec.topology,
            "z": spec.z_dim,
            "h_img": spec.h_dim,
            "h_msg": spec.h_msg,
            "h_lab": spec.h_lab,
            "share_init": spec.share_init,
        },
        "epochs": spec.epochs,
        "infer_steps": spec.infer_steps,
        "eval_steps": spec.eval_steps,
        "infer_lr": spec.infer_lr,
        "eval_lr": spec.eval_lr,
        "weight_lr": spec.weight_lr,
        "min_lr": spec.min_lr,
        "degrade": {"kind": spec.degradation.kind, "level": spec.degradation.level},
        "conditions": {"image": spec.conditions.image, "msg": spec.conditions.msg, "both": spec.conditions.both},
        "sampler": sampler_document(spec.sampler),
        "reference": reference_document(spec.reference),
        "balance": spec.balance,
        "eval_curve": list(spec.eval_curve),
        "thoughts": spec.thoughts,
    }


def train_byproduct(spec: ByproductSpec, directory: Path | None = None) -> list[EpochReport]:
    """Train the graph on MNIST with clues, then write the per-type read-out, grids, and plot.

    Validation images are degraded once with a fixed seed so every epoch and every
    condition sees the same pixels. Training images are degraded afresh each batch.
    Trainer's train accuracy is measured on undegraded training images.
    """
    torch.manual_seed(spec.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = load_dataset(batch_size=256)
    reference_clean, confusion = clean_reference(train_loader, spec.reference, device, spec.seed)
    if spec.degradation.kind == "none":
        reference_degraded = reference_clean
    else:
        reference_degraded = train_reference(train_loader, spec.degradation, spec.reference, device, spec.seed)
    degraded_test = _degraded_loader(test_loader, spec.degradation, device, spec.seed + 1)
    regime = ByproductRegime(
        _byproduct_graph(spec),
        spec.pc_config(),
        ClueSampler(spec.sampler, confusion.float(), seed=spec.seed),
        spec.degradation,
        spec.conditions,
        BalanceConfig() if spec.balance else None,
        device=device,
        seed=spec.seed,
    )
    reports = Trainer(regime, train_loader, degraded_test).train(epochs=spec.epochs)
    if directory is None:
        return reports
    prior = label_prior(torch.cat([labels for _, labels in train_loader]))
    evaluation = evaluate_hub(
        regime,
        degraded_test,
        reference_clean,
        reference_degraded,
        prior,
        spec.sampler,
        confusion.float(),
        directory,
        spec.eval_curve,
        spec.thoughts,
        reference_clean_accuracy=reference_accuracy(reference_clean, test_loader, Degradation(), device, spec.seed),
        reference_degraded_accuracy=reference_accuracy(
            reference_degraded, degraded_test, Degradation(), device, spec.seed
        ),
        seed=spec.seed,
    )
    document: dict[str, object] = {
        "degradation": spec.degradation.describe(),
        "validation_images": sum(labels.shape[0] for _, labels in degraded_test),
        "train_accuracy_note": "Trainer's train accuracy is on undegraded training images.",
        "ratios_z": dict(regime.ratios),
        "prior": [round(float(p), 6) for p in prior],
        "confusion": [[int(count) for count in row] for row in confusion],
        **evaluation.document(),
    }
    write_yaml(directory / "evaluation.yaml", document)
    return reports


def _byproduct_graph(spec: ByproductSpec) -> PCGraph:
    """Build the hub or consensus graph named in the spec."""
    if spec.topology == "consensus":
        return consensus_graph(
            z_dim=spec.z_dim,
            h_img=spec.h_dim,
            h_msg=spec.h_msg,
            h_lab=spec.h_lab,
            share_init=spec.share_init,
        )
    return hub_graph(z_dim=spec.z_dim, h_dim=spec.h_dim)


def _degraded_loader(
    loader: DataLoader,
    degradation: Degradation,
    device: str | torch.device,
    seed: int,
) -> DataLoader:
    """A fixed, already-degraded copy of a loader's images, in its order."""
    generator = torch.Generator(device=device).manual_seed(seed)
    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_images, batch_labels in loader:
        images.append(degradation.apply(batch_images.to(device), generator).cpu())
        labels.append(batch_labels)
    return DataLoader(TensorDataset(torch.cat(images), torch.cat(labels)), batch_size=loader.batch_size, shuffle=False)


def _degradation(value: object) -> Degradation:
    """The image degradation: `none`, `noise` with sigma, or `occlude` with a patch side."""
    raw = string_key_mapping(value, "degrade")
    reject_unknown(raw, _DEGRADE_KEYS, "degrade")
    require(raw, {"kind"}, "degrade")
    kind = raw["kind"]
    level = as_float(raw.get("level", 0.0), "level")
    if kind == "none":
        return Degradation("none", 0.0)
    if kind == "noise":
        if level <= 0.0:
            raise ValueError("noise needs a positive level")
        return Degradation("noise", level)
    if kind == "occlude":
        if level < 1 or level > 28 or level != int(level):
            raise ValueError("occlude needs a whole level between 1 and 28")
        return Degradation("occlude", level)
    raise ValueError("degrade kind must be none, noise, or occlude")


def _conditions(value: object) -> ConditionWeights:
    """Weights for image-only, clue-only, and both; omitted conditions keep weight one."""
    raw = string_key_mapping(value, "conditions")
    reject_unknown(raw, _CONDITION_KEYS, "conditions")
    weights = ConditionWeights(
        image=optional_float(raw, "image", 1.0),
        msg=optional_float(raw, "msg", 1.0),
        both=optional_float(raw, "both", 1.0),
    )
    if min(weights.as_list()) < 0 or sum(weights.as_list()) <= 0:
        raise ValueError("condition weights must be non-negative and not all zero")
    return weights


def _eval_curve(value: object) -> tuple[int, ...]:
    """Settling step counts for the image-only evaluation curve."""
    if not isinstance(value, list) or not value:
        raise ValueError("eval_curve must be a non-empty list of step counts")
    return tuple(at_least(as_int(steps, "eval_curve"), "eval_curve", 0) for steps in value)


def _topology(value: object) -> GraphTopology:
    """Hub or consensus graph for a byproduct spec."""
    if value == "hub":
        return "hub"
    if value == "consensus":
        return "consensus"
    raise ValueError("graph topology must be hub or consensus")


KIND = register(
    Kind(
        name="byproduct",
        parse=parse_byproduct,
        dump=dump_byproduct,
        run=train_byproduct,
        argv=runner_argv,
    )
)
