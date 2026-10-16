from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from pc_core.clue_table import measure_clue_table
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
from pc_core.experiment.yamlio import as_int, reject_unknown, require, run_name, write_yaml
from pc_core.messages import SamplerConfig
from pc_core.reference import ReferenceConfig
from pc_core.trainer import EpochReport

KindName = Literal["mnist", "clue-bits", "byproduct"]

_CLUE_BITS_REQUIRED = {"kind", "name", "seed"}
_CLUE_BITS_OPTIONAL = {"sampler", "reference"}


@dataclass(frozen=True)
class ClueBitsSpec:
    """Measure bits and clue-only Bayes accuracy per clue type on the training labels."""

    name: str
    seed: int
    sampler: SamplerConfig
    reference: ReferenceConfig

    kind: KindName = "clue-bits"


def parse_clue_bits(raw: Mapping[str, object]) -> ClueBitsSpec:
    """Parse a clue-bits measurement spec."""
    reject_unknown(raw, _CLUE_BITS_REQUIRED | _CLUE_BITS_OPTIONAL, "spec")
    require(raw, _CLUE_BITS_REQUIRED, "spec")
    return ClueBitsSpec(
        name=run_name(raw["name"]),
        seed=as_int(raw["seed"], "seed"),
        sampler=parse_sampler(raw.get("sampler", {})),
        reference=parse_reference(raw.get("reference", {})),
    )


def dump_clue_bits(spec: object) -> dict[str, object]:
    """YAML document for a clue-bits spec, including resolved defaults."""
    if not isinstance(spec, ClueBitsSpec):
        raise TypeError("dump_clue_bits expects a ClueBitsSpec")
    return {
        "kind": spec.kind,
        "name": spec.name,
        "seed": spec.seed,
        "sampler": sampler_document(spec.sampler),
        "reference": reference_document(spec.reference),
    }


def measure_clue_bits(spec: ClueBitsSpec, directory: Path | None = None) -> list[EpochReport]:
    """Write the per-type bits table to `table.yaml` in the result directory and print it."""
    train_loader, _ = load_dataset(batch_size=256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = measure_clue_table(train_loader, spec.sampler, spec.reference, device, spec.seed)
    print(table.render())
    if directory is not None:
        write_yaml(directory / "table.yaml", table.document())
    return []


KIND = register(
    Kind(
        name="clue-bits",
        parse=parse_clue_bits,
        dump=dump_clue_bits,
        run=measure_clue_bits,
        argv=runner_argv,
    )
)
