import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from pc_core.dataset import load_dataset
from pc_core.experiment.spec import Kind, register
from pc_core.experiment.yamlio import (
    as_bool,
    as_float,
    as_int,
    at_least,
    negative_float,
    optional_float,
    reject_unknown,
    require,
    run_name,
)
from pc_core.models import InitScheme, mlp_graph
from pc_core.regimes import BalancedRegime, ForwardInitRegime, PCConfig, PCRegime
from pc_core.trainer import EpochReport, Trainer

RegimeName = Literal["zero", "forward", "balanced"]
KindName = Literal["mnist", "clue-bits", "byproduct"]

REGIMES: dict[str, type[PCRegime]] = {
    "zero": PCRegime,
    "forward": ForwardInitRegime,
    "balanced": BalancedRegime,
}

_SPEC_REQUIRED = {
    "name",
    "seed",
    "regime",
    "init",
    "hidden",
    "epochs",
    "infer_steps",
    "eval_steps",
    "learn_precision",
    "min_log_var",
    "image_weight",
}
_SPEC_OPTIONAL = {"kind", "infer_lr", "eval_lr", "weight_lr", "min_lr"}
_CLI_NAME = "cli"


@dataclass(frozen=True)
class ExperimentSpec:
    """One named training run, including every setting that changes the result."""

    name: str
    seed: int
    regime: RegimeName
    init: InitScheme
    hidden: tuple[int, ...]
    epochs: int
    infer_steps: int
    eval_steps: int
    learn_precision: bool
    min_log_var: float
    image_weight: float
    infer_lr: float
    eval_lr: float
    weight_lr: float
    min_lr: float

    kind: KindName = "mnist"

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


def parse_mnist(raw: Mapping[str, object]) -> ExperimentSpec:
    """Parse an MNIST training spec, fill omitted learning rates, and reject unknown keys."""
    reject_unknown(raw, _SPEC_REQUIRED | _SPEC_OPTIONAL, "spec")
    require(raw, _SPEC_REQUIRED, "spec")
    defaults = PCConfig()
    return ExperimentSpec(
        name=run_name(raw["name"]),
        seed=as_int(raw["seed"], "seed"),
        regime=_regime(raw["regime"]),
        init=_init(raw["init"]),
        hidden=_hidden(raw["hidden"]),
        epochs=at_least(as_int(raw["epochs"], "epochs"), "epochs", 1),
        infer_steps=at_least(as_int(raw["infer_steps"], "infer_steps"), "infer_steps", 0),
        eval_steps=at_least(as_int(raw["eval_steps"], "eval_steps"), "eval_steps", 0),
        learn_precision=as_bool(raw["learn_precision"], "learn_precision"),
        min_log_var=negative_float(raw["min_log_var"], "min_log_var"),
        image_weight=as_float(raw["image_weight"], "image_weight"),
        infer_lr=optional_float(raw, "infer_lr", defaults.infer_lr),
        eval_lr=optional_float(raw, "eval_lr", defaults.eval_lr),
        weight_lr=optional_float(raw, "weight_lr", defaults.weight_lr),
        min_lr=optional_float(raw, "min_lr", defaults.min_lr),
    )


def dump_mnist(spec: object) -> dict[str, object]:
    """YAML document for an MNIST spec. Omits `kind` because it is the default."""
    if not isinstance(spec, ExperimentSpec):
        raise TypeError("dump_mnist expects an ExperimentSpec")
    return {
        "name": spec.name,
        "seed": spec.seed,
        "regime": spec.regime,
        "init": spec.init,
        "hidden": list(spec.hidden),
        "epochs": spec.epochs,
        "infer_steps": spec.infer_steps,
        "eval_steps": spec.eval_steps,
        "learn_precision": spec.learn_precision,
        "min_log_var": spec.min_log_var,
        "image_weight": spec.image_weight,
        "infer_lr": spec.infer_lr,
        "eval_lr": spec.eval_lr,
        "weight_lr": spec.weight_lr,
        "min_lr": spec.min_lr,
    }


def training_parser() -> argparse.ArgumentParser:
    """Flags for one training run. The experiment runner builds the same arguments."""
    defaults = PCConfig()
    parser = argparse.ArgumentParser(description="Train a predictive-coding graph on MNIST.")
    parser.add_argument("--regime", choices=sorted(REGIMES), default="forward")
    parser.add_argument("--init", choices=["forward", "leapfrog", "skip2"], default="forward")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--infer-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learn-precision", action="store_true")
    parser.add_argument("--min-log-var", type=float, default=-4.0)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--infer-lr", type=float, default=defaults.infer_lr)
    parser.add_argument("--eval-lr", type=float, default=defaults.eval_lr)
    parser.add_argument("--weight-lr", type=float, default=defaults.weight_lr)
    parser.add_argument("--min-lr", type=float, default=defaults.min_lr)
    return parser


def parse_cli(argv: Sequence[str] | None = None) -> ExperimentSpec:
    """Build a spec from the main.py flags. `argv` defaults to the process arguments."""
    args = training_parser().parse_args(None if argv is None else list(argv))
    return spec_from_namespace(args, name=_CLI_NAME)


def spec_from_namespace(args: argparse.Namespace, name: str) -> ExperimentSpec:
    """Build a spec from parsed training flags."""
    hidden = tuple(args.hidden)
    return ExperimentSpec(
        name=name,
        seed=args.seed,
        regime=_regime(args.regime),
        init=_init(args.init),
        hidden=hidden,
        epochs=args.epochs,
        infer_steps=args.infer_steps,
        eval_steps=args.eval_steps,
        learn_precision=args.learn_precision,
        min_log_var=args.min_log_var,
        image_weight=args.image_weight,
        infer_lr=args.infer_lr,
        eval_lr=args.eval_lr,
        weight_lr=args.weight_lr,
        min_lr=args.min_lr,
    )


def training_argv(spec: ExperimentSpec) -> list[str]:
    """The main.py command that reproduces this spec."""
    argv = [
        "uv",
        "run",
        "main.py",
        "--regime",
        spec.regime,
        "--init",
        spec.init,
        "--hidden",
        *[str(width) for width in spec.hidden],
        "--epochs",
        str(spec.epochs),
        "--infer-steps",
        str(spec.infer_steps),
        "--eval-steps",
        str(spec.eval_steps),
        "--seed",
        str(spec.seed),
        "--min-log-var",
        _flag_float(spec.min_log_var),
        "--image-weight",
        _flag_float(spec.image_weight),
        "--infer-lr",
        _flag_float(spec.infer_lr),
        "--eval-lr",
        _flag_float(spec.eval_lr),
        "--weight-lr",
        _flag_float(spec.weight_lr),
        "--min-lr",
        _flag_float(spec.min_lr),
    ]
    if spec.learn_precision:
        argv.append("--learn-precision")
    return argv


def argv_mnist(spec: object) -> list[str]:
    """Kind registry wrapper around `training_argv`."""
    if not isinstance(spec, ExperimentSpec):
        raise TypeError("argv_mnist expects an ExperimentSpec")
    return training_argv(spec)


def train_mnist(spec: ExperimentSpec, directory: Path | None = None) -> list[EpochReport]:
    """Train one spec on MNIST through Trainer and return the metrics it logs.

    `directory` is accepted so every kind shares one trainer signature; the Trainer
    metrics are this kind's whole result, so nothing is written there.
    """
    torch.manual_seed(spec.seed)
    # 256 is the batch size the command-line entry point has always used.
    train_loader, test_loader = load_dataset(batch_size=256)
    graph = mlp_graph(
        hidden_dims=spec.hidden,
        init=spec.init,
        learn_precision=spec.learn_precision,
        min_log_var=spec.min_log_var,
        image_weight=spec.image_weight,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    regime = REGIMES[spec.regime](graph, spec.pc_config(), device=device)
    return Trainer(regime, train_loader, test_loader).train(epochs=spec.epochs)


def _hidden(value: object) -> tuple[int, ...]:
    """Positive hidden widths."""
    if not isinstance(value, list) or not value:
        raise ValueError("hidden must be a non-empty list of widths")
    return tuple(at_least(as_int(width, "hidden"), "hidden", 1) for width in value)


def _regime(value: object) -> RegimeName:
    """One of the training regimes main.py accepts."""
    if value == "zero":
        return "zero"
    if value == "forward":
        return "forward"
    if value == "balanced":
        return "balanced"
    raise ValueError("regime must be zero, forward, or balanced")


def _init(value: object) -> InitScheme:
    """One of the init schemes mlp_graph accepts."""
    if value == "forward":
        return "forward"
    if value == "leapfrog":
        return "leapfrog"
    if value == "skip2":
        return "skip2"
    raise ValueError("init must be forward, leapfrog, or skip2")


def _flag_float(value: float) -> str:
    """Format a float so argparse parses it back to the same value."""
    return format(value, ".12g")


KIND = register(
    Kind(
        name="mnist",
        parse=parse_mnist,
        dump=dump_mnist,
        run=train_mnist,
        argv=argv_mnist,
    )
)
