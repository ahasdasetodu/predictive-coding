from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from pc_core.experiment.yamlio import (
    as_float,
    as_int,
    at_least,
    optional_float,
    reject_unknown,
    render_yaml,
    require,
    string_key_mapping,
)
from pc_core.messages import FAMILIES, SamplerConfig
from pc_core.reference import ReferenceConfig
from pc_core.trainer import EpochReport

_SAMPLER_KEYS = {"positive", "maybe_true", "family_weights"}
_REFERENCE_KEYS = {"hidden", "epochs", "lr"}
_DEFAULT_KIND = "mnist"


class Spec(Protocol):
    """A parsed experiment spec of any kind."""

    kind: str
    name: str
    seed: int


@dataclass(frozen=True)
class Kind:
    """One experiment kind: parse YAML, dump it, run it, reproduce it."""

    name: str
    parse: Callable[[Mapping[str, object]], Spec]
    dump: Callable[[object], dict[str, object]]
    run: Callable[..., Sequence[EpochReport]]
    argv: Callable[[object], list[str]]


KINDS: dict[str, Kind] = {}
TRAINERS: dict[str, Callable[..., Sequence[EpochReport]]] = {}


def register(kind: Kind) -> Kind:
    """Add a kind to the parse/dump/run registry."""
    KINDS[kind.name] = kind
    TRAINERS[kind.name] = kind.run
    return kind


def parse_sampler(value: object) -> SamplerConfig:
    """The clue sampler settings, with defaults for omitted keys."""
    raw = string_key_mapping(value, "sampler")
    reject_unknown(raw, _SAMPLER_KEYS, "sampler")
    defaults = SamplerConfig()
    weights = dict(defaults.family_weights)
    if "family_weights" in raw:
        given = string_key_mapping(raw["family_weights"], "family_weights")
        reject_unknown(given, set(FAMILIES), "family_weights")
        weights = {family: as_float(given.get(family, 0.0), family) for family in FAMILIES}
    return SamplerConfig(
        family_weights=weights,
        positive=optional_float(raw, "positive", defaults.positive),
        maybe_true=optional_float(raw, "maybe_true", defaults.maybe_true),
    )


def parse_reference(value: object) -> ReferenceConfig:
    """The reference classifier settings, with defaults for omitted keys."""
    raw = string_key_mapping(value, "reference")
    reject_unknown(raw, _REFERENCE_KEYS, "reference")
    defaults = ReferenceConfig()
    return ReferenceConfig(
        hidden=at_least(as_int(raw.get("hidden", defaults.hidden), "hidden"), "hidden", 1),
        epochs=at_least(as_int(raw.get("epochs", defaults.epochs), "epochs"), "epochs", 1),
        lr=optional_float(raw, "lr", defaults.lr),
    )


def sampler_document(sampler: SamplerConfig) -> dict[str, object]:
    """YAML form of the clue sampler settings."""
    return {
        "positive": sampler.positive,
        "maybe_true": sampler.maybe_true,
        "family_weights": {family: float(weight) for family, weight in sampler.family_weights.items()},
    }


def reference_document(reference: ReferenceConfig) -> dict[str, object]:
    """YAML form of the reference classifier settings."""
    return {"hidden": reference.hidden, "epochs": reference.epochs, "lr": reference.lr}


def runner_argv(spec: Spec) -> list[str]:
    """The module command that reproduces a non-MNIST spec."""
    return ["uv", "run", "python", "-m", "pc_core.experiment", f"experiments/specs/{spec.name}.yaml"]


def load_spec(path: Path) -> Spec:
    """Load a spec file and reject unknown keys."""
    return parse_spec(path.read_text(encoding="utf-8"))


def parse_spec(text: str) -> Spec:
    """Parse a spec document of any kind. A spec without `kind` is an MNIST training run."""
    raw = string_key_mapping(yaml.safe_load(text), "spec")
    _ensure_kinds()
    name = raw.get("kind", _DEFAULT_KIND)
    kind = KINDS.get(name) if isinstance(name, str) else None
    if kind is None:
        raise ValueError("kind must be mnist, clue-bits, or byproduct")
    return kind.parse(raw)


def dump_spec(spec: Spec) -> str:
    """Serialize a resolved spec. Parsing the text returns the same spec."""
    kind = _kind_of(spec)
    return render_yaml(kind.dump(spec))


def spec_argv(spec: Spec) -> list[str]:
    """The command that reproduces this spec: main.py flags for MNIST runs, the runner otherwise."""
    return _kind_of(spec).argv(spec)


def _kind_of(spec: Spec) -> Kind:
    """The registered kind for `spec`, after kinds have been imported."""
    _ensure_kinds()
    kind = KINDS.get(spec.kind)
    if kind is None:
        raise ValueError("kind must be mnist, clue-bits, or byproduct")
    return kind


def _ensure_kinds() -> None:
    """Import kind modules so they can register themselves."""
    if KINDS:
        return
    from pc_core.experiment import byproduct, clue_bits, mnist

    _ = (byproduct, clue_bits, mnist)
