"""Clues about a digit: grammar, sampling, encoding, information, and image degradation.

A clue is a short true statement about the label, rendered as one-hot tokens. Nothing
here touches a network; the module exists so the training regime and the evaluation
draw clues the same way and measure them with the same arithmetic.
"""

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations
from typing import Literal

import torch
from torch import Tensor

from pc_core.dataset import _MEAN, _STD

DIGITS: tuple[int, ...] = tuple(range(10))
WORDS: tuple[str, ...] = (
    "is", "not", "even", "odd", "less", "more", "than", "maybe", "or", "between", "and", "pad",
)
VOCAB: tuple[str, ...] = tuple(str(digit) for digit in DIGITS) + WORDS
PAD = "pad"
MAX_LEN = 5
MESSAGE_DIM = MAX_LEN * len(VOCAB)
BACKGROUND = (0.0 - _MEAN) / _STD

Kind = Literal[
    "is", "not is", "even", "not even", "odd", "not odd",
    "less than", "not less than", "more than", "not more than",
    "or", "not or", "between", "not between", "maybe",
]
KINDS: tuple[Kind, ...] = (
    "is", "not is", "even", "not even", "odd", "not odd",
    "less than", "not less than", "more than", "not more than",
    "or", "not or", "between", "not between", "maybe",
)
Family = Literal["is", "parity", "less", "more", "or", "between", "maybe"]
FAMILIES: tuple[Family, ...] = ("is", "parity", "less", "more", "or", "between", "maybe")

_ALL = frozenset(DIGITS)
_EVEN = frozenset(digit for digit in DIGITS if digit % 2 == 0)
_ODD = _ALL - _EVEN
_TOKEN_INDEX = {token: index for index, token in enumerate(VOCAB)}


@dataclass(frozen=True)
class Clue:
    """One statement about a digit, the digits it allows, and its reported type."""

    kind: Kind
    tokens: tuple[str, ...]
    allowed: frozenset[int]

    @property
    def text(self) -> str:
        """The statement as words."""
        return " ".join(self.tokens)

    def holds(self, label: int) -> bool:
        """Whether the statement is true of `label`. `maybe` has no truth value and always holds."""
        return label in self.allowed


def _clue(kind: Kind, tokens: Iterable[str | int], allowed: Iterable[int]) -> Clue:
    """Build a clue from tokens that may be digits."""
    return Clue(kind, tuple(str(token) for token in tokens), frozenset(allowed))


def _between_pairs() -> list[tuple[int, int]]:
    """Every inclusive range a < b except 0..9, which is always true."""
    return [(low, high) for low, high in combinations(DIGITS, 2) if (low, high) != (0, 9)]


def _or_pairs() -> list[tuple[int, int]]:
    """Every unordered pair of distinct digits."""
    return list(combinations(DIGITS, 2))


def _is(k: int) -> Clue:
    return _clue("is", ("is", k), {k})


def _not_is(k: int) -> Clue:
    return _clue("not is", ("not", "is", k), _ALL - {k})


def _parity(kind: Kind) -> Clue:
    tokens = {"even": ("even",), "odd": ("odd",), "not even": ("not", "even"), "not odd": ("not", "odd")}
    allowed = {"even": _EVEN, "odd": _ODD, "not even": _ODD, "not odd": _EVEN}
    return _clue(kind, tokens[kind], allowed[kind])


def _less(k: int) -> Clue:
    return _clue("less than", ("less", "than", k), range(0, k))


def _not_less(k: int) -> Clue:
    return _clue("not less than", ("not", "less", "than", k), range(k, 10))


def _more(k: int) -> Clue:
    return _clue("more than", ("more", "than", k), range(k + 1, 10))


def _not_more(k: int) -> Clue:
    return _clue("not more than", ("not", "more", "than", k), range(0, k + 1))


def _or(k: int, j: int) -> Clue:
    low, high = sorted((k, j))
    return _clue("or", (low, "or", high), {low, high})


def _not_or(k: int, j: int) -> Clue:
    low, high = sorted((k, j))
    return _clue("not or", ("not", low, "or", high), _ALL - {low, high})


def _between(low: int, high: int) -> Clue:
    return _clue("between", ("between", low, "and", high), range(low, high + 1))


def _not_between(low: int, high: int) -> Clue:
    return _clue("not between", ("not", "between", low, "and", high), _ALL - frozenset(range(low, high + 1)))


def _maybe(k: int) -> Clue:
    return _clue("maybe", ("maybe", k), _ALL)


@cache
def enumerate_clues() -> tuple[Clue, ...]:
    """Every statement the grammar can make that is neither always true nor always false."""
    clues: list[Clue] = []
    clues += [_is(k) for k in DIGITS]
    clues += [_not_is(k) for k in DIGITS]
    clues += [_parity(kind) for kind in ("even", "not even", "odd", "not odd")]
    clues += [_less(k) for k in range(1, 10)]
    clues += [_not_less(k) for k in range(1, 10)]
    clues += [_more(k) for k in range(0, 9)]
    clues += [_not_more(k) for k in range(0, 9)]
    clues += [_or(k, j) for k, j in _or_pairs()]
    clues += [_not_or(k, j) for k, j in _or_pairs()]
    clues += [_between(low, high) for low, high in _between_pairs()]
    clues += [_not_between(low, high) for low, high in _between_pairs()]
    clues += [_maybe(k) for k in DIGITS]
    return tuple(clues)


@cache
def clues_of_kind(kind: Kind) -> tuple[Clue, ...]:
    """Every statement of one reported type."""
    return tuple(clue for clue in enumerate_clues() if clue.kind == kind)


def kind_available(kind: Kind, label: int) -> bool:
    """Whether a true statement of this type exists for `label`."""
    return any(clue.holds(label) for clue in clues_of_kind(kind))


@dataclass(frozen=True)
class SamplerConfig:
    """How clues are drawn during training.

    `family_weights` picks a statement family. Where a family has a fixed parameter
    (`less`, `more`, `between`) it is drawn without looking at the label and `not` is
    prepended when the statement is false, so a clue's likelihood is the same for every
    digit it allows. For `is`, `or` and parity, `positive` is the chance of the plain
    form; the likelihood is still constant within the allowed set.
    """

    family_weights: Mapping[Family, float] = field(
        default_factory=lambda: {family: 1.0 for family in FAMILIES}
    )
    positive: float = 0.5
    maybe_true: float = 0.6


class ClueSampler:
    """Draw true clues for labels, by family during training or by type for evaluation."""

    def __init__(self, config: SamplerConfig, confusion: Tensor, seed: int = 0) -> None:
        if confusion.shape != (10, 10):
            raise ValueError("confusion must be 10 x 10")
        self.config = config
        # Rows: the true digit; columns: how often an image-only reader calls it each other digit.
        self.confusion = _normalise_off_diagonal(confusion)
        self.random = random.Random(seed)
        self._families = [family for family, weight in config.family_weights.items() if weight > 0]
        self._family_weights = [config.family_weights[family] for family in self._families]
        self._between = _between_pairs()
        self._or = _or_pairs()

    def sample(self, labels: Sequence[int]) -> list[Clue]:
        """One clue per label, drawn by family weight."""
        return [self.sample_family(self._draw_family(), int(label)) for label in labels]

    def sample_family(self, family: Family, label: int) -> Clue:
        """One clue from a family. The family decides the sign of the statement."""
        rng = self.random
        positive = rng.random() < self.config.positive
        if family == "is":
            return _is(label) if positive else _not_is(rng.choice([k for k in DIGITS if k != label]))
        if family == "parity":
            if label % 2 == 0:
                return _parity("even") if positive else _parity("not odd")
            return _parity("odd") if positive else _parity("not even")
        if family == "less":
            k = rng.randint(1, 9)
            return _less(k) if label < k else _not_less(k)
        if family == "more":
            k = rng.randint(0, 8)
            return _more(k) if label > k else _not_more(k)
        if family == "or":
            if positive:
                return _or(label, rng.choice([k for k in DIGITS if k != label]))
            return _not_or(*rng.choice([pair for pair in self._or if label not in pair]))
        if family == "between":
            low, high = rng.choice(self._between)
            return _between(low, high) if low <= label <= high else _not_between(low, high)
        if family == "maybe":
            return self._maybe(label)
        raise ValueError(f"unknown family {family!r}")

    def sample_kind(self, kind: Kind, label: int) -> Clue:
        """One true clue of a given type, for evaluating every type on every image."""
        if kind == "maybe":
            return self._maybe(label)
        options = [clue for clue in clues_of_kind(kind) if clue.holds(label)]
        if not options:
            raise ValueError(f"no true {kind!r} statement exists for {label}")
        return self.random.choice(options)

    def _maybe(self, label: int) -> Clue:
        """Name the label with probability `maybe_true`, otherwise a digit confused with it."""
        if self.random.random() < self.config.maybe_true:
            return _maybe(label)
        weights = self.confusion[label].tolist()
        return _maybe(self.random.choices(DIGITS, weights=weights)[0])

    def _draw_family(self) -> Family:
        return self.random.choices(self._families, weights=self._family_weights)[0]


def _normalise_off_diagonal(confusion: Tensor) -> Tensor:
    """Zero the diagonal and make each row a distribution over the other digits."""
    off = confusion.detach().clone().double()
    off.fill_diagonal_(0.0)
    totals = off.sum(dim=1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("every digit needs at least one confusion to draw a wrong digit from")
    return (off / totals).float()


def encode(clues: Sequence[Clue]) -> Tensor:
    """One-hot tokens per position, padded to `MAX_LEN`, flattened to `MESSAGE_DIM`."""
    vectors = torch.zeros(len(clues), MAX_LEN, len(VOCAB))
    for row, clue in enumerate(clues):
        if len(clue.tokens) > MAX_LEN:
            raise ValueError(f"clue {clue.text!r} is longer than {MAX_LEN} tokens")
        padded = clue.tokens + (PAD,) * (MAX_LEN - len(clue.tokens))
        for position, token in enumerate(padded):
            vectors[row, position, _TOKEN_INDEX[token]] = 1.0
    return vectors.flatten(1)


def decode(vector: Tensor) -> tuple[str, ...]:
    """The most likely token at each position, with padding stripped."""
    if vector.shape != (MESSAGE_DIM,):
        raise ValueError(f"expected a vector of length {MESSAGE_DIM}")
    indices = vector.reshape(MAX_LEN, len(VOCAB)).argmax(dim=1).tolist()
    return tuple(VOCAB[index] for index in indices if VOCAB[index] != PAD)


def nearest_clue(vectors: Tensor, clues: Sequence[Clue]) -> list[Clue]:
    """The valid clue closest to each row of `vectors` in squared distance."""
    codes = encode(clues).to(vectors.device)
    distances = torch.cdist(vectors, codes)
    return [clues[index] for index in distances.argmin(dim=1).tolist()]


def label_prior(labels: Tensor) -> Tensor:
    """The empirical label distribution."""
    counts = torch.bincount(labels.long().cpu(), minlength=10).double()
    return (counts / counts.sum()).float()


def entropy_bits(distribution: Tensor) -> float:
    """Shannon entropy in bits; zero entries contribute nothing."""
    p = distribution.double()
    p = p[p > 0]
    return float(-(p * p.log2()).sum().item())


def clue_bits(clue: Clue, prior: Tensor) -> float:
    """Information in one truth-valued clue: entropy of the prior minus that of the prior restricted to the allowed digits."""
    if clue.kind == "maybe":
        raise ValueError("maybe has no allowed set; use type_bits with a likelihood table")
    restricted = prior.clone()
    for digit in DIGITS:
        if digit not in clue.allowed:
            restricted[digit] = 0.0
    return entropy_bits(prior) - entropy_bits(restricted / restricted.sum())


def likelihood_table(kind: Kind, clues: Sequence[Clue], config: SamplerConfig, confusion: Tensor) -> Tensor:
    """P(clue | digit) for each clue of one type, up to a factor shared by the type.

    Truth-valued types are indicator rows, which is exact for the family sampler.
    `maybe k` is the chance of naming `k` given the digit, through the confusion rows.
    """
    if any(clue.kind != kind for clue in clues):
        raise ValueError("every clue must be of the requested kind")
    if kind == "maybe":
        off = _normalise_off_diagonal(confusion)
        table = torch.zeros(len(clues), 10)
        for row, clue in enumerate(clues):
            k = int(clue.tokens[1])
            for digit in DIGITS:
                table[row, digit] = (config.maybe_true if k == digit else 0.0) + (1 - config.maybe_true) * off[digit, k]
        return table
    table = torch.zeros(len(clues), 10)
    for row, clue in enumerate(clues):
        for digit in clue.allowed:
            table[row, digit] = 1.0
    return table


@dataclass(frozen=True)
class KindInformation:
    """What one clue type is worth on its own: bits carried and clue-only Bayes accuracy."""

    kind: Kind
    bits: float
    bayes_accuracy: float


def kind_information(kind: Kind, prior: Tensor, config: SamplerConfig, confusion: Tensor) -> KindInformation:
    """Bits and clue-only Bayes accuracy for one type, averaged over the clues the sampler draws."""
    clues = clues_of_kind(kind)
    table = likelihood_table(kind, clues, config, confusion).double()
    joint = prior.double()[None, :] * table
    clue_mass = joint.sum(dim=1)
    posterior = joint / clue_mass[:, None]
    weights = clue_mass / clue_mass.sum()
    expected_entropy = sum(
        float(weight) * entropy_bits(posterior[row]) for row, weight in enumerate(weights)
    )
    bayes = float((weights * posterior.max(dim=1).values).sum().item())
    return KindInformation(kind, entropy_bits(prior) - expected_entropy, bayes)


def bayes_posterior(image_posterior: Tensor, clues: Sequence[Clue], config: SamplerConfig, confusion: Tensor) -> Tensor:
    """Combine an image-only posterior with clues: multiply by each clue's likelihood row and renormalise.

    For truth-valued clues this zeroes the digits the clue rules out.
    """
    if image_posterior.shape[0] != len(clues):
        raise ValueError("one clue per posterior row")
    rows = torch.stack(
        [likelihood_table(clue.kind, [clue], config, confusion)[0] for clue in clues]
    ).to(image_posterior.device)
    combined = image_posterior * rows
    return combined / combined.sum(dim=1, keepdim=True).clamp_min(1e-12)


DegradationKind = Literal["none", "noise", "occlude"]


@dataclass(frozen=True)
class Degradation:
    """Remove information from flat normalised images so a clue has something to add.

    `noise` adds Gaussian noise with standard deviation `level`, in normalised units.
    `occlude` sets a random square of side `level` pixels to the background value.
    """

    kind: DegradationKind = "none"
    level: float = 0.0

    def apply(self, images: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """A degraded copy of `images`; labels are the caller's and untouched."""
        if self.kind == "none":
            return images
        if self.kind == "noise":
            noise = torch.randn(images.shape, generator=generator, device=images.device)
            return images + self.level * noise
        if self.kind == "occlude":
            return _occlude(images, int(self.level), generator)
        raise ValueError(f"unknown degradation {self.kind!r}")

    def describe(self) -> str:
        """A short label for logs and plots."""
        if self.kind == "none":
            return "clean"
        if self.kind == "noise":
            return f"noise sigma={self.level:g}"
        return f"occlude {int(self.level)}x{int(self.level)}"


def _occlude(images: Tensor, size: int, generator: torch.Generator | None) -> Tensor:
    """Set one random `size` x `size` patch per image to the background value."""
    if not 1 <= size <= 28:
        raise ValueError("occlusion size must be between 1 and 28 pixels")
    batch = images.shape[0]
    grid = images.reshape(batch, 28, 28).clone()
    tops = torch.randint(0, 28 - size + 1, (batch,), generator=generator, device=images.device)
    lefts = torch.randint(0, 28 - size + 1, (batch,), generator=generator, device=images.device)
    rows = torch.arange(28, device=images.device)
    in_rows = (rows[None, :] >= tops[:, None]) & (rows[None, :] < tops[:, None] + size)
    in_cols = (rows[None, :] >= lefts[:, None]) & (rows[None, :] < lefts[:, None] + size)
    mask = in_rows[:, :, None] & in_cols[:, None, :]
    grid[mask] = BACKGROUND
    return grid.flatten(1)


def uniform_prior() -> Tensor:
    """One tenth per digit."""
    return torch.full((10,), 0.1)


def uniform_confusion() -> Tensor:
    """Every wrong digit equally likely; a stand-in when no classifier has been measured."""
    return torch.ones(10, 10) - torch.eye(10)
