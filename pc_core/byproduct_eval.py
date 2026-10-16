"""Read out what the trained hub does: classify, describe, and imagine, per clue type.

Everything here is reported per type and never pooled, and every accuracy is paired
with the image-only accuracy on the same images, so a type that only exists for some
digits is compared against the right baseline.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from pc_core.byproduct import ByproductRegime
from pc_core.messages import (
    KINDS,
    Clue,
    ClueSampler,
    Kind,
    SamplerConfig,
    bayes_posterior,
    encode,
    enumerate_clues,
    kind_available,
    kind_information,
    nearest_clue,
)
from pc_core.reference import ReferenceClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen before this import)

GRID_ROWS = 8


@dataclass(frozen=True)
class KindResult:
    """Everything measured for one clue type, in percent, on the images that can carry it."""

    kind: str
    count: int
    bits: float
    bayes_clue_only: float
    image_only: float
    image_true: float
    image_shuffled: float
    clue_only: float
    ceiling: float
    reference_image_only: float
    imagine_plain: float
    imagine_committed: float


@dataclass
class _Tally:
    """Running correct counts for one type."""

    count: int = 0
    image_only: int = 0
    image_true: int = 0
    image_shuffled: int = 0
    clue_only: int = 0
    ceiling: int = 0
    reference_image_only: int = 0
    imagine_plain: int = 0
    imagine_committed: int = 0

    def percent(self, field: str) -> float:
        """One count as a percentage of the images this type was scored on."""
        return 100.0 * getattr(self, field) / max(self.count, 1)


@dataclass(frozen=True)
class Evaluation:
    """The full read-out of one trained hub."""

    image_only: float
    reference_clean: float
    reference_degraded: float
    eval_curve: dict[int, float]
    decoded_true: float
    volunteered: dict[str, int]
    imagine_labels: list[int]
    kinds: tuple[KindResult, ...]

    def document(self) -> dict[str, object]:
        """A YAML-ready mapping."""
        return {
            "image_only_accuracy": self.image_only,
            "reference_clean_accuracy": self.reference_clean,
            "reference_degraded_accuracy": self.reference_degraded,
            "eval_curve": {int(steps): accuracy for steps, accuracy in self.eval_curve.items()},
            "think": {"decoded_true_percent": self.decoded_true, "volunteered": dict(self.volunteered)},
            "imagine_labels": list(self.imagine_labels),
            "kinds": [asdict(result) for result in self.kinds],
        }


def evaluate_hub(
    regime: ByproductRegime,
    loader: DataLoader,
    reference_clean: ReferenceClassifier,
    reference_degraded: ReferenceClassifier,
    prior: Tensor,
    sampler_config: SamplerConfig,
    confusion: Tensor,
    directory: Path,
    eval_curve: Sequence[int],
    thoughts: int,
    reference_clean_accuracy: float,
    reference_degraded_accuracy: float,
    seed: int = 0,
) -> Evaluation:
    """Measure, write grids, thoughts, and the gain plot into `directory`, and return the numbers."""
    regime.model.eval()
    sampler = ClueSampler(sampler_config, confusion, seed=seed)
    clues = enumerate_clues()
    tallies = {kind: _Tally() for kind in KINDS}
    grids: dict[Kind, list[tuple[str, Tensor, Tensor]]] = {kind: [] for kind in KINDS}
    thought_lines: list[str] = []
    volunteered: Counter[str] = Counter()
    decoded_true = 0
    total = 0
    # Settling needs gradients on the states, so this loop is not under no_grad.
    for images, labels in loader:
        images = images.to(regime.device)
        labels = labels.to(regime.device)
        states = regime.infer(regime.clamp(images, None, None, "image"))
        predicted = states[regime.label_node].argmax(dim=1)
        decoded = nearest_clue(states[regime.msg_node], clues)
        for row, (clue, label) in enumerate(zip(decoded, labels.tolist())):
            volunteered[clue.kind] += 1
            decoded_true += int(clue.holds(label))
            if total + row < thoughts:
                thought_lines.append(_thought(states[regime.msg_node][row], label, int(predicted[row]), clues))
        total += labels.shape[0]
        for kind in KINDS:
            _tally_kind(
                kind, regime, images, labels, predicted, sampler, sampler_config, confusion,
                reference_clean, reference_degraded, tallies[kind], grids[kind],
            )
    (directory / "thoughts.txt").write_text("\n".join(thought_lines) + "\n", encoding="utf-8")
    for kind in KINDS:
        _save_grid(directory / f"imagine-{kind.replace(' ', '-')}.png", kind, grids[kind])
    imagine_labels = _imagine_labels(regime, reference_clean, directory / "imagine-labels.png")
    results = tuple(
        KindResult(
            kind=kind,
            count=tally.count,
            bits=kind_information(kind, prior, sampler_config, confusion).bits,
            bayes_clue_only=100.0 * kind_information(kind, prior, sampler_config, confusion).bayes_accuracy,
            image_only=tally.percent("image_only"),
            image_true=tally.percent("image_true"),
            image_shuffled=tally.percent("image_shuffled"),
            clue_only=tally.percent("clue_only"),
            ceiling=tally.percent("ceiling"),
            reference_image_only=tally.percent("reference_image_only"),
            imagine_plain=tally.percent("imagine_plain"),
            imagine_committed=tally.percent("imagine_committed"),
        )
        for kind, tally in tallies.items()
    )
    evaluation = Evaluation(
        image_only=_image_only_accuracy(regime, loader, regime.config.eval_steps),
        reference_clean=reference_clean_accuracy,
        reference_degraded=reference_degraded_accuracy,
        eval_curve={steps: _image_only_accuracy(regime, loader, steps) for steps in eval_curve},
        decoded_true=100.0 * decoded_true / max(total, 1),
        volunteered=dict(volunteered),
        imagine_labels=imagine_labels,
        kinds=results,
    )
    plot_gain(evaluation, directory / "gain.png", regime.degradation.describe())
    return evaluation


def _tally_kind(
    kind: Kind,
    regime: ByproductRegime,
    images: Tensor,
    labels: Tensor,
    predicted: Tensor,
    sampler: ClueSampler,
    config: SamplerConfig,
    confusion: Tensor,
    reference_clean: ReferenceClassifier,
    reference_degraded: ReferenceClassifier,
    tally: _Tally,
    grid: list[tuple[str, Tensor, Tensor]],
) -> None:
    """Score one type on the images of this batch that can carry it."""
    mask = torch.tensor([kind_available(kind, label) for label in labels.tolist()], device=labels.device)
    if int(mask.sum()) < 2:
        return
    sub_images = images[mask]
    sub_labels = labels[mask]
    true_clues = [sampler.sample_kind(kind, label) for label in sub_labels.tolist()]
    shuffled = true_clues[1:] + true_clues[:1]
    with_true = regime.infer(regime.clamp(sub_images, None, true_clues, "both"))
    with_shuffled = regime.infer(regime.clamp(sub_images, None, shuffled, "both"))
    clue_only = regime.infer(regime.clamp(sub_images, None, true_clues, "msg"))
    committed_labels = clue_only[regime.label_node].argmax(dim=1)
    committed = regime.infer(
        {**regime.clamp(sub_images, None, true_clues, "msg"), regime.label_node: F.one_hot(committed_labels, 10).float()}
    )
    reference_posterior = reference_degraded.posterior(sub_images)
    ceiling = bayes_posterior(reference_posterior, true_clues, config, confusion).argmax(dim=1)
    plain_guess = reference_clean.posterior(clue_only[regime.input_node]).argmax(dim=1)
    committed_guess = reference_clean.posterior(committed[regime.input_node]).argmax(dim=1)
    allowed = torch.tensor(
        [[digit in clue.allowed for digit in range(10)] for clue in true_clues], device=labels.device
    )
    tally.count += int(mask.sum())
    tally.image_only += int((predicted[mask] == sub_labels).sum())
    tally.image_true += int((with_true[regime.label_node].argmax(dim=1) == sub_labels).sum())
    tally.image_shuffled += int((with_shuffled[regime.label_node].argmax(dim=1) == sub_labels).sum())
    tally.clue_only += int((committed_labels == sub_labels).sum())
    tally.ceiling += int((ceiling == sub_labels).sum())
    tally.reference_image_only += int((reference_posterior.argmax(dim=1) == sub_labels).sum())
    tally.imagine_plain += int(allowed.gather(1, plain_guess[:, None]).sum())
    tally.imagine_committed += int(allowed.gather(1, committed_guess[:, None]).sum())
    seen = {text for text, _, _ in grid}
    for row, clue in enumerate(true_clues):
        if len(grid) >= GRID_ROWS:
            break
        if clue.text in seen:
            continue
        seen.add(clue.text)
        grid.append((clue.text, clue_only[regime.input_node][row].cpu(), committed[regime.input_node][row].cpu()))


def _image_only_accuracy(regime: ByproductRegime, loader: DataLoader, steps: int) -> float:
    """Percent correct with only the image clamped, settled for `steps`."""
    correct = 0
    total = 0
    for images, labels in loader:
        states = regime.infer(regime.clamp(images.to(regime.device), None, None, "image"), steps)
        correct += int((states[regime.label_node].argmax(dim=1).cpu() == labels).sum())
        total += labels.shape[0]
    return 100.0 * correct / max(total, 1)


def _thought(msg: Tensor, label: int, predicted: int, clues: Sequence[Clue]) -> str:
    """One line: the true and read-off label, then the five nearest statements with their distances."""
    distances = torch.cdist(msg[None, :], encode(clues).to(msg.device))[0]
    order = distances.argsort()[:5].tolist()
    ranked = ", ".join(
        f"{clues[index].text} ({distances[index]:.2f}{'' if clues[index].holds(label) else ' false'})"
        for index in order
    )
    return f"label {label} read {predicted}: {ranked}"


def _imagine_labels(regime: ByproductRegime, reference: ReferenceClassifier, path: Path) -> list[int]:
    """Clamp each label alone, read off the image, save the row of prototypes, and return what the reference calls them."""
    labels = torch.arange(10, device=regime.device)
    states = regime.infer({regime.label_node: F.one_hot(labels, 10).float()})
    images = states[regime.input_node]
    guesses = reference.posterior(images).argmax(dim=1).tolist()
    figure, axes = plt.subplots(1, 10, figsize=(12, 1.6))
    for digit, axis in enumerate(axes):
        axis.imshow(images[digit].cpu().reshape(28, 28).numpy(), cmap="gray")
        axis.set_title(f"{digit} -> {guesses[digit]}", fontsize=8)
        axis.axis("off")
    figure.suptitle("label clamped, image read off; title is label -> reference guess", fontsize=9)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return guesses


def _save_grid(path: Path, kind: str, rows: Sequence[tuple[str, Tensor, Tensor]]) -> None:
    """Rows of imagined images for one type: the clue, plain imagination, committed imagination."""
    if not rows:
        return
    figure, axes = plt.subplots(len(rows), 2, figsize=(3.2, 1.5 * len(rows)), squeeze=False)
    for row, (text, plain, committed) in enumerate(rows):
        axes[row, 0].imshow(plain.reshape(28, 28).numpy(), cmap="gray")
        axes[row, 1].imshow(committed.reshape(28, 28).numpy(), cmap="gray")
        axes[row, 0].set_ylabel(text, fontsize=7, rotation=0, labelpad=45, va="center")
        for column in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_title(f"{kind}: plain", fontsize=8)
    axes[0, 1].set_title("committed", fontsize=8)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_gain(evaluation: Evaluation, path: Path, degradation: str) -> None:
    """Gain over image-only against bits per type, with the ceiling and the shuffled control;
    below it, clue-only accuracy against the clue-only Bayes accuracy on the same x axis."""
    figure, (gain, alone) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, height_ratios=[3, 2])
    for result in evaluation.kinds:
        base = result.image_only
        gain.scatter(result.bits, result.image_true - base, color="C0", zorder=3)
        gain.scatter(result.bits, result.image_shuffled - base, facecolors="none", edgecolors="C3", zorder=3)
        gain.scatter(result.bits, result.ceiling - result.reference_image_only, marker="_", color="C2", s=120, zorder=2)
        gain.annotate(result.kind, (result.bits, result.image_true - base), fontsize=7, xytext=(3, 3), textcoords="offset points")
        alone.scatter(result.bits, result.clue_only, color="C0", zorder=3)
        alone.scatter(result.bits, result.bayes_clue_only, marker="x", color="C7", zorder=2)
        alone.annotate(result.kind, (result.bits, result.clue_only), fontsize=7, xytext=(3, 3), textcoords="offset points")
    gain.axhline(0.0, color="black", linewidth=0.8)
    gain.scatter([], [], color="C0", label="image + true clue")
    gain.scatter([], [], facecolors="none", edgecolors="C3", label="image + shuffled clue")
    gain.scatter([], [], marker="_", color="C2", s=120, label="ceiling: reference posterior x clue, minus reference image-only")
    gain.set_ylabel("accuracy minus image-only\non the same images (points)")
    gain.set_title(f"{degradation}; image-only {evaluation.image_only:.1f}%")
    gain.legend(fontsize=8, loc="best")
    alone.scatter([], [], color="C0", label="clue only (model)")
    alone.scatter([], [], marker="x", color="C7", label="clue-only Bayes")
    alone.set_xlabel("bits per clue type (empirical training prior)")
    alone.set_ylabel("clue-only accuracy (%)")
    alone.legend(fontsize=8, loc="best")
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
