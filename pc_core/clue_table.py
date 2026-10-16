"""Measure what each clue type is worth on the MNIST training labels.

The table is a result: bits per type under the empirical label prior and the accuracy a
clue-only Bayes reader reaches. `maybe k` needs the confusion rows of an image-only
reader, so a reference classifier is trained on clean images first.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pc_core.messages import (
    KINDS,
    Degradation,
    KindInformation,
    SamplerConfig,
    clues_of_kind,
    entropy_bits,
    kind_information,
    label_prior,
    uniform_prior,
)
from pc_core.reference import ReferenceConfig, clean_reference, reference_accuracy


@dataclass(frozen=True)
class ClueTable:
    """Per-type information on the training prior, with the inputs that produced it."""

    prior: Tensor
    confusion: Tensor
    reference_accuracy: float
    rows: tuple[KindInformation, ...]
    uniform_rows: tuple[KindInformation, ...]

    def document(self) -> dict[str, object]:
        """A YAML-ready mapping."""
        return {
            "prior": [round(float(p), 6) for p in self.prior],
            "prior_entropy_bits": entropy_bits(self.prior),
            "reference_accuracy": self.reference_accuracy,
            "confusion": [[int(count) for count in row] for row in self.confusion],
            "kinds": [
                {
                    "kind": row.kind,
                    "statements": len(clues_of_kind(row.kind)),
                    "bits": row.bits,
                    "bayes_accuracy": 100.0 * row.bayes_accuracy,
                    "uniform_bits": uniform.bits,
                }
                for row, uniform in zip(self.rows, self.uniform_rows)
            ],
        }

    def render(self) -> str:
        """A fixed-width text table."""
        lines = [f"{'kind':<16}{'n':>5}{'bits':>9}{'uniform':>9}{'bayes %':>9}"]
        for row, uniform in zip(self.rows, self.uniform_rows):
            lines.append(
                f"{row.kind:<16}{len(clues_of_kind(row.kind)):>5}{row.bits:>9.3f}"
                f"{uniform.bits:>9.3f}{100 * row.bayes_accuracy:>9.2f}"
            )
        lines.append(f"prior entropy {entropy_bits(self.prior):.4f} bits; reference accuracy {self.reference_accuracy:.2f}%")
        return "\n".join(lines)


def measure_clue_table(
    train_loader: DataLoader,
    sampler: SamplerConfig,
    reference: ReferenceConfig,
    device: str | torch.device,
    seed: int,
) -> ClueTable:
    """Bits and clue-only Bayes accuracy per type on the training labels."""
    labels = torch.cat([batch_labels for _, batch_labels in train_loader])
    prior = label_prior(labels)
    model, confusion = clean_reference(train_loader, reference, device, seed)
    accuracy = reference_accuracy(model, train_loader, Degradation(), device, seed)
    rows = tuple(kind_information(kind, prior, sampler, confusion.float()) for kind in KINDS)
    uniform = tuple(kind_information(kind, uniform_prior(), sampler, confusion.float()) for kind in KINDS)
    return ClueTable(prior, confusion, accuracy, rows, uniform)
