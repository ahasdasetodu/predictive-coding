"""A small feedforward classifier used as a calibrated reference, not as the model under study.

It has the shape of the hub's image pathway (Linear, Tanh, Linear) and is trained with
cross-entropy, so its softmax is a proper posterior estimate. It supplies the confusion
matrix that defines `maybe k`, the posterior behind the Bayes ceiling, and the scorer for
imagined images.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pc_core.messages import Degradation


@dataclass(frozen=True)
class ReferenceConfig:
    """Width and training budget of the reference classifier."""

    hidden: int = 256
    epochs: int = 3
    lr: float = 1e-3


class ReferenceClassifier(nn.Module):
    """Linear, Tanh, Linear over flat images; softmax gives the class posterior."""

    def __init__(self, hidden: int = 256, input_dim: int = 28 * 28, classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, classes))

    def forward(self, images: Tensor) -> Tensor:
        """Class logits."""
        return self.net(images)

    @torch.no_grad()
    def posterior(self, images: Tensor) -> Tensor:
        """Softmax class probabilities."""
        return F.softmax(self.net(images), dim=1)


def train_reference(
    loader: DataLoader,
    degradation: Degradation,
    config: ReferenceConfig,
    device: str | torch.device,
    seed: int,
) -> ReferenceClassifier:
    """Fit the reference classifier on images degraded the way the experiment degrades them."""
    torch.manual_seed(seed)
    model = ReferenceClassifier(config.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(config.epochs):
        for images, labels in loader:
            images = degradation.apply(images.to(device), generator)
            optimizer.zero_grad()
            F.cross_entropy(model(images), labels.to(device)).backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def reference_accuracy(
    model: ReferenceClassifier,
    loader: DataLoader,
    degradation: Degradation,
    device: str | torch.device,
    seed: int,
) -> float:
    """Percent correct on a loader, degraded with a fixed seed."""
    generator = torch.Generator(device=device).manual_seed(seed)
    correct = 0
    total = 0
    for images, labels in loader:
        images = degradation.apply(images.to(device), generator)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == labels.to(device)).sum().item())
        total += labels.shape[0]
    return 100.0 * correct / total


def clean_reference(
    loader: DataLoader,
    config: ReferenceConfig,
    device: str | torch.device,
    seed: int,
) -> tuple[ReferenceClassifier, Tensor]:
    """A reference trained on clean images, with its confusion counts on the same loader."""
    clean = Degradation()
    model = train_reference(loader, clean, config, device, seed)
    return model, confusion_matrix(model, loader, clean, device, seed)


@torch.no_grad()
def confusion_matrix(
    model: ReferenceClassifier,
    loader: DataLoader,
    degradation: Degradation,
    device: str | torch.device,
    seed: int,
) -> Tensor:
    """Counts of (true digit, predicted digit) over a loader."""
    generator = torch.Generator(device=device).manual_seed(seed)
    counts = torch.zeros(10, 10, dtype=torch.long)
    for images, labels in loader:
        images = degradation.apply(images.to(device), generator)
        predictions = model(images).argmax(dim=1).cpu()
        for truth, guess in zip(labels.tolist(), predictions.tolist()):
            counts[truth, guess] += 1
    return counts
