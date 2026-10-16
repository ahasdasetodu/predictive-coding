import unittest
from pathlib import Path

import torch
from torch import Tensor
from torchvision import datasets

from pc_core.dataset import _MEAN, _STD

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


class MnistMeasurementTests(unittest.TestCase):
    """Lock the training-set measurements that do not require fitting a model."""

    images: Tensor
    labels: Tensor

    @classmethod
    def setUpClass(cls) -> None:
        """Load normalized MNIST training images from the local data directory."""
        if not (DATA_ROOT / "MNIST").is_dir():
            raise unittest.SkipTest("MNIST is not in ./data")
        raw = datasets.MNIST(root=str(DATA_ROOT), train=True, download=False)
        images = raw.data.float().div(255.0).sub(_MEAN).div(_STD).flatten(1)
        cls.images = images
        cls.labels = raw.targets.long()

    def test_zero_prediction_energy(self) -> None:
        """Predicting zeros scores 392.02 per training image."""
        energy = 0.5 * self.images.pow(2).sum(dim=1).mean().item()
        self.assertAlmostEqual(energy, 392.02, places=2)

    def test_pixel_split(self) -> None:
        """107 pixels are near-constant and 677 vary."""
        near_constant, varying = _pixel_split(self.images, self.labels)
        self.assertEqual(near_constant, 107)
        self.assertEqual(varying, 677)

    def test_fisher_correlation(self) -> None:
        """Log within-class variance and log Fisher ratio correlate 0.9613 on the varying pixels."""
        correlation = _fisher_correlation(self.images, self.labels)
        self.assertAlmostEqual(correlation, 0.9613, places=4)


def _within_and_fisher(images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Pooled within-class variance and the Fisher ratio of each pixel."""
    counts = torch.stack([(labels == digit).sum() for digit in range(10)]).float()
    means = torch.stack([images[labels == digit].mean(dim=0) for digit in range(10)])
    within_sum = torch.zeros(images.shape[1])
    for digit in range(10):
        within_sum += (images[labels == digit] - means[digit]).pow(2).sum(dim=0)
    within_var = within_sum / images.shape[0]
    global_mean = images.mean(dim=0)
    between_var = ((means - global_mean).pow(2) * counts[:, None]).sum(dim=0) / images.shape[0]
    return within_var, between_var / within_var.clamp_min(1e-12)


def _pixel_split(images: Tensor, labels: Tensor) -> tuple[int, int]:
    """Count near-constant and varying pixels on the training set."""
    within_var, _fisher = _within_and_fisher(images, labels)
    near_constant = int((within_var.sqrt() <= 0.01).sum())
    return near_constant, int(images.shape[1] - near_constant)


def _fisher_correlation(images: Tensor, labels: Tensor) -> float:
    """Correlation of log within-class variance with log Fisher ratio on varying pixels."""
    within_var, fisher = _within_and_fisher(images, labels)
    varying = within_var.sqrt() > 0.01
    return _correlation(within_var[varying].log(), fisher[varying].log())


def _correlation(left: Tensor, right: Tensor) -> float:
    """Pearson correlation of two equal-length vectors."""
    left = left - left.mean()
    right = right - right.mean()
    return float((left * right).sum() / left.norm() / right.norm())


if __name__ == "__main__":
    unittest.main()
