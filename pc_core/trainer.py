from dataclasses import dataclass

from torch.utils.data import DataLoader

from pc_core.regimes import TrainResult, TrainingRegime


@dataclass(frozen=True)
class BatchReport:
    """One batch the trainer logged: its index, loss, and regime metrics."""

    batch_idx: int
    loss: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class EpochReport:
    """Train and validation accuracy, plus the batches logged for one epoch."""

    epoch: int
    train_accuracy: float
    validation_accuracy: float
    batches: tuple[BatchReport, ...]


class Trainer:
    """Run reusable training and evaluation loops for a pluggable regime."""

    def __init__(
        self,
        regime: TrainingRegime,
        dataloader: DataLoader,
        test_dataloader: DataLoader,
        report_every: int = 100,
    ) -> None:
        self.regime = regime
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.report_every = report_every

    def train(self, epochs: int = 1) -> list[EpochReport]:
        """Train for complete epochs and return the accuracies and batches that were logged."""
        print("Starting training")
        self.regime.start_training(epochs)
        reports: list[EpochReport] = []
        for epoch in range(epochs):
            self.regime.model.train()
            batches: list[BatchReport] = []
            for batch_idx, batch in enumerate(self.dataloader):
                result = self.regime.train_batch(batch)
                if batch_idx % self.report_every == 0:
                    self._report_batch(epoch, batch_idx, result)
                    batches.append(BatchReport(batch_idx, result.loss, dict(result.metrics)))
            train_accuracy = self.evaluate(self.dataloader, "Train")
            validation_accuracy = self.evaluate(self.test_dataloader, "Validation")
            self.regime.end_epoch()
            reports.append(EpochReport(epoch, train_accuracy, validation_accuracy, tuple(batches)))
        return reports

    def evaluate(self, dataloader: DataLoader | None = None, split: str = "Validation") -> float:
        """Aggregate and report classification accuracy for one data loader."""
        if dataloader is None:
            dataloader = self.test_dataloader
        was_training = self.regime.model.training
        self.regime.model.eval()
        correct = 0
        total = 0
        for batch in dataloader:
            result = self.regime.evaluate_batch(batch)
            correct += result.correct
            total += result.total
        if total == 0:
            raise ValueError("cannot evaluate an empty data loader")
        accuracy = 100 * correct / total
        print(f"--> Epoch {split} Accuracy: {accuracy:.2f}%")
        self.regime.model.train(was_training)
        return accuracy

    def _report_batch(self, epoch: int, batch_idx: int, result: TrainResult) -> None:
        """Print loss and measurements reported by the active regime."""
        measurements = " | ".join(
            f"{name}: {value:.6f}" for name, value in result.metrics.items()
        )
        suffix = f" | {measurements}" if measurements else ""
        print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {result.loss:.4f}{suffix}")
