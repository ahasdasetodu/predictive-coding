import unittest

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from pc_core.balance import BalanceConfig
from pc_core.graph import GaussianEnergy
from pc_core.models import mlp_graph
from pc_core.regimes import (
    BalancedRegime,
    EvaluationResult,
    ForwardInitRegime,
    PCConfig,
    PCRegime,
    TrainResult,
)
from pc_core.trainer import Trainer

SMALL_CONFIG = PCConfig(infer_steps=4, eval_steps=4, infer_lr=0.5, eval_lr=0.5, weight_lr=0.1)


class RecordingRegime:
    """Record trainer delegation while returning deterministic results."""

    def __init__(self) -> None:
        self.model = nn.Linear(2, 2)
        self.started_with: int | None = None
        self.train_calls = 0
        self.evaluate_calls = 0
        self.epoch_ends = 0

    def start_training(self, epochs: int) -> None:
        """Record the requested run length."""
        self.started_with = epochs

    def train_batch(self, batch: tuple[Tensor, Tensor]) -> TrainResult:
        """Record one delegated training batch."""
        self.train_calls += 1
        return TrainResult(loss=1.0)

    def evaluate_batch(self, batch: tuple[Tensor, Tensor]) -> EvaluationResult:
        """Record one delegated evaluation batch."""
        self.evaluate_calls += 1
        return EvaluationResult(correct=1, total=batch[1].size(0))

    def end_epoch(self) -> None:
        """Record one completed epoch."""
        self.epoch_ends += 1


class TrainerTests(unittest.TestCase):
    """Verify generic orchestration independently of predictive coding."""

    def test_trainer_delegates_complete_lifecycle(self) -> None:
        """Delegate run, batch, evaluation, and epoch lifecycle operations."""
        dataset = TensorDataset(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=2)
        regime = RecordingRegime()
        trainer = Trainer(regime, loader, loader, report_every=10)

        trainer.train(epochs=2)

        self.assertEqual(regime.started_with, 2)
        self.assertEqual(regime.train_calls, 2)
        self.assertEqual(regime.evaluate_calls, 4)
        self.assertEqual(regime.epoch_ends, 2)


class GraphTests(unittest.TestCase):
    """Verify the graph engine independently of any regime."""

    def test_propagate_keeps_clamped_nodes(self) -> None:
        """Fill free nodes from init edges without overwriting clamped ones."""
        graph = mlp_graph(hidden_dims=[3], input_dim=4, output_dim=2)
        clamped = {"image": torch.randn(5, 4), "label": torch.ones(5, 2)}

        states = graph.propagate(clamped)

        self.assertIs(states["image"], clamped["image"])
        self.assertIs(states["label"], clamped["label"])
        self.assertEqual(states["h1"].shape, (5, 3))

    def test_leapfrog_fills_every_free_node(self) -> None:
        """Reach every node from the image alone, for odd and even depths."""
        for hidden_dims in ([3], [5, 3], [7, 5, 3]):
            graph = mlp_graph(hidden_dims=hidden_dims, input_dim=4, output_dim=2, init="leapfrog")

            states = graph.propagate({"image": torch.randn(5, 4)})

            self.assertEqual(set(states), set(graph.nodes))
            for name, width in graph.nodes.items():
                self.assertEqual(states[name].shape, (5, width))


class GaussianEnergyTests(unittest.TestCase):
    """Verify that a learned variance tracks the residual it is fitted to."""

    def test_learned_variance_reaches_mean_squared_residual(self) -> None:
        """Descending on the energy drives the variance to the maximum-likelihood value."""
        torch.manual_seed(7)
        target = torch.randn(16, 5)
        prediction = target + 0.7 * torch.randn(16, 5)
        expected = (prediction - target).pow(2).mean().item()
        loss = GaussianEnergy()
        optimizer = torch.optim.Adam(loss.parameters(), lr=0.05)

        for _ in range(2000):
            optimizer.zero_grad()
            loss(prediction, target).backward()
            optimizer.step()

        self.assertAlmostEqual(loss.log_var.exp().item(), expected, places=3)

    def test_variance_starts_at_one_and_respects_its_floor(self) -> None:
        """Begin equivalent to squared error, and stay above the floor under pressure."""
        loss = GaussianEnergy(min_log_var=-2.0)
        self.assertAlmostEqual(loss.log_var.item(), 0.0, places=5)

        optimizer = torch.optim.Adam(loss.parameters(), lr=0.1)
        exact = torch.zeros(8, 3)
        for _ in range(500):
            optimizer.zero_grad()
            loss(exact, exact).backward()
            optimizer.step()

        self.assertGreaterEqual(loss.log_var.item(), -2.0)

    def test_learned_variances_are_registered_parameters(self) -> None:
        """Expose one log-variance per energy edge to the weight optimiser."""
        graph = mlp_graph(hidden_dims=[3], input_dim=4, output_dim=2, learn_precision=True)

        names = set(graph.log_variances())

        self.assertEqual(names, {edge.name for edge in graph.energy_edges})
        registered = {id(parameter) for parameter in graph.parameters()}
        for edge in graph.energy_edges:
            self.assertIn(id(edge.loss.raw_log_var), registered)


class EdgeBalancerTests(unittest.TestCase):
    """Verify that calibration drives every edge towards equal say at the nodes it touches."""

    def test_balancing_reaches_equal_say_from_either_side(self) -> None:
        """Converge to balanced gradient norms whether the image edge starts loud or quiet."""
        for start in (8.0, 0.02):
            torch.manual_seed(7)
            graph = mlp_graph(hidden_dims=[8], input_dim=6, output_dim=2, image_weight=start)
            regime = BalancedRegime(
                graph, SMALL_CONFIG, balance=BalanceConfig(interval=1, step=0.3)
            )
            images = torch.randn(16, 6)
            labels = torch.randint(0, 2, (16,))
            regime.start_training(epochs=1)
            for _ in range(80):
                regime.train_batch((images, labels))

            self.assertLess(regime.extra_metrics()["imbalance"], 1.1)
            self.assertAlmostEqual(max(edge.weight for edge in graph.energy_edges), 1.0, places=6)


class PCRegimeTests(unittest.TestCase):
    """Verify settling, weight updates, and evaluation of the base regime."""

    def _regime(self) -> PCRegime:
        """Construct a small deterministic regime for fast tests."""
        torch.manual_seed(7)
        return PCRegime(mlp_graph(hidden_dims=[3], input_dim=4, output_dim=2), SMALL_CONFIG)

    def test_settling_does_not_increase_energy(self) -> None:
        """Reduce the energy by optimizing the free hidden state."""
        regime = self._regime()
        labels = torch.tensor([0, 1, 0, 1])
        clamped = {"image": torch.randn(4, 4), "label": F.one_hot(labels, 2).float()}
        states = regime._free_states(clamped)
        before = regime.model.energy(states).item()

        regime.settle(states, ["h1"], steps=10, lr=0.5)

        after = regime.model.energy(states).item()
        self.assertLessEqual(after, before)

    def test_train_batch_updates_model_weights(self) -> None:
        """Change at least one model parameter during a training batch."""
        regime = self._regime()
        batch = (torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))
        before = [parameter.detach().clone() for parameter in regime.model.parameters()]
        regime.start_training(epochs=1)

        result = regime.train_batch(batch)

        changed = any(
            not torch.equal(old, new)
            for old, new in zip(before, regime.model.parameters())
        )
        self.assertTrue(changed)
        self.assertGreaterEqual(result.loss, 0.0)

    def test_evaluate_batch_returns_valid_counts(self) -> None:
        """Return bounded classification counts for an evaluation batch."""
        regime = self._regime()
        batch = (torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))

        result = regime.evaluate_batch(batch)

        self.assertEqual(result.total, 4)
        self.assertGreaterEqual(result.correct, 0)
        self.assertLessEqual(result.correct, result.total)


class ForwardInitRegimeTests(unittest.TestCase):
    """Verify that the init edges learn to start states near where settling ends."""

    def test_forward_init_starts_at_lower_energy_than_zeros(self) -> None:
        """After training, propagated states have lower energy than zero states."""
        torch.manual_seed(7)
        graph = mlp_graph(hidden_dims=[8], input_dim=6, output_dim=2)
        regime = ForwardInitRegime(graph, SMALL_CONFIG)
        prototypes = torch.randn(2, 6)
        labels = torch.randint(0, 2, (32,))
        images = prototypes[labels] + 0.1 * torch.randn(32, 6)
        regime.start_training(epochs=1)
        for _ in range(100):
            regime.train_batch((images, labels))

        clamped = {"image": images, "label": F.one_hot(labels, 2).float()}
        with torch.no_grad():
            forward_energy = graph.energy(regime.init_states(clamped)).item()
            zero_energy = graph.energy(PCRegime.init_states(regime, clamped)).item()

        self.assertLess(forward_energy, zero_energy)


if __name__ == "__main__":
    unittest.main()
