import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pc_core.balance import BalanceConfig
from pc_core.byproduct import ByproductRegime, ConditionWeights
from pc_core.byproduct_eval import evaluate_hub
from pc_core.graph import Edge, PCGraph
from pc_core.messages import KINDS, ClueSampler, Degradation, SamplerConfig, uniform_confusion, uniform_prior
from pc_core.models import HUB_ENERGY_EDGES, HUB_INIT_EDGES, graph_from_edges, hub_graph, mlp_graph
from pc_core.reference import ReferenceClassifier
from pc_core.regimes import PCConfig

SMALL_CONFIG = PCConfig(infer_steps=2, eval_steps=2, infer_lr=0.1, eval_lr=0.1, weight_lr=0.01)


def _small_hub() -> PCGraph:
    """A hub with tiny widths for fast tests."""
    torch.manual_seed(0)
    return hub_graph(z_dim=3, h_dim=4, msg_dim=6, input_dim=5, output_dim=2)


def _regime(conditions: ConditionWeights, input_dim: int = 16, balance: BalanceConfig | None = BalanceConfig(interval=1)) -> ByproductRegime:
    """A small regime on a hub that takes real clue codes and ten labels."""
    torch.manual_seed(0)
    graph = hub_graph(z_dim=4, h_dim=6, input_dim=input_dim, output_dim=10)
    sampler = ClueSampler(SamplerConfig(), uniform_confusion(), seed=0)
    return ByproductRegime(graph, SMALL_CONFIG, sampler, Degradation(), conditions, balance, seed=0)


def _batch(input_dim: int = 16, size: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Random images with every digit present."""
    torch.manual_seed(1)
    return torch.randn(size, input_dim), torch.arange(size) % 10


class ReachablePropagationTests(unittest.TestCase):
    """Reachable init order fills what the clamped nodes can reach and averages competing proposals."""

    def test_image_only_reaches_every_node(self) -> None:
        """Image fills h_img and z through learned edges; msg and label are read off z."""
        graph = _small_hub()
        image = torch.randn(7, 5)
        states = graph.propagate({"image": image})
        self.assertIs(states["image"], image)
        self.assertEqual(set(states), set(graph.nodes))
        readout = {edge.name: edge for edge in graph.readout_init_edges()}
        with torch.no_grad():
            self.assertTrue(torch.allclose(states["label"], readout["z->label"].predict(states)))
            self.assertTrue(torch.allclose(states["msg"], readout["z->msg"].predict(states)))
            learned = {edge.name: edge for edge in graph.learned_init_edges()}
            self.assertTrue(torch.allclose(states["h_img"], learned["image->h_img"].predict(states)))

    def test_message_only_reaches_every_node(self) -> None:
        """Msg proposes z alone; h_img and the image are read off z."""
        graph = _small_hub()
        msg = torch.randn(7, 6)
        states = graph.propagate({"msg": msg})
        self.assertEqual(set(states), set(graph.nodes))
        learned = {edge.name: edge for edge in graph.learned_init_edges()}
        readout = {edge.name: edge for edge in graph.readout_init_edges()}
        with torch.no_grad():
            self.assertTrue(torch.allclose(states["z"], learned["msg->z"].predict(states)))
            self.assertTrue(torch.allclose(states["h_img"], readout["z->h_img"].predict(states)))
            self.assertTrue(torch.allclose(states["image"], readout["h_img->image"].predict(states)))

    def test_both_modalities_average_their_proposals(self) -> None:
        """With image and msg clamped, z is the mean of the two learned proposals."""
        graph = _small_hub()
        clamped = {"image": torch.randn(7, 5), "msg": torch.randn(7, 6)}
        states = graph.propagate(clamped)
        learned = {edge.name: edge for edge in graph.learned_init_edges()}
        with torch.no_grad():
            expected = 0.5 * (learned["h_img->z"].predict(states) + learned["msg->z"].predict(states))
        self.assertTrue(torch.allclose(states["z"], expected, atol=1e-6))
        self.assertEqual(
            [edge.name for edge in graph.proposing_edges(clamped)],
            ["image->h_img", "h_img->z", "msg->z"],
        )

    def test_unreached_nodes_stay_absent(self) -> None:
        """A graph with no path from the clamped node leaves the rest for the regime to zero."""
        nodes = {"a": 2, "b": 3, "c": 4}
        graph = graph_from_edges(nodes, ["b->c"], ["c->b"])
        states = graph.propagate({"a": torch.randn(1, 2)})
        self.assertEqual(set(states), {"a"})

    def test_cycle_among_init_edges_is_rejected(self) -> None:
        """Two learned init edges that depend on each other cannot be ordered."""
        nodes = {"a": 2, "b": 3, "c": 4}
        graph = graph_from_edges(nodes, ["a->c"], ["a->b", "a->c", "b->c", "c->b"])
        with self.assertRaises(ValueError):
            graph.propagate({"a": torch.randn(1, 2)})

    def test_fixed_order_is_unchanged(self) -> None:
        """mlp_graph still runs its init edges in list order and fills every node."""
        graph = mlp_graph(hidden_dims=[3], input_dim=4, output_dim=2)
        self.assertEqual(graph.init_order, "fixed")
        states = graph.propagate({"image": torch.randn(5, 4)})
        self.assertEqual(set(states), set(graph.nodes))


class HubGraphTests(unittest.TestCase):
    """The hub builder produces the documented edges and shares read-off modules."""

    def test_hub_edges_and_widths(self) -> None:
        """Energy edges, learned init edges, and read-off edges match the design."""
        graph = hub_graph()
        self.assertEqual(graph.nodes, {"image": 784, "msg": 110, "z": 128, "h_img": 256, "label": 10})
        self.assertEqual(tuple(edge.name for edge in graph.energy_edges), HUB_ENERGY_EDGES)
        self.assertEqual(tuple(edge.name for edge in graph.learned_init_edges()), HUB_INIT_EDGES)
        self.assertEqual(tuple(edge.name for edge in graph.readout_init_edges()), HUB_ENERGY_EDGES)
        energy = {edge.name: edge for edge in graph.energy_edges}
        for edge in graph.readout_init_edges():
            self.assertIs(edge.module, energy[edge.name].module)
        parameter_ids = [id(parameter) for parameter in graph.parameters()]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))

    def test_data_nodes_get_identity_and_hidden_nodes_tanh(self) -> None:
        """Predictions into image, msg, and label are not squashed; into z and h_img they are."""
        graph = hub_graph(z_dim=3, h_dim=4, msg_dim=6, input_dim=5, output_dim=2)
        for edge in graph.energy_edges + graph.learned_init_edges():
            expected = nn.Identity if edge.dst in {"image", "msg", "label"} else nn.Tanh
            self.assertIsInstance(edge.module[-1], expected, edge.name)

    def test_builder_rejects_bad_edges(self) -> None:
        """Unknown nodes and read-offs that are not energy edges are errors."""
        nodes = {"a": 2, "b": 3}
        with self.assertRaises(ValueError):
            graph_from_edges(nodes, ["a->x"], [])
        with self.assertRaises(ValueError):
            graph_from_edges(nodes, ["a->b"], [], readout_edges=["b->a"])
        with self.assertRaises(ValueError):
            graph_from_edges(nodes, ["a->b"], [], proposal_edges=["b->a"])
        joined = graph_from_edges({"a": 2, "b": 3, "c": 1}, ["a+b->c"], [])
        self.assertIsInstance(joined.energy_edges[0], Edge)
        self.assertEqual(joined.energy_edges[0].sources, ("a", "b"))


class ByproductRegimeTests(unittest.TestCase):
    """The regime trains under each condition, trains only the init edges that proposed, and balances on both-present batches."""

    def test_each_condition_trains_and_logs(self) -> None:
        """Every condition updates weights, records its code, and names the edges that proposed."""
        expected = {
            "image": (ConditionWeights(1, 0, 0), ["image->h_img", "h_img->z", "label->z"], 0.0),
            "msg": (ConditionWeights(0, 1, 0), ["msg->z", "label->z"], 1.0),
            "both": (ConditionWeights(0, 0, 1), ["image->h_img", "h_img->z", "msg->z", "label->z"], 2.0),
        }
        for condition, (weights, proposing, code) in expected.items():
            regime = _regime(weights)
            regime.start_training(epochs=1)
            before = [parameter.detach().clone() for parameter in regime.model.parameters()]
            result = regime.train_batch(_batch())
            self.assertEqual(regime.condition, condition)
            self.assertEqual([edge.name for edge in regime.proposing], proposing)
            self.assertEqual(result.metrics["condition"], code)
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, regime.model.parameters())))
            self.assertIn("gap[z]", result.metrics)

    def test_image_batch_classifies_from_the_image_alone(self) -> None:
        """With an image present, the extra loss is cross-entropy of the image-only init pass."""
        regime = _regime(ConditionWeights(1, 0, 0))
        images, labels = _batch()
        regime.condition = "image"
        loss = regime.extra_loss({"image": images, "label": F.one_hot(labels, 10).float()})
        logits = regime.model.propagate({"image": images})["label"]
        expected = F.cross_entropy(logits, labels, reduction="sum")
        self.assertAlmostEqual(loss.item(), expected.item(), places=4)

    def test_message_batch_keeps_the_copy_loss(self) -> None:
        """A message-only batch still trains the proposing init edges against the settled states."""
        regime = _regime(ConditionWeights(0, 1, 0))
        regime.condition = "msg"
        regime.proposing = [edge for edge in regime.model.learned_init_edges() if edge.name == "msg->z"]
        settled = {name: torch.randn(4, width) for name, width in regime.model.nodes.items()}
        loss = regime.extra_loss(settled)
        expected = regime.model.energy(settled, regime.proposing)
        self.assertAlmostEqual(loss.item(), expected.item(), places=4)

    def test_extra_loss_uses_only_proposing_edges(self) -> None:
        """On an image-only batch the clue init edge gets no gradient."""
        regime = _regime(ConditionWeights(1, 0, 0))
        regime.start_training(epochs=1)
        msg_edge = next(edge for edge in regime.model.learned_init_edges() if edge.name == "msg->z")
        before = [parameter.detach().clone() for parameter in msg_edge.module.parameters()]
        regime.train_batch(_batch())
        for old, new in zip(before, msg_edge.module.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_balancer_and_ratios_only_on_both_present_batches(self) -> None:
        """Ratios at z and balancer observations appear on both-present batches and not otherwise."""
        regime = _regime(ConditionWeights(1, 0, 0))
        regime.start_training(epochs=1)
        result = regime.train_batch(_batch())
        self.assertEqual(regime.balancer.observations, 0)
        self.assertFalse(any(key.startswith("ratio_z") for key in result.metrics))

        regime = _regime(ConditionWeights(0, 0, 1))
        regime.start_training(epochs=1)
        result = regime.train_batch(_batch())
        self.assertEqual(regime.balancer.observations, 1)
        self.assertIn("ratio_z[image/msg]", result.metrics)
        self.assertIn("ratio_z[image/label]", result.metrics)
        self.assertIn("imbalance", result.metrics)
        self.assertGreater(result.metrics["ratio_z[image/msg]"], 0.0)

    def test_evaluate_batch_is_image_only(self) -> None:
        """Evaluation clamps the image alone and counts label hits."""
        regime = _regime(ConditionWeights())
        images, labels = _batch()
        result = regime.evaluate_batch((images, labels))
        self.assertEqual(result.total, 20)
        self.assertLessEqual(result.correct, 20)


class EvaluationTests(unittest.TestCase):
    """The read-out covers every type, writes its artifacts, and pairs each type with its own baseline."""

    def test_evaluate_hub_writes_artifacts_for_every_kind(self) -> None:
        """Synthetic data through the full read-out produces every kind, grids, thoughts, and the plot."""
        regime = _regime(ConditionWeights(), input_dim=784, balance=None)
        images, labels = _batch(input_dim=784, size=40)
        loader = DataLoader(TensorDataset(images, labels), batch_size=20)
        reference = ReferenceClassifier(hidden=4)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            evaluation = evaluate_hub(
                regime, loader, reference, reference, uniform_prior(), SamplerConfig(), uniform_confusion(),
                directory, eval_curve=(1, 2), thoughts=3, reference_clean_accuracy=10.0, reference_degraded_accuracy=10.0,
            )
            self.assertEqual([result.kind for result in evaluation.kinds], list(KINDS))
            for result in evaluation.kinds:
                self.assertGreater(result.count, 0)
                for value in (result.image_only, result.image_true, result.image_shuffled, result.clue_only, result.ceiling):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)
            even = next(result for result in evaluation.kinds if result.kind == "even")
            self.assertEqual(even.count, 20)
            self.assertEqual(set(evaluation.eval_curve), {1, 2})
            self.assertEqual(len(evaluation.imagine_labels), 10)
            self.assertTrue((directory / "gain.png").is_file())
            self.assertTrue((directory / "imagine-labels.png").is_file())
            self.assertTrue((directory / "imagine-not-between.png").is_file())
            self.assertEqual(len((directory / "thoughts.txt").read_text(encoding="utf-8").strip().splitlines()), 3)
            document = evaluation.document()
            self.assertIn("think", document)
            self.assertEqual(len(document["kinds"]), len(KINDS))


if __name__ == "__main__":
    unittest.main()
