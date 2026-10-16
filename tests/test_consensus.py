"""Tests for the three-tower consensus graph and informed-energy rule."""

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from pc_core.balance import BalanceConfig
from pc_core.byproduct import ByproductRegime, ConditionWeights
from pc_core.messages import ClueSampler, Degradation, SamplerConfig, uniform_confusion
from pc_core.models import (
    CONSENSUS_ENERGY_EDGES,
    CONSENSUS_PROPOSAL_EDGES,
    CONSENSUS_READOUT_EDGES,
    consensus_graph,
    informed_energy_edges,
)
from pc_core.regimes import PCConfig

SMALL_CONFIG = PCConfig(infer_steps=2, eval_steps=2, infer_lr=0.1, eval_lr=0.1, weight_lr=0.01)


def _small_consensus() -> ByproductRegime:
    """A tiny consensus regime that takes real clue codes and ten labels."""
    torch.manual_seed(0)
    graph = consensus_graph(z_dim=4, h_img=6, h_msg=5, h_lab=5, input_dim=16, output_dim=10)
    sampler = ClueSampler(SamplerConfig(), uniform_confusion(), seed=0)
    return ByproductRegime(graph, SMALL_CONFIG, sampler, Degradation(), ConditionWeights(), BalanceConfig(interval=1), seed=0)


def _batch(size: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Random images with every digit present."""
    torch.manual_seed(1)
    return torch.randn(size, 16), torch.arange(size) % 10


class ConsensusGraphTests(unittest.TestCase):
    """The builder wires bidirectional towers into a shared state with shared init modules."""

    def test_nodes_edges_and_activations(self) -> None:
        """Widths, energy names, identity on latents, Tanh on hidden layers, shared modules."""
        graph = consensus_graph()
        self.assertEqual(
            graph.nodes,
            {
                "image": 784,
                "h_img": 256,
                "z_img": 128,
                "msg": 110,
                "h_msg": 128,
                "z_msg": 128,
                "label": 10,
                "h_lab": 128,
                "z_lab": 128,
                "s": 128,
            },
        )
        self.assertEqual(tuple(edge.name for edge in graph.energy_edges), CONSENSUS_ENERGY_EDGES)
        self.assertEqual(tuple(edge.name for edge in graph.proposal_edges), CONSENSUS_PROPOSAL_EDGES)
        self.assertEqual(tuple(edge.name for edge in graph.readout_init_edges()), CONSENSUS_READOUT_EDGES)
        self.assertEqual(graph.learned_init_edges(), [])
        energy = {edge.name: edge for edge in graph.energy_edges}
        for edge in graph.proposal_edges + graph.readout_edges:
            self.assertIs(edge.module, energy[edge.name].module)
        hidden = {"h_img", "h_msg", "h_lab"}
        identity = {"image", "msg", "label", "z_img", "z_msg", "z_lab", "s"}
        for edge in graph.energy_edges:
            expected = nn.Tanh if edge.dst in hidden else nn.Identity
            self.assertIsInstance(edge.module[-1], expected, edge.name)
            self.assertIn(edge.dst, hidden | identity)

    def test_informed_energy_drops_unobserved_votes(self) -> None:
        """Image-only omits the clue vote and clue bottom-up; both-present keeps every edge."""
        graph = consensus_graph(z_dim=3, h_img=4, h_msg=4, h_lab=4, input_dim=5, output_dim=2)
        image_only = {edge.name for edge in informed_energy_edges(graph, {"image"})}
        self.assertNotIn("z_msg->s", image_only)
        self.assertNotIn("msg->h_msg", image_only)
        self.assertNotIn("h_msg->z_msg", image_only)
        self.assertNotIn("z_lab->s", image_only)
        self.assertNotIn("label->h_lab", image_only)
        self.assertIn("z_img->s", image_only)
        self.assertIn("image->h_img", image_only)
        self.assertIn("s->z_msg", image_only)
        self.assertIn("h_msg->msg", image_only)
        self.assertIn("h_lab->label", image_only)
        both = informed_energy_edges(graph, {"image", "msg", "label"})
        self.assertEqual([edge.name for edge in both], list(CONSENSUS_ENERGY_EDGES))

    def test_image_only_proposes_s_from_the_image_vote(self) -> None:
        """Image fills the image tower and s; msg and label are read off s."""
        graph = consensus_graph(z_dim=3, h_img=4, h_msg=4, h_lab=4, input_dim=5, output_dim=2)
        image = torch.randn(7, 5)
        states = graph.propagate({"image": image})
        self.assertEqual(set(states), set(graph.nodes))
        proposal = {edge.name: edge for edge in graph.proposal_edges}
        readout = {edge.name: edge for edge in graph.readout_edges}
        with torch.no_grad():
            self.assertTrue(torch.allclose(states["h_img"], proposal["image->h_img"].predict(states)))
            self.assertTrue(torch.allclose(states["z_img"], proposal["h_img->z_img"].predict(states)))
            self.assertTrue(torch.allclose(states["s"], proposal["z_img->s"].predict(states)))
            self.assertTrue(torch.allclose(states["z_msg"], readout["s->z_msg"].predict(states)))
            self.assertTrue(torch.allclose(states["msg"], readout["h_msg->msg"].predict(states)))
            self.assertTrue(torch.allclose(states["label"], readout["h_lab->label"].predict(states)))

    def test_both_modalities_average_their_votes(self) -> None:
        """With image and msg clamped, s is the mean of the two proposal votes."""
        graph = consensus_graph(z_dim=3, h_img=4, h_msg=4, h_lab=4, msg_dim=6, input_dim=5, output_dim=2)
        clamped = {"image": torch.randn(7, 5), "msg": torch.randn(7, 6)}
        states = graph.propagate(clamped)
        proposal = {edge.name: edge for edge in graph.proposal_edges}
        with torch.no_grad():
            expected = 0.5 * (proposal["z_img->s"].predict(states) + proposal["z_msg->s"].predict(states))
        self.assertTrue(torch.allclose(states["s"], expected, atol=1e-6))


class ConsensusRegimeTests(unittest.TestCase):
    """The byproduct regime drops uninformed votes and still classifies from the image."""

    def test_each_condition_trains(self) -> None:
        """Every condition updates weights and records its code."""
        expected = {"image": 0.0, "msg": 1.0, "both": 2.0}
        for condition, code in expected.items():
            weights = ConditionWeights(
                image=float(condition == "image"),
                msg=float(condition == "msg"),
                both=float(condition == "both"),
            )
            torch.manual_seed(0)
            graph = consensus_graph(z_dim=4, h_img=6, h_msg=5, h_lab=5, input_dim=16, output_dim=10)
            sampler = ClueSampler(SamplerConfig(), uniform_confusion(), seed=0)
            regime = ByproductRegime(
                graph, SMALL_CONFIG, sampler, Degradation(), weights, BalanceConfig(interval=1), seed=0
            )
            regime.start_training(epochs=1)
            before = [parameter.detach().clone() for parameter in regime.model.parameters()]
            result = regime.train_batch(_batch())
            self.assertEqual(regime.condition, condition)
            self.assertEqual(result.metrics["condition"], code)
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, regime.model.parameters())))
            self.assertIn("gap[s]", result.metrics)
            if condition == "msg":
                self.assertEqual(regime.proposing, [])
                self.assertEqual(result.metrics["extra_loss"], 0.0)

    def test_image_batch_does_not_train_the_clue_encoder(self) -> None:
        """On an image-only batch the msg bottom-up energy modules get no gradient."""
        weights = ConditionWeights(1, 0, 0)
        torch.manual_seed(0)
        graph = consensus_graph(z_dim=4, h_img=6, h_msg=5, h_lab=5, input_dim=16, output_dim=10)
        sampler = ClueSampler(SamplerConfig(), uniform_confusion(), seed=0)
        regime = ByproductRegime(graph, SMALL_CONFIG, sampler, Degradation(), weights, None, seed=0)
        msg_up = [edge for edge in regime.model.energy_edges if edge.name in {"msg->h_msg", "h_msg->z_msg", "z_msg->s"}]
        before = [parameter.detach().clone() for edge in msg_up for parameter in edge.module.parameters()]
        regime.start_training(epochs=1)
        regime.train_batch(_batch())
        after = [parameter for edge in msg_up for parameter in edge.module.parameters()]
        for old, new in zip(before, after):
            self.assertTrue(torch.equal(old, new))

    def test_infer_image_only_returns_msg_and_label(self) -> None:
        """Evaluation from the image fills the clue and the label."""
        regime = _small_consensus()
        images, _ = _batch()
        states = regime.infer(regime.clamp(images, None, None, "image"))
        self.assertIn("msg", states)
        self.assertIn("label", states)
        self.assertEqual(states["msg"].shape, (20, 110))
        self.assertEqual(states["label"].shape, (20, 10))

    def test_both_present_logs_vote_ratios(self) -> None:
        """Ratios at s compare the image vote to the clue and label votes."""
        weights = ConditionWeights(0, 0, 1)
        torch.manual_seed(0)
        graph = consensus_graph(z_dim=4, h_img=6, h_msg=5, h_lab=5, input_dim=16, output_dim=10)
        sampler = ClueSampler(SamplerConfig(), uniform_confusion(), seed=0)
        regime = ByproductRegime(
            graph, SMALL_CONFIG, sampler, Degradation(), weights, BalanceConfig(interval=1), seed=0
        )
        regime.start_training(epochs=1)
        result = regime.train_batch(_batch())
        self.assertIn("ratio_z[image/msg]", result.metrics)
        self.assertIn("ratio_z[image/label]", result.metrics)
        self.assertGreater(result.metrics["ratio_z[image/msg]"], 0.0)


class ConsensusOwnInitTests(unittest.TestCase):
    """Own-module proposals keep energy local and train the label by cross-entropy."""

    def test_learned_proposals_do_not_share_energy_modules(self) -> None:
        """Proposal names match the bottom-up hops and each hop owns its Linear."""
        graph = consensus_graph(z_dim=3, h_img=4, h_msg=4, h_lab=4, input_dim=5, output_dim=2, share_init=False)
        learned = graph.learned_init_edges()
        self.assertEqual(tuple(edge.name for edge in learned), CONSENSUS_PROPOSAL_EDGES)
        self.assertEqual(tuple(edge.name for edge in graph.readout_init_edges()), CONSENSUS_READOUT_EDGES)
        energy = {edge.name: edge for edge in graph.energy_edges}
        for edge in learned:
            self.assertIsNot(edge.module, energy[edge.name].module)
        for edge in graph.readout_edges:
            self.assertIs(edge.module, energy[edge.name].module)

    def test_image_only_proposes_s_from_the_learned_vote(self) -> None:
        """Image fills the image tower and s through own-module proposals; label is read off s."""
        graph = consensus_graph(z_dim=3, h_img=4, h_msg=4, h_lab=4, input_dim=5, output_dim=2, share_init=False)
        image = torch.randn(7, 5)
        states = graph.propagate({"image": image})
        self.assertEqual(set(states), set(graph.nodes))
        proposal = {edge.name: edge for edge in graph.proposal_edges}
        readout = {edge.name: edge for edge in graph.readout_edges}
        with torch.no_grad():
            self.assertTrue(torch.allclose(states["h_img"], proposal["image->h_img"].predict(states)))
            self.assertTrue(torch.allclose(states["z_img"], proposal["h_img->z_img"].predict(states)))
            self.assertTrue(torch.allclose(states["s"], proposal["z_img->s"].predict(states)))
            self.assertTrue(torch.allclose(states["label"], readout["h_lab->label"].predict(states)))

    def test_cross_entropy_trains_init_encoder_not_energy_encoder(self) -> None:
        """Image-only CE updates the learned image hop and the label readout, not energy image->h_img."""
        torch.manual_seed(0)
        graph = consensus_graph(z_dim=4, h_img=6, h_msg=5, h_lab=5, input_dim=16, output_dim=10, share_init=False)
        sampler = ClueSampler(SamplerConfig(), uniform_confusion(), seed=0)
        regime = ByproductRegime(graph, SMALL_CONFIG, sampler, Degradation(), ConditionWeights(), None, seed=0)
        regime.condition = "image"
        images, labels = _batch()
        graph.zero_grad()
        regime.extra_loss({"image": images, "label": F.one_hot(labels, 10).float()}).backward()
        init_img = next(edge for edge in graph.learned_init_edges() if edge.name == "image->h_img")
        energy_img = next(edge for edge in graph.energy_edges if edge.name == "image->h_img")
        label_out = next(edge for edge in graph.energy_edges if edge.name == "h_lab->label")
        self.assertGreater(float(init_img.module[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNone(energy_img.module[0].weight.grad)
        self.assertGreater(float(label_out.module[0].weight.grad.abs().sum()), 0.0)

    def test_message_batch_copies_the_learned_clue_proposals(self) -> None:
        """A message-only batch trains the own-module clue hops against the settled states."""
        torch.manual_seed(0)
        graph = consensus_graph(z_dim=4, h_img=6, h_msg=5, h_lab=5, input_dim=16, output_dim=10, share_init=False)
        sampler = ClueSampler(SamplerConfig(), uniform_confusion(), seed=0)
        regime = ByproductRegime(
            graph, SMALL_CONFIG, sampler, Degradation(), ConditionWeights(0, 1, 0), None, seed=0
        )
        regime.start_training(epochs=1)
        result = regime.train_batch(_batch())
        proposing = {edge.name for edge in regime.proposing}
        self.assertEqual(
            proposing,
            {"msg->h_msg", "h_msg->z_msg", "z_msg->s", "label->h_lab", "h_lab->z_lab", "z_lab->s"},
        )
        self.assertGreater(result.metrics["extra_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
