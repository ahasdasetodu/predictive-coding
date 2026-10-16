"""Training regime for the hub graph: classify with whichever modalities a batch has.

The label is always clamped. Each batch has the image, the clue, or both. A modality
that is absent is a free leaf that starts at its own prediction from `z`, so it pulls
on nothing. No objective mentions language; the clue is only ever an input.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from pc_core.balance import BalanceConfig, EdgeBalancer
from pc_core.graph import Edge, PCGraph, States
from pc_core.messages import Clue, ClueSampler, Degradation, encode
from pc_core.models import informed_energy_edges
from pc_core.regimes import Batch, ForwardInitRegime, PCConfig, TrainResult

Condition = Literal["image", "msg", "both"]
CONDITIONS: tuple[Condition, ...] = ("image", "msg", "both")


@dataclass(frozen=True)
class ConditionWeights:
    """How often a training batch has the image only, the clue only, or both."""

    image: float = 1.0
    msg: float = 1.0
    both: float = 1.0

    def as_list(self) -> list[float]:
        """Weights in `CONDITIONS` order."""
        return [self.image, self.msg, self.both]


class ByproductRegime(ForwardInitRegime):
    """Forward init on the hub with random modality presence and the label always clamped.

    The balancer, when present, only measures on batches where every leaf is clamped:
    there every energy edge touches data and the clamped set is fixed, so its running
    norms are not mixed across conditions. Its weights apply to every batch.
    """

    msg_node = "msg"
    z_node = "z"

    def __init__(
        self,
        model: PCGraph,
        config: PCConfig,
        sampler: ClueSampler,
        degradation: Degradation,
        conditions: ConditionWeights = ConditionWeights(),
        balance: BalanceConfig | None = BalanceConfig(),
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        super().__init__(model, config, device=device)
        self.z_node = "s" if "s" in model.nodes else "z"
        self.sampler = sampler
        self.degradation = degradation
        self.conditions = conditions
        self.balancer: EdgeBalancer | None = None if balance is None else EdgeBalancer(self.model, balance)
        self.random = random.Random(seed)
        self.degrade_generator = torch.Generator(device=device).manual_seed(seed)
        self.condition: Condition = "both"
        self.proposing: list[Edge] = []
        self.ratios: dict[str, float] = {}
        self.gaps: dict[str, float] = {}
        self._init_states: States = {}
        self._clamped_names: set[str] = set()

    def train_batch(self, batch: Batch) -> TrainResult:
        """Degrade the images, draw clues and a condition, and train on the clamped leaves."""
        images, labels = batch
        images = self.degradation.apply(self._flat(images), self.degrade_generator)
        clues = self.sampler.sample(labels.tolist())
        self.condition = self.random.choices(CONDITIONS, weights=self.conditions.as_list())[0]
        clamped = self.clamp(images, labels, clues, self.condition)
        self.proposing = self.model.proposing_edges(clamped)
        return self.train_clamped(clamped)

    def train_clamped(self, clamped: States) -> TrainResult:
        """Record the clamped set so settling drops uninformed votes, then train."""
        self._clamped_names = set(clamped)
        return super().train_clamped(clamped)

    def infer(self, clamped: States, steps: int | None = None) -> States:
        """Record the clamped set so evaluation settling drops uninformed votes."""
        self._clamped_names = set(clamped)
        return super().infer(clamped, steps)

    def current_energy_edges(self) -> Sequence[Edge] | None:
        """On a consensus graph, drop votes and bottom-up edges from unclamped towers."""
        if "z_img" not in self.model.nodes:
            return None
        return informed_energy_edges(self.model, self._clamped_names)

    def clamp(self, images: Tensor, labels: Tensor | None, clues: Sequence[Clue] | None, condition: Condition) -> States:
        """The clamped leaves for one condition. `labels` and `clues` may be omitted at evaluation."""
        clamped: States = {}
        if labels is not None:
            clamped[self.label_node] = self._one_hot(labels)
        if condition in ("image", "both"):
            clamped[self.input_node] = images.to(self.device)
        if condition in ("msg", "both"):
            if clues is None:
                raise ValueError("a clue is needed when the message is clamped")
            clamped[self.msg_node] = encode(clues).to(self.device)
        return clamped

    def extra_loss(self, settled: States) -> Tensor:
        """Classify from the image when the batch has one; otherwise copy the proposing edges.

        A message-only batch has no image, so it keeps the settled-state copy loss. The
        message is not a classification input here: that would train the clue to name the digit.
        """
        if self.condition == "msg":
            if not self.proposing:
                return torch.zeros((), device=self.device)
            return self.model.energy(settled, self.proposing)
        return self.init_label_loss(settled[self.input_node], settled[self.label_node].argmax(dim=1))

    def observe_init(self, states: States) -> None:
        """On batches with every leaf clamped, measure the pulls at z and let the balancer act."""
        self._init_states = {
            name: state.detach().clone() for name, state in states.items() if state.requires_grad
        }
        if self.condition != "both":
            return
        self._measure_ratios(states)
        if self.balancer is not None:
            self.balancer.observe(states)

    def observe_settled(self, settled: States) -> None:
        """Per-node distance settling moved each free state, averaged over the batch."""
        self.gaps = {
            name: (settled[name] - start).norm(dim=1).mean().item()
            for name, start in self._init_states.items()
        }

    def extra_metrics(self) -> dict[str, float]:
        """Condition code, ratios at z, settling gaps, and the balancer's weights."""
        metrics: dict[str, float] = {"condition": float(CONDITIONS.index(self.condition))}
        metrics.update({f"ratio_z[{name}]": value for name, value in self.ratios.items()})
        metrics.update({f"gap[{name}]": value for name, value in self.gaps.items()})
        if self.balancer is not None:
            metrics.update(self.balancer.report())
        return metrics

    def _measure_ratios(self, states: States) -> None:
        """Weighted gradient norms at the hub node, as ratios against the image's pull."""
        if self.z_node == "s":
            self._measure_vote_ratios(states)
            return
        norms: dict[str, float] = {}
        for edge in self.model.energy_edges:
            if self.z_node not in edge.sources:
                continue
            (gradient,) = torch.autograd.grad(edge.energy(states), states[self.z_node], retain_graph=True)
            norms[edge.dst] = max(gradient.norm(dim=1).mean().item(), 1e-12)
        image_pull = norms.get("h_img", norms.get(self.input_node))
        if image_pull is None:
            return
        self._store_ratios(image_pull, norms, skip={"h_img", self.input_node})

    def _measure_vote_ratios(self, states: States) -> None:
        """Weighted gradient norm each vote `z_* -> s` puts on `s`, against the image vote."""
        norms: dict[str, float] = {}
        for edge in self.model.energy_edges:
            if edge.dst != self.z_node or len(edge.sources) != 1:
                continue
            src = edge.sources[0]
            (gradient,) = torch.autograd.grad(edge.energy(states), states[self.z_node], retain_graph=True)
            leaf = {"z_img": "image", "z_msg": "msg", "z_lab": "label"}.get(src, src)
            norms[leaf] = max(gradient.norm(dim=1).mean().item(), 1e-12)
        image_pull = norms.get(self.input_node)
        if image_pull is None:
            return
        self._store_ratios(image_pull, norms, skip={self.input_node})

    def _store_ratios(self, image_pull: float, norms: dict[str, float], skip: set[str]) -> None:
        """Smooth each non-image pull into `self.ratios` as image over that pull."""
        for leaf, norm in norms.items():
            if leaf in skip:
                continue
            key = f"image/{leaf}"
            value = image_pull / norm
            previous = self.ratios.get(key, value)
            # Same smoothing as the balancer, so the two logs move together.
            self.ratios[key] = previous + 0.5 * (value - previous)
