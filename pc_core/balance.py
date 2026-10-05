import math
from dataclasses import dataclass

import torch

from pc_core.graph import Edge, PCGraph, States


@dataclass(frozen=True)
class BalanceConfig:
    """How often to measure gradient balance and how hard to correct it."""

    interval: int = 20
    step: float = 0.3
    momentum: float = 0.5
    anchor_data: bool = True


class EdgeBalancer:
    """Rescale energy-edge weights until every edge has equal say at the nodes it touches.

    Accuracy peaks where the gradient terms arriving at a node have equal norm, so
    measuring that imbalance and correcting it replaces a hand-tuned weight per edge.
    """

    def __init__(self, graph: PCGraph, config: BalanceConfig = BalanceConfig()) -> None:
        self.graph = graph
        self.config = config
        self.norms: dict[tuple[str, str], float] = {}
        self.clamped: set[str] = set()
        self.observations = 0

    def observe(self, states: States) -> None:
        """Measure the imbalance every `interval` calls, and correct the weights when it does."""
        self.observations += 1
        if self.observations % self.config.interval:
            return
        self._measure(states)
        self._correct()

    def report(self) -> dict[str, float]:
        """Each energy edge's weight, plus how far from equal say the worst node still is."""
        spreads = [
            max(norms.values()) / min(norms.values())
            for norms in self._by_node().values()
            if len(norms) > 1
        ]
        weights = {f"w[{edge.name}]": edge.weight for edge in self.graph.energy_edges}
        return {**weights, "imbalance": max(spreads, default=1.0)}

    def _measure(self, states: States) -> None:
        """Per-sample gradient norm that each edge imposes on each free node it touches."""
        free = [name for name, state in states.items() if state.requires_grad]
        self.clamped = set(states) - set(free)
        for edge in self.graph.energy_edges:
            incident = [name for name in free if name in (*edge.sources, edge.dst)]
            if not incident:
                continue
            gradients = torch.autograd.grad(
                edge.energy(states),
                [states[name] for name in incident],
                retain_graph=True,
            )
            for name, gradient in zip(incident, gradients):
                norm = max(gradient.norm(dim=1).mean().item(), 1e-12)
                key = (edge.name, name)
                previous = self.norms.get(key, norm)
                self.norms[key] = previous + (1 - self.config.momentum) * (norm - previous)

    def _by_node(self) -> dict[str, dict[str, float]]:
        """The measured gradient norms regrouped by the node they were measured at."""
        by_node: dict[str, dict[str, float]] = {}
        for (edge_name, node), norm in self.norms.items():
            by_node.setdefault(node, {})[edge_name] = norm
        return by_node

    def _correct(self) -> None:
        """Step each weight towards equal say, then renormalise so the largest weight is one."""
        imbalances: dict[str, list[float]] = {}
        for norms in self._by_node().values():
            if len(norms) < 2:
                continue
            # The norms already include the current weights, so equal norms is the fixed point.
            reference = sum(math.log(norm) for norm in norms.values()) / len(norms)
            for edge_name, norm in norms.items():
                imbalances.setdefault(edge_name, []).append(math.log(norm) - reference)

        for edge in self.graph.energy_edges:
            excess = imbalances.get(edge.name)
            if excess:
                edge.weight *= math.exp(-self.config.step * sum(excess) / len(excess))
        # Only ratios matter, so pin the scale. It has to be pinned to the edges that touch
        # clamped data: an edge between two free nodes can have its residual settled towards
        # zero, so it looks quiet, and normalising on the loudest edge then lets the data
        # edges be quietened to match it until the energy is all internal consistency.
        anchors = [edge for edge in self.graph.energy_edges if self._touches_data(edge)]
        loudest = max(edge.weight for edge in anchors or self.graph.energy_edges)
        for edge in self.graph.energy_edges:
            edge.weight /= loudest

    def _touches_data(self, edge: Edge) -> bool:
        """Whether either end of this edge is a clamped node, so it carries real data."""
        return self.config.anchor_data and bool(self.clamped & {*edge.sources, edge.dst})
