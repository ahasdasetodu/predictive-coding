import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

States = dict[str, Tensor]


def squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    """Half the squared error, summed over features and over the batch."""
    # Summing over the batch keeps each sample's settling gradient independent of batch size.
    return 0.5 * (prediction - target).pow(2).sum()


class GaussianEnergy(nn.Module):
    """Gaussian negative log-likelihood with one learned variance for a whole edge.

    `squared_error` is this with the variance pinned to one. Learning it lets an edge
    report how well it actually predicts, so settling weighs errors in standard
    deviations rather than raw units.
    """

    def __init__(self, min_log_var: float = -4.0) -> None:
        super().__init__()
        if min_log_var >= 0.0:
            raise ValueError("min_log_var must be negative so a variance of one stays reachable")
        self.min_log_var = min_log_var
        # Softplus keeps the variance above its floor with a gradient everywhere, so an edge
        # that has bottomed out can still climb back when its residual grows. A hard clamp
        # would zero the gradient and strand it. Start at a variance of one.
        offset = math.log(math.expm1(-min_log_var))
        self.raw_log_var = nn.Parameter(torch.tensor(min_log_var + offset))

    @property
    def log_var(self) -> Tensor:
        """The learned log-variance, held above `min_log_var`."""
        return self.min_log_var + F.softplus(self.raw_log_var - self.min_log_var)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Precision-weighted error plus the Gaussian normaliser, summed like `squared_error`."""
        log_var = self.log_var
        error = 0.5 * (prediction - target).pow(2).sum() / log_var.exp()
        # The normaliser is what stops the variance growing without bound; it needs the same
        # batch-summed scale as the error term, hence numel rather than the feature count.
        return error + 0.5 * target.numel() * log_var


@dataclass(eq=False)
class Edge:
    """A directed prediction of the `dst` state from one or more source states.

    Several sources are concatenated in order before entering `module`.
    """

    src: str | tuple[str, ...]
    dst: str
    module: nn.Module
    loss: Callable[[Tensor, Tensor], Tensor] = squared_error
    weight: float = 1.0

    @property
    def sources(self) -> tuple[str, ...]:
        """The source node names, in the order they are concatenated."""
        return (self.src,) if isinstance(self.src, str) else self.src

    @property
    def name(self) -> str:
        """Identify the edge by its endpoints."""
        return f"{'+'.join(self.sources)}->{self.dst}"

    def predict(self, states: Mapping[str, Tensor]) -> Tensor:
        """Predict the destination state from the source states."""
        return self.module(torch.cat([states[name] for name in self.sources], dim=1))

    def energy(self, states: Mapping[str, Tensor]) -> Tensor:
        """Weighted error between the destination state and this edge's prediction."""
        return self.weight * self.loss(self.predict(states), states[self.dst])


InitOrder = Literal["fixed", "reachable"]


class PCGraph(nn.Module):
    """State nodes joined by energy edges and initialisation edges.

    `init_order="fixed"` runs the init edges in list order and expects every source to
    be present. `init_order="reachable"` fills with `proposal_edges` first, then
    `readout_edges`, from the clamped nodes, so the same graph serves any clamped subset.
    When those lists are omitted they are inferred: own-module init edges propose,
    energy-module init edges are the read-off.
    """

    def __init__(
        self,
        nodes: Mapping[str, int],
        energy_edges: Sequence[Edge],
        init_edges: Sequence[Edge] = (),
        init_order: InitOrder = "fixed",
        proposal_edges: Sequence[Edge] | None = None,
        readout_edges: Sequence[Edge] | None = None,
    ) -> None:
        super().__init__()
        self.nodes: dict[str, int] = dict(nodes)
        self.energy_edges: list[Edge] = list(energy_edges)
        self.init_order: InitOrder = init_order
        if proposal_edges is None and readout_edges is None:
            # Own-module init edges propose; energy-module init edges are read-off fills.
            self.init_edges: list[Edge] = list(init_edges)
            shared = {id(edge.module) for edge in self.energy_edges}
            self.proposal_edges: list[Edge] = [edge for edge in self.init_edges if id(edge.module) not in shared]
            self.readout_edges: list[Edge] = [edge for edge in self.init_edges if id(edge.module) in shared]
        else:
            self.proposal_edges = list(proposal_edges or ())
            self.readout_edges = list(readout_edges or ())
            self.init_edges = list(init_edges) if init_edges else self.proposal_edges + self.readout_edges
        self._check_edges(self.energy_edges)
        self._check_edges(self.init_edges)
        edges = self.energy_edges + self.init_edges
        # An init edge may reuse an energy module, so the same module must be registered once.
        self.edge_modules = nn.ModuleList(_unique_modules(edges))

    def log_variances(self) -> dict[str, float]:
        """The learned log-variance of every energy edge that has one, keyed by edge name."""
        return {
            edge.name: edge.loss.log_var.item()
            for edge in self.energy_edges
            if isinstance(edge.loss, GaussianEnergy)
        }

    def edge_energies(
        self,
        states: Mapping[str, Tensor],
        edges: Sequence[Edge] | None = None,
    ) -> dict[str, Tensor]:
        """Return each edge's weighted energy, keyed by edge name."""
        if edges is None:
            edges = self.energy_edges
        return {edge.name: edge.energy(states) for edge in edges}

    def energy(
        self,
        states: Mapping[str, Tensor],
        edges: Sequence[Edge] | None = None,
    ) -> Tensor:
        """Sum the weighted edge energies; defaults to the energy edges."""
        terms = list(self.edge_energies(states, edges).values())
        if not terms:
            raise ValueError("energy needs at least one edge")
        return torch.stack(terms).sum()

    def learned_init_edges(self) -> list[Edge]:
        """Proposal edges that own their module.

        A proposal may reuse an energy encoder. That edge is already trained by the
        energy, so a second copy loss on it would count the same residual twice.
        """
        shared = self._shared_module_ids()
        return [edge for edge in self.proposal_edges if id(edge.module) not in shared]

    def readout_init_edges(self) -> list[Edge]:
        """Phase-2 init edges: a free leaf starts at its own top-down prediction."""
        return list(self.readout_edges)

    def propagate(self, clamped: Mapping[str, Tensor]) -> States:
        """Propose starting states for the free nodes from the clamped ones.

        Fixed order fills every node. Reachable order fills what the init edges can reach
        and leaves the rest absent; the regime starts those at zero.
        """
        if self.init_order == "reachable":
            return self._propagate_reachable(clamped)
        return self._propagate_fixed(clamped)

    def proposing_edges(self, clamped: Iterable[str]) -> list[Edge]:
        """The learned init edges that would run from these clamped nodes under reachable order.

        Read-off edges are excluded: they reuse energy modules and are trained by the energy.
        """
        reached = _reachable(set(clamped), self.learned_init_edges())
        return [
            edge
            for edge in self.learned_init_edges()
            if all(name in reached for name in edge.sources) and edge.dst not in set(clamped)
        ]

    def _propagate_reachable(self, clamped: Mapping[str, Tensor]) -> States:
        """Proposal edges fill bottom-up, then readout edges fill the remaining leaves."""
        states = dict(clamped)
        _fill_reachable(states, self.proposal_edges)
        _fill_reachable(states, self.readout_edges)
        return states

    def _propagate_fixed(self, clamped: Mapping[str, Tensor]) -> States:
        """Run the init edges in list order, filling every node that is not clamped."""
        states = dict(clamped)
        for edge in self.init_edges:
            if edge.dst in clamped:
                continue
            if edge.dst in states:
                raise ValueError(f"init edge {edge.name} overwrites {edge.dst}")
            missing = [name for name in edge.sources if name not in states]
            if missing:
                raise KeyError(f"init edge {edge.name} runs before {missing} have states")
            states[edge.dst] = edge.predict(states)
        return states

    def _shared_module_ids(self) -> set[int]:
        """Identities of the modules the energy edges own."""
        return {id(edge.module) for edge in self.energy_edges}

    def _check_edges(self, edges: Sequence[Edge]) -> None:
        """Reject edges with unknown endpoints or duplicate names within one set."""
        seen: set[str] = set()
        for edge in edges:
            for node in (*edge.sources, edge.dst):
                if node not in self.nodes:
                    raise ValueError(f"edge {edge.name} uses unknown node {node!r}")
            if edge.name in seen:
                raise ValueError(f"duplicate edge {edge.name}")
            seen.add(edge.name)


def _reachable(filled: set[str], edges: Sequence[Edge]) -> set[str]:
    """Nodes that `edges` can fill starting from `filled`, by repeated passes."""
    reached = set(filled)
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.dst not in reached and all(name in reached for name in edge.sources):
                reached.add(edge.dst)
                changed = True
    return reached


def _fill_reachable(states: States, edges: Sequence[Edge]) -> None:
    """Fill every node `edges` can reach from the filled states, in place.

    A node waits until every reachable edge into it can run, then takes the mean of
    their predictions. Two init edges into one node are two estimates of the same
    settled state, so the mean uses both and keeps either usable on its own.
    """
    reached = _reachable(set(states), edges)
    pending = reached - set(states)
    while pending:
        progressed = False
        for node in sorted(pending):
            proposals = [
                edge
                for edge in edges
                if edge.dst == node and all(name in reached for name in edge.sources)
            ]
            if not all(name in states for edge in proposals for name in edge.sources):
                continue
            states[node] = torch.stack([edge.predict(states) for edge in proposals]).mean(dim=0)
            pending.remove(node)
            progressed = True
        if not progressed:
            raise ValueError(f"init edges into {sorted(pending)} depend on each other")


def _unique_modules(edges: Sequence[Edge]) -> list[nn.Module]:
    """Modules and learned losses on `edges`, each object once."""
    modules: list[nn.Module] = []
    seen: set[int] = set()
    for edge in edges:
        candidates = [edge.module]
        if isinstance(edge.loss, nn.Module):
            candidates.append(edge.loss)
        for module in candidates:
            if id(module) in seen:
                continue
            seen.add(id(module))
            modules.append(module)
    return modules
