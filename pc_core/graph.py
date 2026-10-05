import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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


class PCGraph(nn.Module):
    """State nodes joined by energy edges and initialisation edges."""

    def __init__(
        self,
        nodes: Mapping[str, int],
        energy_edges: Sequence[Edge],
        init_edges: Sequence[Edge] = (),
    ) -> None:
        super().__init__()
        self.nodes: dict[str, int] = dict(nodes)
        self.energy_edges: list[Edge] = list(energy_edges)
        self.init_edges: list[Edge] = list(init_edges)
        self._check_edges(self.energy_edges)
        self._check_edges(self.init_edges)
        edges = self.energy_edges + self.init_edges
        self.edge_modules = nn.ModuleList(
            [edge.module for edge in edges]
            # A learned-variance loss carries parameters, so it has to be registered too.
            + [edge.loss for edge in edges if isinstance(edge.loss, nn.Module)]
        )

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

    def propagate(self, clamped: Mapping[str, Tensor]) -> States:
        """Run the init edges in list order, filling every node that is not clamped."""
        states = dict(clamped)
        for edge in self.init_edges:
            if edge.dst in clamped:
                continue
            missing = [name for name in edge.sources if name not in states]
            if missing:
                raise KeyError(f"init edge {edge.name} runs before {missing} have states")
            states[edge.dst] = edge.predict(states)
        return states

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
