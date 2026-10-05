from collections.abc import Callable, Sequence
from typing import Literal

from torch import Tensor, nn

from pc_core.graph import Edge, GaussianEnergy, PCGraph, squared_error

ActivationFactory = Callable[[], nn.Module]
EdgeLoss = Callable[[Tensor, Tensor], Tensor]
InitScheme = Literal["forward", "leapfrog"]


def mlp_graph(
    hidden_dims: int | Sequence[int] = 128,
    input_dim: int = 28 * 28,
    output_dim: int = 10,
    init: InitScheme | None = "forward",
    output_activation: ActivationFactory = nn.Identity,
    learn_precision: bool = False,
    min_log_var: float = -4.0,
    image_weight: float = 1.0,
) -> PCGraph:
    """Generative chain label -> hidden -> image, with optional bottom-up init edges.

    `init="forward"` steps up one node at a time. `init="leapfrog"` jumps two nodes up,
    then fills the skipped node from the nodes on both sides of it.

    `learn_precision` gives every energy edge its own learned variance instead of a
    fixed weight of one. Init edges keep squared error; they sit outside the energy.

    `image_weight` scales the energy of the edge into the image. Below one it keeps the
    784 pixels from dominating settling, which costs the label its say.
    """
    widths = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)
    if not widths:
        raise ValueError("hidden_dims must contain at least one width")
    names = ["label", *(f"h{i + 1}" for i in range(len(widths))), "image"]
    nodes = dict(zip(names, [output_dim, *widths, input_dim]))
    # The label and image ends hold data values, so predictions into them are not squashed.
    end_activation = {"label": output_activation, "image": output_activation}

    def edge(src: str | tuple[str, ...], dst: str, loss: EdgeLoss = squared_error) -> Edge:
        """Build a linear prediction edge with the activation its destination needs."""
        sources = (src,) if isinstance(src, str) else src
        in_dim = sum(nodes[name] for name in sources)
        activation = end_activation.get(dst, nn.Tanh)
        return Edge(src, dst, nn.Sequential(nn.Linear(in_dim, nodes[dst]), activation()), loss)

    def energy_loss() -> EdgeLoss:
        """A fresh loss for one energy edge; each edge owns its own variance."""
        return GaussianEnergy(min_log_var) if learn_precision else squared_error

    energy_edges = [edge(above, below, energy_loss()) for above, below in zip(names, names[1:])]
    energy_edges[-1].weight = image_weight
    bottom_up = names[::-1]
    if init == "forward":
        init_edges = [edge(below, above) for below, above in zip(bottom_up, bottom_up[1:])]
    elif init == "leapfrog":
        init_edges = _leapfrog_edges(bottom_up, edge)
    else:
        init_edges = []
    return PCGraph(nodes, energy_edges, init_edges)


def _leapfrog_edges(
    bottom_up: list[str],
    edge: Callable[[str | tuple[str, ...], str], Edge],
) -> list[Edge]:
    """Jump from each known node two up, then fill the gap from both sides."""
    edges: list[Edge] = []
    i = 0
    while i < len(bottom_up) - 1:
        if i + 2 < len(bottom_up):
            low, mid, high = bottom_up[i : i + 3]
            edges += [edge(low, high), edge((low, high), mid)]
            i += 2
        else:
            edges.append(edge(bottom_up[i], bottom_up[i + 1]))
            i += 1
    return edges
