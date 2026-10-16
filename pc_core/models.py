from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

from torch import Tensor, nn

from pc_core.graph import Edge, GaussianEnergy, PCGraph, squared_error
from pc_core.messages import MESSAGE_DIM

ActivationFactory = Callable[[], nn.Module]
EdgeLoss = Callable[[Tensor, Tensor], Tensor]
InitScheme = Literal["forward", "leapfrog", "skip2"]
GraphTopology = Literal["hub", "consensus"]

# Nodes that hold data values; predictions into them are not squashed.
DATA_NODES: frozenset[str] = frozenset({"image", "msg", "label"})
# Consensus latents stay linear so the image-to-label init path has two Tanh hops, like the hub.
LATENT_NODES: frozenset[str] = frozenset({"z_img", "z_msg", "z_lab", "s"})
IDENTITY_NODES: frozenset[str] = DATA_NODES | LATENT_NODES
HUB_ENERGY_EDGES: tuple[str, ...] = ("z->h_img", "h_img->image", "z->msg", "z->label")
HUB_INIT_EDGES: tuple[str, ...] = ("image->h_img", "h_img->z", "msg->z", "label->z")
CONSENSUS_PROPOSAL_EDGES: tuple[str, ...] = (
    "image->h_img",
    "h_img->z_img",
    "z_img->s",
    "msg->h_msg",
    "h_msg->z_msg",
    "z_msg->s",
    "label->h_lab",
    "h_lab->z_lab",
    "z_lab->s",
)
CONSENSUS_READOUT_EDGES: tuple[str, ...] = (
    "s->z_img",
    "z_img->h_img",
    "h_img->image",
    "s->z_msg",
    "z_msg->h_msg",
    "h_msg->msg",
    "s->z_lab",
    "z_lab->h_lab",
    "h_lab->label",
)
CONSENSUS_ENERGY_EDGES: tuple[str, ...] = CONSENSUS_PROPOSAL_EDGES + CONSENSUS_READOUT_EDGES
# data, hidden, latent for each I/O tower.
CONSENSUS_TOWERS: dict[str, tuple[str, str, str]] = {
    "image": ("image", "h_img", "z_img"),
    "msg": ("msg", "h_msg", "z_msg"),
    "label": ("label", "h_lab", "z_lab"),
}
_Z_TO_DATA: dict[str, str] = {latent: data for data, _, latent in CONSENSUS_TOWERS.values()}
_NODE_TO_TOWER: dict[str, tuple[str, str, str]] = {
    name: tower for tower in CONSENSUS_TOWERS.values() for name in tower
}


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
    then fills the skipped node from the nodes on both sides of it. `init="skip2"`
    learns a bottom-up edge from every node to the node two above it, and fills the
    one node beside the image with the generative edge that already predicts it.

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
    elif init == "skip2":
        init_edges = _skip2_edges(bottom_up, edge, energy_edges)
    else:
        init_edges = []
    return PCGraph(nodes, energy_edges, init_edges)


def graph_from_edges(
    nodes: Mapping[str, int],
    energy_edges: Sequence[str],
    init_edges: Sequence[str],
    readout_edges: Sequence[str] = (),
    proposal_edges: Sequence[str] = (),
) -> PCGraph:
    """Build a reachable-order graph from edge strings such as `"a+b->c"`.

    Energy edges and `init_edges` get their own Linear module, with Tanh into hidden
    nodes and identity into data and latent nodes. `proposal_edges` and `readout_edges`
    name energy edges whose module is reused: proposals fill bottom-up from the
    clamped nodes, readouts fill the remaining leaves from their own predictions.
    """
    energy = [_edge_from_text(text, nodes) for text in energy_edges]
    by_name = {edge.name: edge for edge in energy}

    def reused(text: str, kind: str) -> Edge:
        """Copy an energy edge as an init edge that shares its module."""
        name = _edge_from_text(text, nodes).name
        if name not in by_name:
            raise ValueError(f"{kind} edge {name} is not an energy edge")
        shared = by_name[name]
        return Edge(shared.src, shared.dst, shared.module)

    proposal = [_edge_from_text(text, nodes) for text in init_edges]
    proposal.extend(reused(text, "proposal") for text in proposal_edges)
    readout = [reused(text, "readout") for text in readout_edges]
    return PCGraph(
        nodes,
        energy,
        proposal + readout,
        init_order="reachable",
        proposal_edges=proposal,
        readout_edges=readout,
    )


def hub_graph(
    z_dim: int = 128,
    h_dim: int = 256,
    msg_dim: int = MESSAGE_DIM,
    input_dim: int = 28 * 28,
    output_dim: int = 10,
) -> PCGraph:
    """One latent `z` that predicts the image (through `h_img`), the clue, and the label.

    Every leaf proposes `z` through a learned init edge, and every energy edge doubles
    as a read-off for a leaf that is not clamped.
    """
    nodes = {"image": input_dim, "msg": msg_dim, "z": z_dim, "h_img": h_dim, "label": output_dim}
    return graph_from_edges(nodes, HUB_ENERGY_EDGES, HUB_INIT_EDGES, readout_edges=HUB_ENERGY_EDGES)


def consensus_graph(
    z_dim: int = 128,
    h_img: int = 256,
    h_msg: int = 128,
    h_lab: int = 128,
    msg_dim: int = MESSAGE_DIM,
    input_dim: int = 28 * 28,
    output_dim: int = 10,
    share_init: bool = True,
) -> PCGraph:
    """Three bidirectional towers whose informed latents vote for a shared state `s`.

    Top-down energy edges are the read-off. When `share_init` is true, bottom-up
    energy modules are also the proposal init. When false, those hops get their
    own modules so image-only cross-entropy does not share weights with energy.
    Unobserved towers do not vote: `informed_energy_edges` drops those.
    """
    nodes = {
        "image": input_dim,
        "h_img": h_img,
        "z_img": z_dim,
        "msg": msg_dim,
        "h_msg": h_msg,
        "z_msg": z_dim,
        "label": output_dim,
        "h_lab": h_lab,
        "z_lab": z_dim,
        "s": z_dim,
    }
    learned = () if share_init else CONSENSUS_PROPOSAL_EDGES
    shared = CONSENSUS_PROPOSAL_EDGES if share_init else ()
    return graph_from_edges(
        nodes,
        CONSENSUS_ENERGY_EDGES,
        learned,
        proposal_edges=shared,
        readout_edges=CONSENSUS_READOUT_EDGES,
    )


def informed_energy_edges(graph: PCGraph, clamped: Iterable[str]) -> list[Edge]:
    """Energy edges whose sources are informed by the clamped data nodes.

    Broadcasts `s -> z_*` and top-down tower edges always run. A vote `z_i -> s`
    and that tower's bottom-up edges run only when the tower's data node is clamped.
    """
    present = set(clamped)
    return [edge for edge in graph.energy_edges if _consensus_edge_informed(edge, present)]


def _consensus_edge_informed(edge: Edge, present: set[str]) -> bool:
    """Whether this consensus energy edge should run given the clamped data nodes."""
    if len(edge.sources) != 1:
        return True
    src = edge.sources[0]
    dst = edge.dst
    if src == "s" and dst in _Z_TO_DATA:
        return True
    if dst == "s" and src in _Z_TO_DATA:
        return _Z_TO_DATA[src] in present
    tower = _NODE_TO_TOWER.get(src) or _NODE_TO_TOWER.get(dst)
    if tower is None:
        return True
    data, hidden, latent = tower
    if (src, dst) in ((data, hidden), (hidden, latent)):
        return data in present
    return True


def _edge_from_text(text: str, nodes: Mapping[str, int]) -> Edge:
    """Parse `"a+b->c"` into a linear edge with the activation its destination needs."""
    if text.count("->") != 1:
        raise ValueError(f"edge {text!r} must be written as src->dst")
    left, dst = (part.strip() for part in text.split("->"))
    sources = tuple(part.strip() for part in left.split("+"))
    for name in (*sources, dst):
        if name not in nodes:
            raise ValueError(f"edge {text!r} uses unknown node {name!r}")
    in_dim = sum(nodes[name] for name in sources)
    activation: nn.Module = nn.Identity() if dst in IDENTITY_NODES else nn.Tanh()
    src: str | tuple[str, ...] = sources[0] if len(sources) == 1 else sources
    return Edge(src, dst, nn.Sequential(nn.Linear(in_dim, nodes[dst]), activation))


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


def _skip2_edges(
    bottom_up: list[str],
    edge: Callable[[str | tuple[str, ...], str], Edge],
    energy_edges: Sequence[Edge],
) -> list[Edge]:
    """Land a learned skip on every node two steps up, and reuse one generative edge.

    Bottom-up, node 0 is the image. The skip from node i lands on node i+2, which
    covers every node except node 1. After image -> node 2, the existing energy edge
    node 2 -> node 1 writes that remaining node. Later skips then run from nodes
    whose states exist. No new top-down module is created, and no skip landing is
    overwritten.
    """
    if len(bottom_up) < 3:
        raise ValueError("skip2 needs a node two steps above the image")
    landing = bottom_up[2]
    gap = bottom_up[1]
    shared = next(
        (existing for existing in energy_edges if existing.sources == (landing,) and existing.dst == gap),
        None,
    )
    if shared is None:
        raise ValueError(f"skip2 needs the generative edge {landing}->{gap}")
    skips = [edge(bottom_up[i], bottom_up[i + 2]) for i in range(len(bottom_up) - 2)]
    return [skips[0], shared, *skips[1:]]
