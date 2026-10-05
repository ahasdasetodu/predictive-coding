"""Streamlit view of the predictive-coding graphs used in the current experiments.

Run with: uv run streamlit run app.py
"""

import html
from dataclasses import dataclass

import streamlit as st
import streamlit.components.v1 as components
from torch import nn

from pc_core.graph import Edge, PCGraph
from pc_core.models import InitScheme, mlp_graph

INPUT_DIM = 28 * 28
OUTPUT_DIM = 10
HIDDEN_SIZES: list[tuple[str, list[int]]] = [
    ("128", [128]),
    ("256 → 128", [256, 128]),
    ("256 → 128 → 64", [256, 128, 64]),
    ("256 → 256 → 128 → 64", [256, 256, 128, 64]),
]
SCHEMES: list[tuple[str, InitScheme | None, str]] = [
    ("Forward", "forward", "Bottom-up, one node at a time."),
    ("Leapfrog", "leapfrog", "Jump two nodes up, then fill the skipped node from both sides."),
    ("No init", None, "Energy edges only. Free states start at zero."),
]

ENERGY = "#1d4e89"
STEP = "#a16207"
JUMP = "#c2410c"
FILL = "#6d28d9"
KIND_COLOR = {"step": STEP, "jump": JUMP, "fill": FILL}

NODE_W = 148.0
NODE_H = 52.0
V_GAP = 36.0
PAD = 20.0
STEP_BOW = 38.0
BOW_STEP = 34.0


@dataclass(frozen=True)
class _NodeBox:
    """Where one state node is drawn."""

    name: str
    dim: int
    x: float
    y: float

    @property
    def cx(self) -> float:
        """Horizontal centre of the box."""
        return self.x + NODE_W / 2

    @property
    def cy(self) -> float:
        """Vertical centre of the box."""
        return self.y + NODE_H / 2


def _layers(module: nn.Module) -> list[nn.Module]:
    """Return the layers of an edge module, or the module itself when it has none."""
    children = list(module.children())
    return children or [module]


def _activation_name(module: nn.Module) -> str:
    """Name the last layer, treating an identity as a linear edge."""
    last = _layers(module)[-1]
    if isinstance(last, nn.Identity):
        return "identity"
    return last.__class__.__name__.lower()


def _parameter_count(module: nn.Module) -> int:
    """Count parameters in one edge module."""
    return sum(parameter.numel() for parameter in module.parameters())


def _bow(span: int) -> float:
    """Push longer jumps farther from the node column than neighbour steps."""
    return STEP_BOW + max(span - 1, 0) * BOW_STEP


def _kind(edge: Edge, index: dict[str, int]) -> str:
    """Classify an init edge as a neighbour step, a two-node jump, or a two-sided fill."""
    if len(edge.sources) > 1:
        return "fill"
    span = abs(index[edge.dst] - index[edge.sources[0]])
    return "jump" if span >= 2 else "step"


def _shape(edge: Edge, nodes: dict[str, int]) -> str:
    """Describe an edge as source widths joined into the destination width."""
    sources = "+".join(str(nodes[name]) for name in edge.sources)
    return f"{sources}→{nodes[edge.dst]}"


def _spread(count: int) -> list[float]:
    """Even offsets centred on zero, so a single arrow stays unshifted."""
    if count == 1:
        return [0.0]
    step = 9.0
    origin = -step * (count - 1) / 2
    return [origin + i * step for i in range(count)]


def _port_shifts(edges: list[Edge], index: dict[str, int]) -> dict[tuple[int, int, str], float]:
    """Separate arrows that share a node, grouping them by the side they use."""
    groups: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for edge_i, edge in enumerate(edges):
        dst_i = index[edge.dst]
        for source_i, source in enumerate(edge.sources):
            upward = dst_i < index[source]
            groups.setdefault((source, "up" if upward else "down"), []).append((edge_i, source_i, "src"))
            groups.setdefault((edge.dst, "down" if upward else "up"), []).append((edge_i, source_i, "dst"))
    shifts: dict[tuple[int, int, str], float] = {}
    for (_node, side), arrows in groups.items():
        bias = -10.0 if side == "up" else 10.0
        for arrow, extra in zip(arrows, _spread(len(arrows))):
            shifts[arrow] = bias + extra
    return shifts


def _layout(names: list[str], nodes: dict[str, int], right_bow: float) -> tuple[dict[str, _NodeBox], float, float]:
    """Place the chain top to bottom and return boxes plus the canvas size."""
    boxes = {
        name: _NodeBox(name, nodes[name], PAD, PAD + i * (NODE_H + V_GAP))
        for i, name in enumerate(names)
    }
    width = PAD + NODE_W + right_bow + PAD + 16
    height = PAD + len(names) * NODE_H + max(len(names) - 1, 0) * V_GAP + PAD
    return boxes, width, height


def _curve(start: tuple[float, float], end: tuple[float, float], bow: float) -> str:
    """Cubic bezier that leaves each node horizontally and bows by `bow` pixels."""
    x0, y0 = start
    x1, y1 = end
    return (
        f"M {x0:.1f},{y0:.1f} "
        f"C {x0 + bow:.1f},{y0:.1f} {x1 + bow:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"
    )


def _curve_mid(start: tuple[float, float], end: tuple[float, float], bow: float) -> tuple[float, float]:
    """Point halfway along `_curve`."""
    x0, y0 = start
    x1, y1 = end
    return (0.5 * (x0 + x1) + 0.75 * bow, 0.5 * (y0 + y1))


def _marker(name: str, color: str) -> str:
    """Arrowhead whose tip sits on the end of a stroke."""
    return (
        f'<marker id="{name}" viewBox="0 0 10 8" refX="9" refY="4" '
        f'markerWidth="8" markerHeight="6" orient="auto">'
        f'<path d="M0,0 L10,4 L0,8 Z" fill="{color}"/></marker>'
    )


def _node_svg(box: _NodeBox) -> str:
    """Draw one state: clamped ends hold data, hidden nodes are free."""
    clamped = box.name in ("label", "image")
    fill = "#e8eef6" if clamped else "#faf7f2"
    role = "clamped" if clamped else "free"
    name = html.escape(box.name)
    return (
        f'<g>'
        f'<rect x="{box.x:.1f}" y="{box.y:.1f}" width="{NODE_W:.1f}" height="{NODE_H:.1f}" '
        f'rx="8" fill="{fill}" stroke="#d6d3d1"/>'
        f'<text x="{box.cx:.1f}" y="{box.cy - 8:.1f}" text-anchor="middle" '
        f'dominant-baseline="central" font-size="15" font-weight="600" fill="#1c1917">{name}</text>'
        f'<text x="{box.cx:.1f}" y="{box.cy + 11:.1f}" text-anchor="middle" '
        f'dominant-baseline="central" font-size="12" fill="#57534e">{box.dim} · {role}</text>'
        f'</g>'
    )


def _energy_svg(edge: Edge, boxes: dict[str, _NodeBox], index: dict[str, int]) -> str:
    """Draw one top-down energy edge, straight when it links neighbours."""
    source = edge.sources[0]
    src, dst = boxes[source], boxes[edge.dst]
    title = html.escape(f"{edge.name} energy")
    if index[edge.dst] == index[source] + 1:
        path = f"M {src.cx:.1f},{src.y + NODE_H:.1f} L {dst.cx:.1f},{dst.y:.1f}"
    else:
        span = abs(index[edge.dst] - index[source])
        path = _curve((src.x, src.cy), (dst.x, dst.cy), -_bow(span))
    return (
        f'<path d="{path}" fill="none" stroke="{ENERGY}" stroke-width="1.8" '
        f'marker-end="url(#arrow-energy)"><title>{title}</title></path>'
    )


def _init_marks(
    edge: Edge,
    edge_i: int,
    boxes: dict[str, _NodeBox],
    index: dict[str, int],
    shifts: dict[tuple[int, int, str], float],
) -> tuple[str, str]:
    """Curves and the propagate-order badge for one init edge."""
    kind = _kind(edge, index)
    color = KIND_COLOR[kind]
    dst = boxes[edge.dst]
    mids: list[tuple[float, float]] = []
    paths: list[str] = []
    for source_i, source in enumerate(edge.sources):
        span = abs(index[edge.dst] - index[source])
        bow = _bow(span)
        start = (boxes[source].x + NODE_W, boxes[source].cy + shifts[(edge_i, source_i, "src")])
        end = (dst.x + NODE_W, dst.cy + shifts[(edge_i, source_i, "dst")])
        paths.append(_curve(start, end, bow))
        mids.append(_curve_mid(start, end, bow))
    title = html.escape(f"{edge_i + 1}. {edge.name} {kind}")
    curves = "".join(
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'marker-end="url(#arrow-{kind})"><title>{title}</title></path>'
        for path in paths
    )
    lx = sum(x for x, _y in mids) / len(mids) + 12
    ly = sum(y for _x, y in mids) / len(mids)
    badge = (
        f'<g><title>{title}</title>'
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="9" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
        f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="central" '
        f'font-size="11" font-weight="600" fill="#ffffff">{edge_i + 1}</text></g>'
    )
    return curves, badge


def _diagram(graph: PCGraph) -> tuple[str, int]:
    """Render the graph as an SVG document. Returns the document and its height."""
    names = list(graph.nodes)
    index = {name: i for i, name in enumerate(names)}
    right_bow = 0.0
    for edge in graph.init_edges:
        spans = [abs(index[edge.dst] - index[source]) for source in edge.sources]
        right_bow = max(right_bow, _bow(max(spans)) + 20)
    boxes, width, height = _layout(names, graph.nodes, right_bow)
    shifts = _port_shifts(graph.init_edges, index)
    markers = _marker("arrow-energy", ENERGY) + "".join(
        _marker(f"arrow-{kind}", color) for kind, color in KIND_COLOR.items()
    )
    nodes = "".join(_node_svg(boxes[name]) for name in names)
    energy = "".join(_energy_svg(edge, boxes, index) for edge in graph.energy_edges)
    curves: list[str] = []
    badges: list[str] = []
    for edge_i, edge in enumerate(graph.init_edges):
        curve, badge = _init_marks(edge, edge_i, boxes, index, shifts)
        curves.append(curve)
        badges.append(badge)
    chain = " → ".join(html.escape(name) for name in names)
    document = (
        '<style>html,body{margin:0;background:#fff}</style>'
        '<div style="display:flex;justify-content:center">'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'style="font-family:ui-sans-serif,system-ui,sans-serif">'
        f"<title>{chain}</title>"
        f"<defs>{markers}</defs>"
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        f"{energy}{''.join(curves)}{nodes}{''.join(badges)}"
        "</svg></div>"
    )
    return document, round(height)


def _edge_row(edge: Edge, nodes: dict[str, int], index: dict[str, int], order: int | None) -> str:
    """One line describing an edge's kind, shape, activation, and size."""
    kind = "energy" if order is None else _kind(edge, index)
    color = ENERGY if order is None else KIND_COLOR[kind]
    prefix = kind if order is None else f"{order} {kind}"
    sources = " + ".join(edge.sources)
    detail = (
        f"{html.escape(sources)} → {html.escape(edge.dst)} · {_shape(edge, nodes)} "
        f"{_activation_name(edge.module)} · {_parameter_count(edge.module):,}"
    )
    return (
        '<div style="margin:0.15rem 0; font-size:0.9rem; line-height:1.35">'
        f'<span style="color:{color}; font-weight:600">{prefix}</span> '
        f'<span style="color:#44403c">{detail}</span></div>'
    )


def _heading(text: str) -> str:
    """Small bold label above an edge list."""
    return f'<div style="font-weight:600; margin:0.6rem 0 0.15rem 0">{html.escape(text)}</div>'


def _column(title: str, blurb: str, graph: PCGraph) -> None:
    """Draw one scheme: the diagram, parameter totals, and both edge lists."""
    names = list(graph.nodes)
    index = {name: i for i, name in enumerate(names)}
    diagram, height = _diagram(graph)
    energy_params = sum(_parameter_count(edge.module) for edge in graph.energy_edges)
    init_params = sum(_parameter_count(edge.module) for edge in graph.init_edges)
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(blurb)
        components.html(diagram, height=height + 4, scrolling=False)
        st.caption(f"{energy_params:,} energy params · {init_params:,} init params")
        if graph.init_edges:
            rows = [
                _edge_row(edge, graph.nodes, index, order)
                for order, edge in enumerate(graph.init_edges, start=1)
            ]
            st.markdown(_heading("Init order") + "".join(rows), unsafe_allow_html=True)
        else:
            st.markdown(_heading("Init") + "Free states start at zero.", unsafe_allow_html=True)
        energy_rows = [_edge_row(edge, graph.nodes, index, None) for edge in graph.energy_edges]
        st.markdown(_heading("Energy") + "".join(energy_rows), unsafe_allow_html=True)


def _legend() -> str:
    """Color key shared by every diagram."""
    items = [
        (ENERGY, "energy, top-down"),
        (STEP, "init step"),
        (JUMP, "init jump"),
        (FILL, "init fill from both sides"),
    ]
    spans = " &nbsp;&nbsp; ".join(
        f'<span style="color:{color}; font-weight:600">● {html.escape(label)}</span>'
        for color, label in items
    )
    return (
        f'<p style="margin:0 0 0.4rem 0">{spans}'
        " &nbsp;&nbsp;·&nbsp;&nbsp; numbers are the order <code>propagate</code> runs init edges</p>"
    )


def main() -> None:
    """Compare forward, leapfrog, and absent initialisation on the experiment widths."""
    st.set_page_config(page_title="PC graphs", layout="wide")
    st.title("Predictive-coding graphs")
    st.markdown(
        "MNIST chain used in the current runs: the label (10) sits at the top, "
        "hidden layers in between, and the image (784) at the bottom. "
        "Energy edges point down the chain. Init edges, when the graph has them, point up it."
    )
    st.markdown(_legend(), unsafe_allow_html=True)

    labels = ["All", *(label for label, _hidden in HIDDEN_SIZES)]
    choice = st.radio("Hidden sizes", labels, horizontal=True)
    selected = HIDDEN_SIZES if choice == "All" else [item for item in HIDDEN_SIZES if item[0] == choice]

    for label, hidden in selected:
        if choice == "All":
            st.subheader(label)
        columns = st.columns(3)
        for column, (title, init, blurb) in zip(columns, SCHEMES):
            graph = mlp_graph(hidden_dims=hidden, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, init=init)
            with column:
                _column(title, blurb, graph)


main()
