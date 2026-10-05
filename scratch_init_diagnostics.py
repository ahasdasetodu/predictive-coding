"""Throwaway diagnostics for why initialisation degrades in deeper predictive-coding graphs.

For each depth and init scheme it trains briefly, then reports, per node, how far the
initial guess lands from the settled state that the init edges are trained to hit:

- gap growing steadily with depth          -> errors pile up along the chain
- gap spiking at nodes reached by a jump   -> jump edges lack capacity (try --deep-jumps)
- eval gap >> teacher-forced gap near label -> train/eval label mismatch
- leapfrog beats forward-matched            -> structure helps, not just extra parameters

Usage:
    python scratch_init_diagnostics.py
    python scratch_init_diagnostics.py --depths 256,128 256,256,128,64 --deep-jumps
    python scratch_init_diagnostics.py --infer-steps 30 --train-batches 100
"""

import argparse
from dataclasses import replace

import torch
from torch import Tensor, nn

from pc_core.dataset import load_dataset
from pc_core.graph import Edge, PCGraph, States
from pc_core.models import _leapfrog_edges, mlp_graph
from pc_core.regimes import ForwardInitRegime, PCConfig


def deep_jump_leapfrog(hidden_dims: list[int]) -> PCGraph:
    """Leapfrog graph whose two-node jumps use a two-layer module instead of one."""
    base = mlp_graph(hidden_dims, init=None)
    nodes = base.nodes
    names = list(nodes)
    bottom_up = names[::-1]

    def edge(src: str | tuple[str, ...], dst: str) -> Edge:
        """Build a fill or one-hop edge as usual, and a deeper module for jumps."""
        sources = (src,) if isinstance(src, str) else src
        in_dim = sum(nodes[name] for name in sources)
        out_act: nn.Module = nn.Identity() if dst in ("label", "image") else nn.GELU()
        is_jump = isinstance(src, str) and abs(names.index(src) - names.index(dst)) == 2
        if is_jump:
            module = nn.Sequential(
                nn.Linear(in_dim, nodes[dst]), nn.GELU(), nn.Linear(nodes[dst], nodes[dst]), out_act
            )
        else:
            module = nn.Sequential(nn.Linear(in_dim, nodes[dst]), out_act)
        return Edge(src, dst, module)

    return PCGraph(nodes, base.energy_edges, _leapfrog_edges(bottom_up, edge))


def init_param_count(graph: PCGraph) -> int:
    """Number of parameters in a graph's init edges."""
    return sum(p.numel() for edge in graph.init_edges for p in edge.module.parameters())


def matched_forward(hidden_dims: list[int], target_params: int) -> PCGraph:
    """Forward init whose edges get one hidden layer, sized to match `target_params`."""
    base = mlp_graph(hidden_dims, init=None)
    nodes = base.nodes
    bottom_up = list(nodes)[::-1]
    pairs = list(zip(bottom_up, bottom_up[1:]))
    # Each edge holds k * (in + 1 + out) + out parameters, so k follows from the target.
    per_k = sum(nodes[src] + 1 + nodes[dst] for src, dst in pairs)
    fixed = sum(nodes[dst] for _, dst in pairs)
    width = max(1, round((target_params - fixed) / per_k))

    def edge(src: str, dst: str) -> Edge:
        """A two-layer bottom-up edge with the shared matched hidden width."""
        out_act: nn.Module = nn.Identity() if dst in ("label", "image") else nn.GELU()
        module = nn.Sequential(
            nn.Linear(nodes[src], width), nn.GELU(), nn.Linear(width, nodes[dst]), out_act
        )
        return Edge(src, dst, module)

    return PCGraph(nodes, base.energy_edges, [edge(src, dst) for src, dst in pairs])


def build_graph(hidden_dims: list[int], init: str, deep_jumps: bool) -> PCGraph:
    """Pick the graph for one experiment cell."""
    leapfrog = deep_jump_leapfrog(hidden_dims) if deep_jumps else mlp_graph(hidden_dims, init="leapfrog")
    if init == "leapfrog":
        return leapfrog
    if init == "forward-matched":
        return matched_forward(hidden_dims, init_param_count(leapfrog))
    return mlp_graph(hidden_dims, init=init)


def settled_targets(regime: ForwardInitRegime, x: Tensor, y: Tensor) -> States:
    """Settle with image and label clamped, exactly as training does."""
    clamped = {"image": regime._flat(x), "label": regime._one_hot(y)}
    states = regime._free_states(clamped)
    regime.settle(states, regime._free(clamped), regime.config.infer_steps, regime.config.infer_lr)
    return {name: state.detach() for name, state in states.items()}


def relative_gaps(init: States, target: States, names: list[str]) -> dict[str, float]:
    """Squared distance from init to target, relative to the target's squared size."""
    return {
        name: ((init[name] - target[name]).pow(2).sum() / target[name].pow(2).sum().clamp_min(1e-8)).item()
        for name in names
    }


def accuracy(regime: ForwardInitRegime, test: list[tuple[Tensor, Tensor]], eval_steps: int) -> float:
    """Test accuracy with a given number of evaluation settling steps."""
    regime.config = replace(regime.config, eval_steps=eval_steps)
    correct = total = 0
    for x, y in test:
        correct += regime.evaluate_batch((x, y)).correct
        total += y.numel()
    return 100 * correct / total


def run_cell(args: argparse.Namespace, hidden_dims: list[int], init: str, train, test) -> None:
    """Train one configuration briefly and print its per-node gaps and accuracies."""
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = build_graph(hidden_dims, init, args.deep_jumps)
    config = PCConfig(infer_steps=args.infer_steps)
    regime = ForwardInitRegime(graph, config, device=device)
    regime.start_training(1)
    for batch_idx, batch in enumerate(train):
        if batch_idx == args.train_batches:
            break
        regime.train_batch(batch)

    free_bottom_up = [name for name in reversed(graph.nodes) if name != "image"]
    forced_sum = {name: 0.0 for name in free_bottom_up}
    eval_sum = {name: 0.0 for name in free_bottom_up}
    for x, y in test:
        target = settled_targets(regime, x, y)
        with torch.no_grad():
            forced = regime.init_states({"image": target["image"], "label": target["label"]})
            guessed = regime.init_states({"image": target["image"]})
        for name, gap in relative_gaps(forced, target, free_bottom_up).items():
            forced_sum[name] += gap / len(test)
        for name, gap in relative_gaps(guessed, target, free_bottom_up).items():
            eval_sum[name] += gap / len(test)

    jumps = "deep jumps" if init == "leapfrog" and args.deep_jumps else ""
    print(f"\n{hidden_dims} {init} {jumps}".rstrip() + f"  ({init_param_count(graph):,} init params)")
    print("  edges: " + ", ".join(edge.name for edge in graph.init_edges))
    print("  node    teacher-forced gap   eval gap   (relative squared error vs settled state)")
    for name in free_bottom_up:
        forced_text = "   clamped" if name == "label" else f"{forced_sum[name]:10.3f}"
        print(f"  {name:6}  {forced_text}           {eval_sum[name]:8.3f}")
    scores = " | ".join(f"{steps} steps {accuracy(regime, test, steps):.1f}%" for steps in args.eval_steps)
    print(f"  accuracy: {scores}")


def main() -> None:
    """Run every depth and init scheme requested on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", nargs="+", default=["128", "256,128", "256,128,64", "256,256,128,64"])
    parser.add_argument(
        "--inits",
        nargs="+",
        choices=["forward", "forward-matched", "leapfrog"],
        default=["forward", "forward-matched", "leapfrog"],
        help="forward-matched widens forward edges to leapfrog's init parameter count",
    )
    parser.add_argument("--deep-jumps", action="store_true", help="two-layer modules on leapfrog jumps")
    parser.add_argument("--infer-steps", type=int, default=10)
    parser.add_argument("--train-batches", type=int, default=234, help="234 is one MNIST epoch")
    parser.add_argument("--test-batches", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[0, 5, 30])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train, test_loader = load_dataset(batch_size=256)
    test = [batch for _, batch in zip(range(args.test_batches), test_loader)]
    for depth in args.depths:
        hidden_dims = [int(width) for width in depth.split(",")]
        for init in args.inits:
            run_cell(args, hidden_dims, init, train, test)


if __name__ == "__main__":
    main()
