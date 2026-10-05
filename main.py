import argparse

import torch

from pc_core.dataset import load_dataset
from pc_core.models import mlp_graph
from pc_core.regimes import BalancedRegime, ForwardInitRegime, PCConfig, PCRegime
from pc_core.trainer import Trainer

REGIMES: dict[str, type[PCRegime]] = {
    "zero": PCRegime,
    "forward": ForwardInitRegime,
    "balanced": BalancedRegime,
}


def main() -> None:
    """Train a one-hidden-layer predictive-coding graph on MNIST."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=sorted(REGIMES), default="forward")
    parser.add_argument("--init", choices=["forward", "leapfrog"], default="forward")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learn-precision", action="store_true")
    parser.add_argument("--min-log-var", type=float, default=-4.0)
    parser.add_argument("--image-weight", type=float, default=1.0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    train, test = load_dataset(batch_size=256)
    graph = mlp_graph(
        hidden_dims=args.hidden,
        init=args.init,
        learn_precision=args.learn_precision,
        min_log_var=args.min_log_var,
        image_weight=args.image_weight,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PCConfig(infer_steps=1, eval_steps=args.eval_steps)
    regime = REGIMES[args.regime](graph, config, device=device)
    trainer = Trainer(regime, train, test)
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
