import argparse
from collections.abc import Sequence
from pathlib import Path

from pc_core.experiment.runner import run_spec
from pc_core.experiment.spec import load_spec


def main(argv: Sequence[str] | None = None) -> None:
    """Run the experiment spec given on the command line."""
    parser = argparse.ArgumentParser(description="Run one experiment spec and write a result directory.")
    parser.add_argument("spec", type=Path)
    args = parser.parse_args(None if argv is None else list(argv))
    print(run_spec(load_spec(args.spec)))


if __name__ == "__main__":
    main()
