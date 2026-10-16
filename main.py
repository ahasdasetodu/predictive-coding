from pc_core.experiment import parse_cli, train_mnist


def main() -> None:
    """Train a predictive-coding graph on MNIST from command-line flags."""
    train_mnist(parse_cli())


if __name__ == "__main__":
    main()
