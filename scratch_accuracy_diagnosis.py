"""Locate the MNIST accuracy ceiling of the generative predictive-coding chain.

Compares, on the same data and width, a linear classifier, the init network trained
end to end, and the chain as it is actually trained. Prints where the class signal
stops: the loss, the clamped label, or the one settling step the encoder copies.

    uv run python scratch_accuracy_diagnosis.py
"""

import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pc_core.dataset import load_dataset
from pc_core.graph import PCGraph
from pc_core.models import graph_from_edges, mlp_graph
from pc_core.regimes import ForwardInitRegime, PCConfig, PCRegime, TrainResult

HIDDEN = 128
EPOCHS = 2
SEED = 0
BATCH = 256


def main() -> None:
    """Run the structural check, then each training rule for the same budget."""
    device = torch.device("cpu")
    train_loader, test_loader = load_dataset(batch_size=BATCH)
    print(f"device {device}  train batches {len(train_loader)}  test batches {len(test_loader)}")
    structural_credit(device)
    run_mlp(train_loader, test_loader, device, "linear-ce", hidden=None, mse=False)
    run_mlp(train_loader, test_loader, device, "init-shaped-ce", hidden=HIDDEN, mse=False)
    run_mlp(train_loader, test_loader, device, "init-shaped-mse", hidden=HIDDEN, mse=True)
    run_pc(
        train_loader,
        test_loader,
        device,
        "generative-pc",
        mlp_graph(HIDDEN, init="forward"),
        PCConfig(infer_steps=1, eval_steps=1),
        ForwardInitRegime,
    )
    run_pc(
        train_loader,
        test_loader,
        device,
        "discriminative-pc-1",
        discriminative_graph(),
        PCConfig(infer_steps=1, eval_steps=1),
        ForwardInitRegime,
    )
    run_pc(
        train_loader,
        test_loader,
        device,
        "discriminative-pc-10",
        discriminative_graph(),
        PCConfig(infer_steps=10, eval_steps=1),
        ForwardInitRegime,
    )
    run_pc(
        train_loader,
        test_loader,
        device,
        "init-cross-entropy",
        mlp_graph(HIDDEN, init="forward"),
        PCConfig(infer_steps=1, eval_steps=1),
        InitClassifierRegime,
    )


def structural_credit(device: torch.device) -> None:
    """Show that the label loss never reaches the encoder, and what one settle step copies."""
    torch.manual_seed(SEED)
    graph = mlp_graph(HIDDEN, init="forward").to(device)
    regime = ForwardInitRegime(graph, PCConfig(infer_steps=1, eval_steps=1), device=device)
    images, labels = next(iter(load_dataset(batch_size=32)[0]))
    images = images.to(device)
    labels = labels.to(device)
    clamped = {"image": images, "label": F.one_hot(labels, 10).float()}
    states = regime._free_states(clamped)
    regime.settle(states, regime._free(clamped), 1, regime.config.infer_lr)
    settled = {name: state.detach() for name, state in states.items()}
    encoder = next(edge for edge in graph.learned_init_edges() if edge.dst == "h1")
    label_edge = next(edge for edge in graph.learned_init_edges() if edge.dst == "label")
    graph.zero_grad()
    label_edge.energy(settled).backward()
    encoder_grad = encoder.module[0].weight.grad
    label_reaches = encoder_grad is not None and float(encoder_grad.abs().sum()) > 0.0
    graph.zero_grad()
    encoder.energy(settled).backward()
    copy_grad = float(encoder.module[0].weight.grad.norm())
    split = edge_gradient_norms(graph, images, labels)
    print(
        f"structural  label loss reaches encoder: {label_reaches}  "
        f"encoder-copy grad norm: {copy_grad:.4f}"
    )
    print(
        "structural  settling gradient norms at h1: "
        + "  ".join(f"{name} {norm:.4f}" for name, norm in split.items())
    )


def run_mlp(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    name: str,
    hidden: int | None,
    mse: bool,
) -> None:
    """Train a feedforward classifier and print its test accuracy."""
    torch.manual_seed(SEED)
    model = classifier(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            logits = model(images)
            loss = mse_one_hot(logits, labels) if mse else F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
        accuracy = mlp_accuracy(model, test_loader, device)
        print(f"{name}  epoch {epoch}  test {accuracy:.2f}%  ({time.perf_counter() - started:.1f}s)")


def run_pc(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    name: str,
    graph: PCGraph,
    config: PCConfig,
    regime_cls: type[PCRegime],
) -> None:
    """Train one predictive-coding regime and report init, settled, and probe accuracies."""
    torch.manual_seed(SEED)
    regime = regime_cls(graph, config, device=device)
    hidden = "h1" if "h1" in graph.nodes else "h"
    regime.start_training(EPOCHS)
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        regime.model.train()
        for batch in train_loader:
            regime.train_batch(batch)
        init_acc, settled_acc = readout_accuracies(regime, test_loader)
        regime.end_epoch()
        print(
            f"{name}  epoch {epoch}  init {init_acc:.2f}%  settled {settled_acc:.2f}%  "
            f"({time.perf_counter() - started:.1f}s)"
        )
    report_representation(name, regime, train_loader, test_loader, hidden)


def discriminative_graph() -> PCGraph:
    """Image to h to a clamped label. The label is a target, never a parent of h."""
    nodes = {"image": 28 * 28, "h": HIDDEN, "label": 10}
    return graph_from_edges(nodes, ["h->label"], ["image->h"], readout_edges=["h->label"])


class InitClassifierRegime(ForwardInitRegime):
    """Keep the generative energy, but train the init net with cross-entropy from the image alone."""

    def train_batch(self, batch: tuple[Tensor, Tensor]) -> TrainResult:
        """Remember the image and class, then take the ordinary predictive-coding step."""
        images, labels = batch
        self._class_images = self._flat(images)
        self._class_labels = labels.to(self.device)
        return super().train_batch(batch)

    def extra_loss(self, settled: dict[str, Tensor]) -> Tensor:
        """Cross-entropy of the init forward. The settled-state copy loss is not used."""
        del settled
        return F.cross_entropy(self._init_logits(self._class_images), self._class_labels, reduction="sum")

    def _init_logits(self, images: Tensor) -> Tensor:
        """Run the learned init edges from the image, with gradients."""
        states: dict[str, Tensor] = {self.input_node: images}
        for edge in self.model.learned_init_edges():
            states[edge.dst] = edge.predict(states)
        return states[self.label_node]


def classifier(hidden: int | None) -> nn.Module:
    """The init network's shape: one tanh layer, or a linear map when `hidden` is None."""
    if hidden is None:
        return nn.Linear(28 * 28, 10)
    return nn.Sequential(nn.Linear(28 * 28, hidden), nn.Tanh(), nn.Linear(hidden, 10))


def mse_one_hot(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean over the batch of half the squared error against a one-hot label."""
    target = F.one_hot(labels, num_classes=logits.shape[1]).float()
    return 0.5 * (logits - target).pow(2).sum(dim=1).mean()


@torch.no_grad()
def mlp_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Percent correct of a feedforward classifier."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        pred = model(images.to(device)).argmax(dim=1)
        labels = labels.to(device)
        correct += int((pred == labels).sum())
        total += labels.numel()
    return 100.0 * correct / total


def readout_accuracies(regime: PCRegime, loader: DataLoader) -> tuple[float, float]:
    """Percent correct of the init label and of the label after evaluation settling."""
    was_training = regime.model.training
    regime.model.eval()
    init_correct = 0
    settled_correct = 0
    total = 0
    for images, labels in loader:
        labels = labels.to(regime.device)
        clamped = {regime.input_node: regime._flat(images)}
        proposed = regime.model.propagate(clamped)
        init_pred = proposed[regime.label_node].argmax(dim=1)
        settled_pred = regime.infer(clamped)[regime.label_node].argmax(dim=1)
        init_correct += int((init_pred == labels).sum())
        settled_correct += int((settled_pred == labels).sum())
        total += labels.numel()
    regime.model.train(was_training)
    return 100.0 * init_correct / total, 100.0 * settled_correct / total


def report_representation(
    name: str,
    regime: PCRegime,
    train_loader: DataLoader,
    test_loader: DataLoader,
    hidden: str,
) -> None:
    """Probe the hidden state the encoder copies, with and without the label clamped."""
    train_h, train_y = collect_hidden(regime, train_loader, hidden, clamp_label=False, settle=True)
    test_h, test_y = collect_hidden(regime, test_loader, hidden, clamp_label=False, settle=True)
    teacher_train, teacher_y = collect_hidden(regime, train_loader, hidden, clamp_label=True, settle=True)
    teacher_test, _ = collect_hidden(regime, test_loader, hidden, clamp_label=True, settle=True)
    init_test, _ = collect_hidden(regime, test_loader, hidden, clamp_label=False, settle=False)
    image_probe = probe_accuracy(train_h, train_y, test_h, test_y)
    teacher_probe = probe_accuracy(teacher_train, teacher_y, teacher_test, test_y)
    copied = (teacher_test - init_test).norm(dim=1).mean().item()
    scale = init_test.norm(dim=1).mean().item()
    print(
        f"{name}  probe image-only h {image_probe:.2f}%  "
        f"probe label-clamped h {teacher_probe:.2f}%  "
        f"mean ||h_teacher - h_init|| {copied:.3f}  mean ||h_init|| {scale:.3f}"
    )
    images, labels = next(iter(test_loader))
    split = edge_gradient_norms(regime.model, regime._flat(images), labels.to(regime.device))
    if split:
        print(
            f"{name}  settling gradient norms: "
            + "  ".join(f"{edge} {norm:.4f}" for edge, norm in split.items())
        )


def collect_hidden(
    regime: PCRegime,
    loader: DataLoader,
    hidden: str,
    clamp_label: bool,
    settle: bool,
) -> tuple[Tensor, Tensor]:
    """Hidden state at init, or after the training settle budget.

    With the label clamped and settling on, this is the state the encoder is trained to copy.
    """
    was_training = regime.model.training
    regime.model.eval()
    states: list[Tensor] = []
    targets: list[Tensor] = []
    steps = regime.config.infer_steps
    for images, labels in loader:
        labels = labels.to(regime.device)
        clamped = {regime.input_node: regime._flat(images)}
        if clamp_label:
            clamped[regime.label_node] = regime._one_hot(labels)
        if settle:
            hidden_state = regime.infer(clamped, steps=steps)[hidden]
        else:
            hidden_state = regime.model.propagate(clamped)[hidden]
        states.append(hidden_state.detach().cpu())
        targets.append(labels.cpu())
    regime.model.train(was_training)
    return torch.cat(states), torch.cat(targets)


def probe_accuracy(train_x: Tensor, train_y: Tensor, test_x: Tensor, test_y: Tensor) -> float:
    """Test accuracy of a linear cross-entropy head fit on frozen features."""
    torch.manual_seed(SEED)
    head = nn.Linear(train_x.shape[1], 10)
    # Full-batch Adam at 1e-2 stalls on these feature scales and under-reports separability.
    opt = torch.optim.Adam(head.parameters(), lr=1e-1)
    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(head(train_x), train_y).backward()
        opt.step()
    with torch.no_grad():
        correct = int((head(test_x).argmax(dim=1) == test_y).sum())
    return 100.0 * correct / test_y.numel()


def edge_gradient_norms(graph: PCGraph, images: Tensor, labels: Tensor) -> dict[str, float]:
    """Mean per-sample gradient norm that each energy edge exerts on the first hidden state."""
    hidden = "h1" if "h1" in graph.nodes else "h"
    if hidden not in graph.nodes:
        return {}
    regime = ForwardInitRegime(graph, PCConfig())
    clamped = {"image": images, "label": F.one_hot(labels, 10).float()}
    states = regime.init_states(clamped)
    state = states[hidden].detach().requires_grad_(True)
    states = {**states, hidden: state}
    norms: dict[str, float] = {}
    for edge in graph.energy_edges:
        if hidden not in (*edge.sources, edge.dst):
            continue
        grad, = torch.autograd.grad(edge.energy(states), state, retain_graph=True)
        norms[edge.name] = grad.norm(dim=1).mean().item()
    return norms


if __name__ == "__main__":
    main()
