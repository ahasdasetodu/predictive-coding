from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from pc_core.balance import BalanceConfig, EdgeBalancer
from pc_core.graph import Edge, PCGraph, States

Batch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class TrainResult:
    """Report one training batch without constraining regime-specific metrics."""

    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Hold additive classification counts for one evaluation batch."""

    correct: int
    total: int


class TrainingRegime(Protocol):
    """Define the algorithm-specific operations used by the generic trainer."""

    model: nn.Module

    def start_training(self, epochs: int) -> None:
        """Initialize run-level state such as learning-rate schedules."""
        ...

    def train_batch(self, batch: Batch) -> TrainResult:
        """Train on one batch and return its reported measurements."""
        ...

    def evaluate_batch(self, batch: Batch) -> EvaluationResult:
        """Evaluate one batch and return additive result counts."""
        ...

    def end_epoch(self) -> None:
        """Advance algorithm-specific state after an epoch."""
        ...


@dataclass(frozen=True)
class PCConfig:
    """Step counts and learning rates shared by predictive-coding regimes."""

    infer_steps: int = 20
    eval_steps: int = 30
    infer_lr: float = 1e-1
    eval_lr: float = 1e-1
    weight_lr: float = 1e-3
    min_lr: float = 0.0


class PCRegime:
    """Settle free states on the graph energy, then update weights on the settled states.

    Subclasses change behaviour through `init_states` and `extra_loss`.
    """

    input_node = "image"
    label_node = "label"

    def __init__(
        self,
        model: PCGraph,
        config: PCConfig = PCConfig(),
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.infer_lr = config.infer_lr
        self.eval_lr = config.eval_lr
        self.weight_lr = config.weight_lr
        self.weight_opt = optim.Adam(self.model.parameters(), lr=config.weight_lr)
        self.weight_sched: CosineAnnealingLR | None = None

    def init_states(self, clamped: States) -> States:
        """Start every free node at zero."""
        batch_size = next(iter(clamped.values())).shape[0]
        states = {
            name: torch.zeros(batch_size, width, device=self.device)
            for name, width in self.model.nodes.items()
        }
        return {**states, **clamped}

    def extra_loss(self, settled: States) -> Tensor:
        """Add a loss to the weight step on top of the energy; none by default."""
        return torch.zeros((), device=self.device)

    def current_energy_edges(self) -> Sequence[Edge] | None:
        """Energy edges for this batch; `None` means every energy edge."""
        return None

    def observe_init(self, states: States) -> None:
        """Hook: inspect the states settling starts from; does nothing by default."""

    def observe_settled(self, settled: States) -> None:
        """Hook: inspect the settled states before the weight step; does nothing by default."""

    def extra_metrics(self) -> dict[str, float]:
        """Add regime-specific values to the training log; none by default."""
        return {}

    def start_training(self, epochs: int) -> None:
        """Start a fresh weight learning-rate schedule for a training run."""
        self.weight_sched = self._cosine(self.weight_opt, self.config.weight_lr, epochs)
        self.weight_lr = self.weight_sched.get_last_lr()[0]

    def train_batch(self, batch: Batch) -> TrainResult:
        """Settle with image and label clamped, then take one weight step."""
        x_data, labels = batch
        clamped = {
            self.input_node: self._flat(x_data),
            self.label_node: self._one_hot(labels),
        }
        return self.train_clamped(clamped)

    def train_clamped(self, clamped: States) -> TrainResult:
        """Settle the free nodes against `clamped`, then take one weight step."""
        batch_size = next(iter(clamped.values())).shape[0]
        states = self._free_states(clamped)
        self.observe_init(states)
        edges = self.current_energy_edges()
        with torch.no_grad():
            init_energy = self.model.energy(states, edges).item()
        self.infer_lr = self.settle(
            states, self._free(clamped), self.config.infer_steps, self.config.infer_lr
        )

        settled = {name: state.detach() for name, state in states.items()}
        self.observe_settled(settled)
        self.weight_opt.zero_grad()
        edge_energies = self.model.edge_energies(settled, edges)
        energy = torch.stack(list(edge_energies.values())).sum()
        extra = self.extra_loss(settled)
        loss = (energy + extra) / batch_size
        loss.backward()
        self.weight_opt.step()
        return TrainResult(
            loss=loss.item(),
            metrics={
                "init_energy": init_energy / batch_size,
                "final_energy": energy.item() / batch_size,
                **{f"E[{name}]": term.item() / batch_size for name, term in edge_energies.items()},
                **{f"logvar[{name}]": value for name, value in self.model.log_variances().items()},
                "extra_loss": extra.item() / batch_size,
                "infer_lr": self.infer_lr,
                "weight_lr": self.weight_lr,
                **self.extra_metrics(),
            },
        )

    def evaluate_batch(self, batch: Batch) -> EvaluationResult:
        """Settle with only the image clamped and count label argmax hits."""
        x_data, labels = batch
        labels = labels.to(self.device)
        states = self.infer({self.input_node: self._flat(x_data)})
        predictions = states[self.label_node].argmax(dim=1)
        return EvaluationResult(
            correct=int((predictions == labels).sum().item()),
            total=labels.size(0),
        )

    def infer(self, clamped: States, steps: int | None = None) -> States:
        """Settle the free nodes against `clamped` for `steps` (default `eval_steps`) and return every state detached."""
        states = self._free_states(clamped)
        count = self.config.eval_steps if steps is None else steps
        self.eval_lr = self.settle(states, self._free(clamped), count, self.config.eval_lr)
        return {name: state.detach() for name, state in states.items()}

    def end_epoch(self) -> None:
        """Advance the weight schedule after one completed epoch."""
        if self.weight_sched is None:
            raise RuntimeError("start_training must be called before end_epoch")
        self.weight_sched.step()
        self.weight_lr = self.weight_sched.get_last_lr()[0]

    def settle(self, states: States, free: list[str], steps: int, lr: float) -> float:
        """Relax the free states on the energy and return the final step's learning rate."""
        state_opt = optim.SGD([states[name] for name in free], lr=lr)
        state_sched = CosineAnnealingLR(state_opt, T_max=max(steps, 1), eta_min=self.config.min_lr)
        used_lr = lr
        for _ in range(steps):
            used_lr = state_sched.get_last_lr()[0]
            state_opt.zero_grad()
            self.model.energy(states, self.current_energy_edges()).backward()
            state_opt.step()
            state_sched.step()
        return used_lr

    def _free_states(self, clamped: States) -> States:
        """Initialise all states and turn the unclamped ones into optimisable leaves."""
        states = self.init_states(clamped)
        for name in self._free(clamped):
            states[name] = states[name].detach().clone().requires_grad_(True)
        return states

    def _free(self, clamped: States) -> list[str]:
        """Names of the nodes settling may move."""
        return [name for name in self.model.nodes if name not in clamped]

    def _flat(self, x_data: Tensor) -> Tensor:
        """Move a batch of inputs to the device as flat vectors."""
        return x_data.to(self.device).reshape(x_data.shape[0], -1)

    def _one_hot(self, labels: Tensor) -> Tensor:
        """Encode class labels as one-hot rows on the device."""
        width = self.model.nodes[self.label_node]
        return F.one_hot(labels.to(self.device), num_classes=width).float()

    def _cosine(self, optimizer: optim.Optimizer, base_lr: float, t_max: int) -> CosineAnnealingLR:
        """Create a fresh cosine schedule after clearing stale initial state."""
        optimizer.param_groups[0]["lr"] = base_lr
        optimizer.param_groups[0].pop("initial_lr", None)
        return CosineAnnealingLR(optimizer, T_max=max(t_max, 1), eta_min=self.config.min_lr)


class ForwardInitRegime(PCRegime):
    """Bottom-up init, with that pass trained to classify the image by cross-entropy."""

    def init_states(self, clamped: States) -> States:
        """Fill free nodes from the init edges; nodes they do not reach start at zero."""
        with torch.no_grad():
            proposed = self.model.propagate(clamped)
        return {**super().init_states(clamped), **proposed}

    def extra_loss(self, settled: States) -> Tensor:
        """Cross-entropy of the label predicted from the clamped image alone."""
        # Settling clamps the label, so the edge that writes it does not run there.
        # This pass starts from the image and uses the clamped label only as the target.
        return self.init_label_loss(settled[self.input_node], settled[self.label_node].argmax(dim=1))

    def init_label_loss(self, images: Tensor, labels: Tensor) -> Tensor:
        """Sum of cross-entropy for the init pass that predicts `labels` from `images` alone."""
        proposed = self.model.propagate({self.input_node: images})
        if self.label_node not in proposed:
            raise ValueError("init edges do not reach the label from the image")
        return F.cross_entropy(proposed[self.label_node], labels, reduction="sum")


class BalancedRegime(ForwardInitRegime):
    """Forward init with energy-edge weights calibrated to give every edge equal say."""

    def __init__(
        self,
        model: PCGraph,
        config: PCConfig = PCConfig(),
        device: str | torch.device = "cpu",
        balance: BalanceConfig = BalanceConfig(),
    ) -> None:
        super().__init__(model, config, device=device)
        self.balancer = EdgeBalancer(self.model, balance)

    def observe_init(self, states: States) -> None:
        """Let the balancer measure the imbalance settling is about to act on."""
        self.balancer.observe(states)

    def extra_metrics(self) -> dict[str, float]:
        """Log the calibrated weight of every energy edge."""
        return self.balancer.report()
