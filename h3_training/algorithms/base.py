"""Common role and method contracts shared by all training algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from h3_training.data.schema import PreparedBatch


@dataclass
class ModelRole:
    name: str
    model: nn.Module
    adapter: Any
    trainable: bool
    scheduler_state: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepOutput:
    losses: Mapping[str, Tensor]
    metrics: Mapping[str, float] = field(default_factory=dict)
    detached: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "total_loss" not in self.losses:
            raise ValueError("losses must contain total_loss")
        if set(self.losses).intersection(self.metrics):
            raise ValueError("metric keys must not collide with loss keys")
        for name, loss in self.losses.items():
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise ValueError(f"loss {name} must be a scalar tensor")


class TrainingMethod(nn.Module, ABC):
    algorithm_name = "base"

    def __init__(self) -> None:
        super().__init__()

    def prepare(self) -> None:
        """Validate and freeze roles before the first forward pass."""

    @abstractmethod
    def training_step(self, batch: PreparedBatch, iteration: int) -> StepOutput:
        raise NotImplementedError

    @abstractmethod
    def optimizers(self, iteration: int) -> Mapping[str, Optimizer]:
        raise NotImplementedError

    @abstractmethod
    def grad_clip_targets(self, iteration: int) -> Mapping[str, nn.Module]:
        raise NotImplementedError

    def schedulers(self) -> Mapping[str, Any]:
        return {}

    def optimizer_map(self) -> Mapping[str, Optimizer]:
        return self.optimizers(1)

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {"module": self.state_dict()}

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.load_state_dict(state["module"])

    def algorithm_state(self) -> Mapping[str, Any]:
        return {}

    def load_algorithm_state(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError("unexpected algorithm state")

    def on_optimizers_stepped(self, names: Sequence[str]) -> None:
        """Hook for EMA and algorithm counters."""
