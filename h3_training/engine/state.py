"""Stable engine state and failure taxonomy."""

from dataclasses import dataclass, field
from typing import Dict, Mapping


class TrainingFailure(RuntimeError):
    """A fail-closed training error with a stable machine-readable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass
class LoopState:
    global_step: int = 0
    microbatches_consumed: int = 0
    accumulation_position: int = 0
    optimizer_steps: Dict[str, int] = field(default_factory=dict)
    sampler_position: int = 0


@dataclass(frozen=True)
class TrainingRunResult:
    loop_state: LoopState
    initial_loss: float
    final_loss: float
    max_gradient_norm: float
    optimizer_steps: Mapping[str, int]
    wall_time_s: float
    peak_memory_bytes: int
