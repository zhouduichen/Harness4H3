"""Pure scheduling primitives for the remote H3 pipeline.

The module deliberately has no SSH, CUDA, or subprocess dependency.  The
campaign uses these small value objects to decide whether a controller or
evaluator reservation leaves a safe elastic training allocation; the remote
scheduler remains the authority that observes and leases real GPUs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


class PipelineStage(str, Enum):
    IDLE = "idle"
    EVALUATING = "evaluating"
    TRAINING = "training"
    PREFETCHING = "prefetching"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class PipelineAllocation:
    """A proposed overlap layout before the remote lease is acquired."""

    training_gpus: Tuple[int, ...]
    free_gpus: Tuple[int, ...]
    mode: str
    plan_must_be_ready_before_training: bool

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "training_gpus": list(self.training_gpus),
            "free_gpus": list(self.free_gpus),
            "mode": self.mode,
            "plan_must_be_ready_before_training": self.plan_must_be_ready_before_training,
        }


@dataclass(frozen=True)
class EvaluationOverlap:
    """A disjoint evaluation/controller/training layout proposal."""

    evaluator_gpus: Tuple[int, ...]
    controller_gpus: Tuple[int, ...]
    training_gpus: Tuple[int, ...]
    mode: str
    disjoint: bool
    reason: str = ""

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "evaluator_gpus": list(self.evaluator_gpus),
            "controller_gpus": list(self.controller_gpus),
            "training_gpus": list(self.training_gpus),
            "mode": self.mode,
            "disjoint": self.disjoint,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PipelineState:
    """Durable control-plane cursor for one-at-a-time pipeline overlap."""

    stage: PipelineStage = PipelineStage.IDLE
    iteration: int = 0
    evaluation_model_id: Optional[str] = None
    training_model_id: Optional[str] = None
    overlap_with_model_id: Optional[str] = None
    updated_at: float = 0.0

    def to_dict(self) -> Mapping[str, Any]:
        value = asdict(self)
        value["stage"] = self.stage.value
        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PipelineState":
        if not isinstance(raw, Mapping):
            raise ValueError("pipeline state must be a mapping")
        stage_raw = str(raw.get("stage", PipelineStage.IDLE.value))
        try:
            stage = PipelineStage(stage_raw)
        except ValueError as exc:
            raise ValueError("invalid pipeline stage: %s" % stage_raw) from exc
        iteration = raw.get("iteration", 0)
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("pipeline iteration must be a non-negative integer")
        return cls(
            stage=stage,
            iteration=iteration,
            evaluation_model_id=(str(raw["evaluation_model_id"]) if raw.get("evaluation_model_id") else None),
            training_model_id=(str(raw["training_model_id"]) if raw.get("training_model_id") else None),
            overlap_with_model_id=(str(raw["overlap_with_model_id"]) if raw.get("overlap_with_model_id") else None),
            updated_at=float(raw.get("updated_at", 0.0) or 0.0),
        )

    def advance(
        self,
        stage: PipelineStage,
        *,
        evaluation_model_id: Optional[str] = None,
        training_model_id: Optional[str] = None,
        overlap_with_model_id: Optional[str] = None,
    ) -> "PipelineState":
        return PipelineState(
            stage=PipelineStage(stage),
            iteration=self.iteration + 1,
            evaluation_model_id=evaluation_model_id,
            training_model_id=training_model_id,
            overlap_with_model_id=overlap_with_model_id,
            updated_at=time.time(),
        )


def _validate_gpu_range(total_gpu_count: int, indices: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(total_gpu_count, bool) or not isinstance(total_gpu_count, int) or total_gpu_count <= 0:
        raise ValueError("total_gpu_count must be a positive integer")
    values = tuple(sorted(set(int(index) for index in indices)))
    if any(index < 0 or index >= total_gpu_count for index in values):
        raise ValueError("reserved GPU index is outside total_gpu_count")
    return values


def pack_overlap_resources(
    total_gpu_count: int,
    reserved: Sequence[int] = (),
    minimum_training_gpus: int = 2,
    maximum_training_gpus: int = 4,
) -> PipelineAllocation:
    """Return the largest safe training group after existing reservations.

    ``reserved`` represents GPUs already owned by an evaluator or another
    campaign-owned role.  This function never claims those cards and never
    makes a partial distributed allocation: callers must wait when the
    minimum cannot be met.  If no card remains after training, the next plan
    must be ready before the worker lease is taken because there is no card on
    which to run the Controller concurrently.
    """

    reserved_indices = _validate_gpu_range(total_gpu_count, reserved)
    if (
        isinstance(minimum_training_gpus, bool)
        or not isinstance(minimum_training_gpus, int)
        or isinstance(maximum_training_gpus, bool)
        or not isinstance(maximum_training_gpus, int)
        or minimum_training_gpus < 0
        or maximum_training_gpus < minimum_training_gpus
        or maximum_training_gpus > total_gpu_count
    ):
        raise ValueError("training GPU bounds are invalid")
    available = tuple(index for index in range(total_gpu_count) if index not in reserved_indices)
    if len(available) < minimum_training_gpus:
        return PipelineAllocation((), available, "waiting", False)
    count = min(len(available), maximum_training_gpus)
    training = available[-count:] if count else ()
    free = tuple(index for index in available if index not in training)
    mode = "evaluation_plus_training" if reserved_indices else "training"
    return PipelineAllocation(training, free, mode, not free)


def evaluation_gpu_count(task_count: int, configured_workers: int, free_gpu_count: int) -> int:
    """Bound evaluator fan-out by tasks, configuration, and live capacity."""

    values = (task_count, configured_workers, free_gpu_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("evaluation counts must be non-negative integers")
    return min(values)


def pack_evaluation_overlap(
    total_gpu_count: int,
    evaluator_gpus: Sequence[int] = (),
    controller_gpus: Sequence[int] = (),
    minimum_training_gpus: int = 2,
) -> EvaluationOverlap:
    """Return a full-card disjoint layout or an explicit waiting result."""

    _validate_gpu_range(total_gpu_count, ())
    if (
        isinstance(minimum_training_gpus, bool)
        or not isinstance(minimum_training_gpus, int)
        or minimum_training_gpus < 2
        or minimum_training_gpus > total_gpu_count
    ):
        raise ValueError("minimum_training_gpus must be in [2, total_gpu_count]")
    evaluator = _validate_gpu_range(total_gpu_count, evaluator_gpus)
    controller = _validate_gpu_range(total_gpu_count, controller_gpus)
    if set(evaluator).intersection(controller):
        raise ValueError("evaluator and controller GPU sets must be disjoint")
    reserved = set(evaluator).union(controller)
    training = tuple(index for index in range(total_gpu_count) if index not in reserved)
    if len(training) < minimum_training_gpus:
        return EvaluationOverlap(
            evaluator,
            controller,
            (),
            "waiting",
            True,
            "fewer_than_minimum_training_gpus",
        )
    return EvaluationOverlap(
        evaluator,
        controller,
        training,
        "evaluation_controller_training",
        not bool(set(evaluator).intersection(controller) or set(evaluator).intersection(training) or set(controller).intersection(training)),
    )


__all__ = [
    "EvaluationOverlap",
    "PipelineAllocation",
    "PipelineStage",
    "PipelineState",
    "evaluation_gpu_count",
    "pack_evaluation_overlap",
    "pack_overlap_resources",
]
