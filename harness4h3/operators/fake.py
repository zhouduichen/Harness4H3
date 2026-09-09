from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from ..archive.model_candidate import ModelCandidate
from ..controller.schemas import CostEstimate, OperatorResult
from ..h3.state import ModelState
from ..target.profile import TargetProfile
from .base import ExecutionContext, OperatorRegistry, OperatorValidationError


class FakeOperatorBackend:
    """Deterministic model transformations for offline integration tests."""

    def __init__(self, failures: Optional[Mapping[str, List[str]]] = None):
        self.failures = {name: list(items) for name, items in (failures or {}).items()}

    def execute(self, name: str, parent: ModelCandidate, args: Mapping[str, Any], runtime: ExecutionContext) -> OperatorResult:
        queued = self.failures.get(name, [])
        if queued:
            failure = queued.pop(0)
            return OperatorResult("failed", None, CostEstimate(wall_time_s=0.1), failure_type=failure, message=failure)
        if name in {"inspect", "benchmark"}:
            return OperatorResult("success", None, CostEstimate(wall_time_s=0.01), metrics=dict(parent.state.measured_metrics))
        if name == "rollback":
            if runtime.model_store is None:
                return OperatorResult("failed", None, CostEstimate(), failure_type="rollback_target_missing", message="model store is required")
            source = runtime.model_store.get(str(args["target_model_id"]))
            state = source.state.derive(
                runtime.child_model_id,
                measured_metrics=copy.deepcopy(dict(source.state.measured_metrics)),
                provenance={"operator": "rollback", "source_model_id": source.id},
            )
            return OperatorResult("success", state, CostEstimate(wall_time_s=0.02))
        metrics = {key: float(value) for key, value in parent.state.measured_metrics.items()}
        if name == "quantize":
            metrics.update(
                quality_score=metrics["quality_score"] * 0.99,
                latency_s=metrics["latency_s"] * 0.80,
                peak_memory_gb=metrics["peak_memory_gb"] * 0.65,
                model_size_gb=metrics["model_size_gb"] * 0.55,
                energy_j=metrics["energy_j"] * 0.80,
            )
            metrics["throughput"] = 1.0 / metrics["latency_s"]
            state = parent.state.derive(
                runtime.child_model_id,
                dtype="int%d" % int(args.get("bits", 4)),
                quantization={"bits": int(args.get("bits", 4)), "scheme": "fake_weight_only"},
                measured_metrics=metrics,
                provenance={"operator": "quantize", "parent_model_id": parent.id},
            )
            return OperatorResult("success", state, CostEstimate(wall_time_s=0.2), metrics={"fake": True})
        if name == "step_distill":
            metrics.update(
                quality_score=metrics["quality_score"] * 0.96,
                latency_s=metrics["latency_s"] * 0.55,
                peak_memory_gb=metrics["peak_memory_gb"] * 0.70,
                energy_j=metrics["energy_j"] * 0.60,
            )
            metrics["throughput"] = 1.0 / metrics["latency_s"]
            target_steps = int(args.get("target_steps", 8))
            state = parent.state.derive(
                runtime.child_model_id,
                sampling_steps=target_steps,
                algorithm_state={"distillation": "fake_step_distill", "target_steps": target_steps},
                measured_metrics=metrics,
                provenance={"operator": "step_distill", "parent_model_id": parent.id},
            )
            return OperatorResult("success", state, CostEstimate(wall_time_s=0.3, gpu_hours=0.01), metrics={"fake": True})
        return OperatorResult("failed", None, CostEstimate(), failure_type="unsupported_operator", message=name)


@dataclass(frozen=True)
class FakeOperator:
    name: str
    description: str
    backend: FakeOperatorBackend
    allowed_args: Mapping[str, Tuple[type, ...]]

    def schema(self) -> Mapping[str, Any]:
        return {name: "/".join(kind.__name__ for kind in kinds) for name, kinds in self.allowed_args.items()}

    def validate(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> None:
        unknown = sorted(set(args) - set(self.allowed_args))
        if unknown:
            raise OperatorValidationError("unsupported argument(s) for %s: %s" % (self.name, ", ".join(unknown)))
        for name, value in args.items():
            if not isinstance(value, self.allowed_args[name]) or isinstance(value, bool):
                raise OperatorValidationError("%s.%s has invalid type" % (self.name, name))
        if self.name == "quantize":
            bits = int(args.get("bits", 4))
            if bits not in {4, 8}:
                raise OperatorValidationError("quantize.bits must be 4 or 8")
            if int(parent.quantization.get("bits", 16)) <= bits:
                raise OperatorValidationError("model is already quantized to %d bits or lower" % bits)
        if self.name == "step_distill":
            steps = int(args.get("target_steps", 8))
            if steps <= 0 or (parent.sampling_steps is not None and steps >= parent.sampling_steps):
                raise OperatorValidationError("target_steps must be positive and below current sampling steps")
        if self.name == "rollback" and not str(args.get("target_model_id", "")):
            raise OperatorValidationError("rollback requires target_model_id")

    def estimate_cost(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> CostEstimate:
        if self.name == "step_distill":
            return CostEstimate(wall_time_s=0.3, gpu_hours=0.01)
        return CostEstimate(wall_time_s=0.2 if self.name == "quantize" else 0.02)

    def execute(self, parent: ModelCandidate, args: Mapping[str, Any], runtime: ExecutionContext) -> OperatorResult:
        return self.backend.execute(self.name, parent, args, runtime)


def build_fake_registry(backend: Optional[FakeOperatorBackend] = None) -> OperatorRegistry:
    backend = backend or FakeOperatorBackend()
    registry = OperatorRegistry()
    registry.register(FakeOperator("inspect", "Inspect normalized model state", backend, {}))
    registry.register(FakeOperator("quantize", "Apply deterministic fake weight quantization", backend, {"bits": (int,)}))
    registry.register(FakeOperator("step_distill", "Apply deterministic fake sampling-step distillation", backend, {"target_steps": (int,)}))
    registry.register(FakeOperator("rollback", "Clone a prior immutable candidate as a new child", backend, {"target_model_id": (str,)}))
    return registry
