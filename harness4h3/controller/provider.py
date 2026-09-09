from __future__ import annotations

from typing import Any, Mapping, Protocol

from .context import ControllerContext
from .schemas import ExperimentPlan


class ControllerProvider(Protocol):
    provider_name: str
    model_name: str

    def plan(self, context: ControllerContext) -> Any:
        ...


class RuleBasedMockController:
    provider_name = "offline"
    model_name = "rule-based-mock"

    def plan(self, context: ControllerContext) -> ExperimentPlan:
        state = context.current_model_state
        target = context.target_profile
        metrics = state.measured_metrics
        quantized = int(state.quantization.get("bits", 16)) <= 4
        memory_blocked = target.max_peak_memory_gb is not None and float(metrics["peak_memory_gb"]) > target.max_peak_memory_gb
        size_blocked = target.max_model_size_gb is not None and float(metrics["model_size_gb"]) > target.max_model_size_gb
        latency_blocked = target.max_latency_s is not None and float(metrics["latency_s"]) > target.max_latency_s
        if (memory_blocked or size_blocked) and not quantized:
            operator = "quantize"
            args: Mapping[str, Any] = {"bits": 4}
            diagnosis = "memory_or_model_size"
            objective = "reduce model size, memory, and latency"
            hypothesis = "four-bit fake quantization reduces memory with bounded quality loss"
            effects = {"peak_memory_gb": "decrease", "model_size_gb": "decrease", "latency_s": "decrease"}
            required = {"wall_time_s": 0.2, "gpu_hours": 0.0}
        elif latency_blocked or memory_blocked:
            operator = "step_distill"
            args = {"target_steps": 8}
            diagnosis = "latency_or_residual_memory"
            objective = "reduce sampling latency and residual working memory"
            hypothesis = "fake step distillation reduces sampling work while preserving the quality floor"
            effects = {"latency_s": "decrease", "peak_memory_gb": "decrease"}
            required = {"wall_time_s": 0.3, "gpu_hours": 0.01}
        else:
            operator = "inspect"
            args = {}
            diagnosis = "no_blocking_constraint"
            objective = "confirm normalized model state"
            hypothesis = "inspection will confirm that no model modification is needed"
            effects = {"model_state": "unchanged"}
            required = {"wall_time_s": 0.02, "gpu_hours": 0.0}
        number = context.budget_state.used_iterations + 1
        acceptance = {"max_quality_drop": target.max_quality_drop if target.max_quality_drop is not None else 1.0}
        if target.min_quality_score is not None:
            acceptance["min_quality_score"] = target.min_quality_score
        return ExperimentPlan(
            experiment_id="exp_%04d" % number,
            parent_model_id=state.model_id,
            diagnosis=diagnosis,
            objective=objective,
            hypothesis=hypothesis,
            operator=operator,
            operator_args=args,
            expected_effects=effects,
            risks=["quality regression"],
            required_budget=required,
            acceptance=acceptance,
            stop_conditions={"critical_regression": True},
            rationale="select the first registered operator that addresses the current hard constraint",
        )
