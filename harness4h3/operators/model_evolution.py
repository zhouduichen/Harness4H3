from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..archive.model_candidate import ModelCandidate
from ..controller.schemas import CostEstimate, OperatorResult
from ..h3.state import ModelState
from ..target.profile import TargetProfile
from .base import ExecutionContext, OperatorRegistry, OperatorValidationError
from .external import ExternalScriptOperator
from ..executor.local import LocalProcessExecutor


MODEL_OPERATORS = (
    "create_student",
    "prune_blocks",
    "prune_heads",
    "prune_channels",
    "distill",
    "step_distill",
    "recovery_finetune",
    "dmd2",
    "quantize",
)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorValidationError("%s must be numeric" % name)
    return float(value)


def _metrics(parent: ModelState) -> Dict[str, float]:
    defaults = {
        "quality_score": 0.9,
        "latency_s": 60.0,
        "peak_memory_gb": 12.0,
        "model_size_gb": 7.0,
        "energy_j": 120.0,
    }
    return {key: float(parent.measured_metrics.get(key, value)) for key, value in defaults.items()}


def _provenance(parent: ModelState, operator: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **copy.deepcopy(dict(parent.provenance)),
        "operator": operator,
        "parent_model_id": parent.model_id,
        "operator_args": copy.deepcopy(dict(args)),
        "offline_simulation": True,
    }


@dataclass(frozen=True)
class ModelEvolutionBackend:
    """Deterministic model-state transformations for offline campaign tests.

    This backend does not read or write weights. A real campaign must replace it
    with an ``ExternalScriptOperator`` backed by a training/distillation process.
    """

    failures: Mapping[str, List[str]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "failures", {name: list(items) for name, items in (self.failures or {}).items()})

    def _failure(self, name: str) -> Optional[OperatorResult]:
        queued = self.failures.get(name, [])
        if queued:
            failure = queued.pop(0)
            return OperatorResult(
                "failed",
                None,
                CostEstimate(wall_time_s=0.1),
                failure_type=failure,
                message=failure,
            )
        return None

    def execute(
        self,
        name: str,
        parent: ModelCandidate,
        args: Mapping[str, Any],
        runtime: ExecutionContext,
    ) -> OperatorResult:
        failure = self._failure(name)
        if failure is not None:
            return failure

        metrics = _metrics(parent.state)
        state_changes: Dict[str, Any] = {}
        cost = CostEstimate(wall_time_s=0.2)
        if name == "create_student":
            ratio = float(args["width_ratio"])
            block_ratio = float(args.get("block_ratio", ratio))
            metrics["quality_score"] *= 1.0 - 0.035 * (1.0 - ratio)
            metrics["model_size_gb"] *= ratio * 0.98
            metrics["peak_memory_gb"] *= ratio * 0.96
            metrics["latency_s"] *= ratio ** 0.9
            metrics["energy_j"] *= ratio ** 0.9
            state_changes.update(
                parameter_count=max(1, int((parent.state.parameter_count or 1) * ratio)),
                trainable_parameter_count=max(1, int((parent.state.parameter_count or 1) * ratio)),
                num_blocks=max(1, int((parent.state.num_blocks or 1) * block_ratio)),
                hidden_size=max(1, int((parent.state.hidden_size or 1) * ratio ** 0.5)),
                num_attention_heads=max(1, int((parent.state.num_attention_heads or 1) * ratio)),
                ffn_width=max(1, int((parent.state.ffn_width or 1) * ratio ** 0.5)),
                algorithm_state={"student": True, "width_ratio": ratio, "block_ratio": block_ratio},
            )
            cost = CostEstimate(wall_time_s=1.0, gpu_hours=0.25)
        elif name in {"prune_blocks", "prune_heads", "prune_channels"}:
            ratio = float(args["ratio"])
            retained = 1.0 - ratio
            metrics["quality_score"] *= 1.0 - 0.06 * ratio
            metrics["model_size_gb"] *= 0.88 + 0.12 * retained
            metrics["peak_memory_gb"] *= 0.82 + 0.18 * retained
            metrics["latency_s"] *= 0.78 + 0.22 * retained
            metrics["energy_j"] *= 0.78 + 0.22 * retained
            state_changes["algorithm_state"] = {
                **copy.deepcopy(dict(parent.state.algorithm_state)),
                "structured_prune": name,
                "prune_ratio": ratio,
            }
            if name == "prune_blocks":
                state_changes["num_blocks"] = max(1, int((parent.state.num_blocks or 1) * retained))
            elif name == "prune_heads":
                state_changes["num_attention_heads"] = max(1, int((parent.state.num_attention_heads or 1) * retained))
            else:
                state_changes["hidden_size"] = max(1, int((parent.state.hidden_size or 1) * retained))
                state_changes["ffn_width"] = max(1, int((parent.state.ffn_width or 1) * retained))
            cost = CostEstimate(wall_time_s=0.6, gpu_hours=0.12)
        elif name == "distill":
            dataset_fraction = float(args["dataset_fraction"])
            training_steps = int(args["training_steps"])
            recovery = min(0.025, 0.01 + dataset_fraction * 0.04 + training_steps / 100000.0)
            metrics["quality_score"] = min(0.999, metrics["quality_score"] + recovery)
            metrics["latency_s"] *= 0.98
            metrics["energy_j"] *= 0.98
            state_changes["algorithm_state"] = {
                **copy.deepcopy(dict(parent.state.algorithm_state)),
                "distillation": "offline_short_distill",
                "dataset_fraction": dataset_fraction,
                "training_steps": training_steps,
            }
            cost = CostEstimate(wall_time_s=2.0, gpu_hours=0.75)
        elif name == "step_distill":
            target_steps = int(args["target_steps"])
            current_steps = parent.state.sampling_steps or 50
            reduction = max(0.1, min(0.8, target_steps / float(current_steps)))
            metrics["quality_score"] *= 1.0 - 0.04 * (1.0 - reduction)
            metrics["latency_s"] *= 0.55 + 0.45 * reduction
            metrics["peak_memory_gb"] *= 0.70 + 0.30 * reduction
            metrics["energy_j"] *= 0.60 + 0.40 * reduction
            state_changes.update(
                sampling_steps=target_steps,
                algorithm_state={
                    **copy.deepcopy(dict(parent.state.algorithm_state)),
                    "step_distillation": "offline_step_distill",
                    "target_steps": target_steps,
                },
            )
            cost = CostEstimate(wall_time_s=2.5, gpu_hours=1.0)
        elif name == "recovery_finetune":
            training_steps = int(args["training_steps"])
            gain = min(0.02, training_steps / 100000.0)
            metrics["quality_score"] = min(0.999, metrics["quality_score"] + gain)
            state_changes["algorithm_state"] = {
                **copy.deepcopy(dict(parent.state.algorithm_state)),
                "recovery_finetune": True,
                "training_steps": training_steps,
            }
            cost = CostEstimate(wall_time_s=1.5, gpu_hours=0.5)
        elif name == "dmd2":
            training_steps = int(args["training_steps"])
            metrics["quality_score"] = min(0.999, metrics["quality_score"] + min(0.02, training_steps / 100000.0))
            state_changes["algorithm_state"] = {
                **copy.deepcopy(dict(parent.state.algorithm_state)),
                "dmd2": "offline_experimental_reference",
                "training_steps": training_steps,
                "experimental": True,
            }
            cost = CostEstimate(wall_time_s=3.0, gpu_hours=1.5)
        elif name == "quantize":
            bits = int(args["bits"])
            factor = 0.55 if bits == 4 else 0.72
            metrics["quality_score"] *= 0.99 if bits == 4 else 0.995
            metrics["model_size_gb"] *= factor
            metrics["peak_memory_gb"] *= factor
            metrics["latency_s"] *= 0.80 if bits == 4 else 0.88
            metrics["energy_j"] *= 0.80 if bits == 4 else 0.88
            state_changes.update(
                dtype="int%d" % bits,
                quantization={"bits": bits, "scheme": "offline_weight_only"},
            )
            cost = CostEstimate(wall_time_s=0.8, gpu_hours=0.2)
        else:
            return OperatorResult("failed", None, CostEstimate(), failure_type="unsupported_operator", message=name)

        metrics["throughput"] = 1.0 / max(metrics["latency_s"], 1e-9)
        state_changes.update(
            measured_metrics=metrics,
            provenance=_provenance(parent.state, name, args),
            runtime_state={
                **copy.deepcopy(dict(parent.state.runtime_state)),
                "metrics_stale": False,
                "operator": name,
                "offline_simulation": True,
            },
        )
        state = parent.state.derive(runtime.child_model_id, **state_changes)
        return OperatorResult(
            "success",
            state,
            cost,
            metrics={"offline_simulation": True, "operator": name},
        )


@dataclass(frozen=True)
class ModelEvolutionOperator:
    name: str
    description: str
    backend: ModelEvolutionBackend
    allowed_args: Mapping[str, Tuple[type, ...]]
    cost: CostEstimate

    def schema(self) -> Mapping[str, Any]:
        return {key: "/".join(kind.__name__ for kind in kinds) for key, kinds in self.allowed_args.items()}

    def validate(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> None:
        if parent.architecture_name.lower().find("h3") < 0:
            raise OperatorValidationError("%s requires an H3 model" % self.name)
        if not isinstance(args, Mapping):
            raise OperatorValidationError("operator arguments must be a mapping")
        unknown = sorted(set(args) - set(self.allowed_args))
        if unknown:
            raise OperatorValidationError("unsupported argument(s) for %s: %s" % (self.name, ", ".join(unknown)))
        missing = sorted(set(self.allowed_args) - set(args))
        if missing:
            raise OperatorValidationError("%s requires argument(s): %s" % (self.name, ", ".join(missing)))
        for key, value in args.items():
            allowed = self.allowed_args[key]
            if isinstance(value, bool) or not isinstance(value, allowed):
                raise OperatorValidationError("%s.%s has invalid type" % (self.name, key))
        if self.name == "create_student":
            ratio = _number(args["width_ratio"], "create_student.width_ratio")
            block_ratio = _number(args["block_ratio"], "create_student.block_ratio")
            if not 0.1 <= ratio < 1.0 or not 0.1 <= block_ratio <= 1.0:
                raise OperatorValidationError("student ratios must be in [0.1, 1.0]")
        elif self.name.startswith("prune_"):
            ratio = _number(args["ratio"], "%s.ratio" % self.name)
            if not 0.01 <= ratio <= 0.8:
                raise OperatorValidationError("%s.ratio must be in [0.01, 0.8]" % self.name)
        elif self.name == "distill":
            fraction = _number(args["dataset_fraction"], "distill.dataset_fraction")
            if not 0.01 <= fraction <= 1.0 or int(args["training_steps"]) <= 0:
                raise OperatorValidationError("distill arguments are out of range")
        elif self.name == "step_distill":
            steps = int(args["target_steps"])
            if steps <= 0 or (parent.sampling_steps is not None and steps >= parent.sampling_steps):
                raise OperatorValidationError("step_distill.target_steps must be positive and below current steps")
        elif self.name == "recovery_finetune" and int(args["training_steps"]) <= 0:
            raise OperatorValidationError("recovery_finetune.training_steps must be positive")
        elif self.name == "dmd2":
            if int(args["training_steps"]) <= 0:
                raise OperatorValidationError("dmd2.training_steps must be positive")
            if "generator_update_interval" in args and int(args["generator_update_interval"]) <= 0:
                raise OperatorValidationError("dmd2.generator_update_interval must be positive")
        elif self.name == "quantize":
            bits = int(args["bits"])
            if bits not in {4, 8}:
                raise OperatorValidationError("quantize.bits must be 4 or 8")
            if int(parent.quantization.get("bits", 16)) <= bits:
                raise OperatorValidationError("model is already quantized to %d bits or lower" % bits)

    def estimate_cost(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> CostEstimate:
        self.validate(parent, args, target)
        return self.cost

    def execute(self, parent: ModelCandidate, args: Mapping[str, Any], runtime: ExecutionContext) -> OperatorResult:
        try:
            return self.backend.execute(self.name, parent, args, runtime)
        except (KeyError, TypeError, ValueError, OperatorValidationError) as exc:
            return OperatorResult("failed", None, CostEstimate(), failure_type="operator_invalid", message=str(exc))


def build_model_evolution_registry(backend: Optional[ModelEvolutionBackend] = None) -> OperatorRegistry:
    backend = backend or ModelEvolutionBackend()
    registry = OperatorRegistry()
    definitions = {
        "create_student": (
            "Create a smaller H3 student architecture before training/distillation",
            {"width_ratio": (float,), "block_ratio": (float,)},
            CostEstimate(wall_time_s=1.0, gpu_hours=0.25),
        ),
        "prune_blocks": ("Structured-prune H3 transformer blocks", {"ratio": (float,)}, CostEstimate(0.6, 0.12)),
        "prune_heads": ("Structured-prune H3 attention heads", {"ratio": (float,)}, CostEstimate(0.6, 0.12)),
        "prune_channels": ("Structured-prune H3 hidden/FFN channels", {"ratio": (float,)}, CostEstimate(0.6, 0.12)),
        "distill": (
            "Distill teacher behavior into the current H3 student",
            {"dataset_fraction": (float,), "training_steps": (int,)},
            CostEstimate(2.0, 0.75),
        ),
        "step_distill": ("Distill the sampling trajectory to fewer steps", {"target_steps": (int,)}, CostEstimate(2.5, 1.0)),
        "recovery_finetune": (
            "Recover quality after structural change with short fine-tuning",
            {"training_steps": (int,)},
            CostEstimate(1.5, 0.5),
        ),
        "dmd2": (
            "Experimental DMD2 reference update with alternating critic and student roles",
            {"training_steps": (int,)},
            CostEstimate(3.0, 1.5),
        ),
        "quantize": ("Quantize model weights to a lower precision", {"bits": (int,)}, CostEstimate(0.8, 0.2)),
    }
    for name, (description, args, cost) in definitions.items():
        registry.register(ModelEvolutionOperator(name, description, backend, args, cost))
    return registry


def build_external_model_evolution_registry(
    command: Sequence[str],
    executor: Optional[LocalProcessExecutor] = None,
    timeout_s: float = 3600.0,
) -> OperatorRegistry:
    """Register real training workers using the existing safe external contract.

    Each worker receives ``--request`` and ``--result`` in an isolated experiment
    directory and must return a new validated checkpoint/state. The command is an
    argv sequence; it is never executed through a shell.
    """
    if not command:
        raise OperatorValidationError("external model-evolution registry requires a command")
    executor = executor or LocalProcessExecutor(timeout_s=timeout_s)
    definitions = {
        "create_student": (
            "Create and train a smaller H3 student",
            {"width_ratio": (float,), "block_ratio": (float,)},
            CostEstimate(wall_time_s=3600.0, gpu_hours=2.0),
        ),
        "prune_blocks": ("Prune H3 blocks and recover weights", {"ratio": (float,)}, CostEstimate(1800.0, 1.0)),
        "prune_heads": ("Prune H3 attention heads and recover weights", {"ratio": (float,)}, CostEstimate(1800.0, 1.0)),
        "prune_channels": ("Prune H3 channels and recover weights", {"ratio": (float,)}, CostEstimate(1800.0, 1.0)),
        "distill": (
            "Distill H3 teacher outputs into the selected student",
            {"dataset_fraction": (float,), "training_steps": (int,)},
            CostEstimate(7200.0, 4.0),
        ),
        "step_distill": ("Distill an H3 sampling trajectory", {"target_steps": (int,)}, CostEstimate(7200.0, 4.0)),
        "recovery_finetune": ("Recovery fine-tune after a structural change", {"training_steps": (int,)}, CostEstimate(3600.0, 2.0)),
        "dmd2": (
            "Experimental DMD2 reference training on TinyH3-compatible workers",
            {"training_steps": (int,)},
            CostEstimate(10800.0, 6.0),
        ),
        "quantize": ("Quantize H3 weights with a real worker", {"bits": (int,)}, CostEstimate(1800.0, 1.0)),
    }
    registry = OperatorRegistry()
    for name, (description, allowed_args, cost) in definitions.items():
        registry.register(ExternalScriptOperator(name, description, command, allowed_args, executor, cost, timeout_s))
    return registry


__all__ = [
    "MODEL_OPERATORS",
    "ModelEvolutionBackend",
    "ModelEvolutionOperator",
    "build_model_evolution_registry",
    "build_external_model_evolution_registry",
]
