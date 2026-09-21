"""Evidence gates for deciding whether a remote H3 child is promotable."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .reward import RewardResult, RewardWeights, compute_reward


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _target(target: Any, name: str, default: Any = None) -> Any:
    direct = _get(target, name, None)
    if direct is not None:
        return direct
    aliases = {
        "max_quality_drop": ("quality", "max_quality_drop"),
        "max_model_size_gb": ("constraints", "max_model_size_gb"),
        "max_peak_memory_gb": ("constraints", "max_peak_memory_gb"),
        "max_latency_s": ("constraints", "max_latency_s"),
        "max_energy_j": ("constraints", "max_energy_j"),
    }
    path = aliases.get(name)
    value = target
    if path and isinstance(value, Mapping):
        value = value.get(path[0], {})
        if isinstance(value, Mapping):
            return value.get(path[1], default)
    return default


@dataclass(frozen=True)
class AcceptanceInput:
    training_metrics: Mapping[str, Any]
    benchmark_summary: Any
    parent_summary: Any
    target: Any
    efficiency_thresholds: Mapping[str, float]
    reward_weights: RewardWeights = field(default_factory=lambda: RewardWeights(1.0, 0.2, 0.2, 0.2))
    research_grade: bool = False
    on_pareto_front: bool = False


@dataclass(frozen=True)
class DecisionResult:
    status: str
    accepted: bool
    violations: Tuple[str, ...]
    reward: Optional[float]
    pareto_eligible: bool
    reward_terms: Mapping[str, float] = field(default_factory=dict)
    missing_metrics: Tuple[str, ...] = ()
    continuation_status: str = "reject"
    advance: bool = False

    def with_continuation(self, status: str, advance: bool) -> "DecisionResult":
        return replace(self, continuation_status=str(status), advance=bool(advance))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _training_violations(metrics: Mapping[str, Any]) -> List[str]:
    operator = str(metrics.get("operator", "")).strip()
    if operator == "quantize":
        violations: List[str] = []
        before = metrics.get("parent_sha256_before") or metrics.get("parent_sha256")
        after = metrics.get("parent_sha256_after") or metrics.get("parent_sha256")
        child = metrics.get("child_sha256")
        if not before or not after or before != after:
            violations.append("quantize_parent_hash_changed")
        if not child or child == before:
            violations.append("quantize_child_hash_unchanged")
        if metrics.get("child_copy_verified") is not True:
            violations.append("quantize_child_copy_unverified")
        if metrics.get("source_is_quantized") is not True:
            violations.append("quantize_source_not_verified")
        if metrics.get("offline_simulation") is True:
            violations.append("quantize_offline_simulation")
        return violations
    if metrics.get("structural_change") is True or metrics.get("operator") == "prune_blocks":
        violations: List[str] = []
        before = metrics.get("parent_sha256_before") or metrics.get("parent_sha256")
        after = metrics.get("parent_sha256_after") or metrics.get("parent_sha256")
        child = metrics.get("child_sha256")
        if not before or not after or before != after:
            violations.append("training_parent_hash_changed")
        if not child or child == before:
            violations.append("training_child_hash_unchanged")
        if metrics.get("child_reloaded") is not True:
            violations.append("training_child_reload")
        try:
            removed_parameters = float(metrics.get("removed_parameter_count", 0))
            child_blocks = float(metrics.get("child_num_blocks", 0))
            parent_blocks = float(metrics.get("parent_num_blocks", 0))
        except (TypeError, ValueError):
            removed_parameters, child_blocks, parent_blocks = 0.0, 0.0, 0.0
        if removed_parameters <= 0 or child_blocks >= parent_blocks:
            violations.append("training_no_structural_change")
        if metrics.get("offline_simulation") is True:
            violations.append("training_offline_simulation")
        return violations
    violations: List[str] = []
    steps = metrics.get("optimizer_steps")
    if not isinstance(steps, (int, float)) or isinstance(steps, bool) or steps <= 0:
        violations.append("training_optimizer_steps")
    gradient = metrics.get("gradient_norm", metrics.get("grad_norm"))
    if not isinstance(gradient, (int, float)) or not math.isfinite(float(gradient)) or gradient <= 0:
        violations.append("training_gradient")
    if metrics.get("changed_trainable_tensors", 0) <= 0:
        violations.append("training_no_changed_tensors")
    if metrics.get("unchanged_frozen_tensors", 0) <= 0:
        violations.append("training_frozen_parent_changed")
    before = metrics.get("parent_sha256_before") or metrics.get("parent_sha256")
    after = metrics.get("parent_sha256_after") or metrics.get("parent_sha256")
    child = metrics.get("child_sha256")
    if not before or not after or before != after:
        violations.append("training_parent_hash_changed")
    if not child or child == before:
        violations.append("training_child_hash_unchanged")
    if metrics.get("child_reloaded") is not True:
        violations.append("training_child_reload")
    return violations


def decide(inputs: AcceptanceInput) -> DecisionResult:
    training = dict(inputs.training_metrics or {})
    violations = _training_violations(training)
    benchmark = inputs.benchmark_summary
    quality = _get(benchmark, "quality_score")
    hardware = _get(benchmark, "hardware", {})
    parent_quality = _get(inputs.parent_summary, "quality_score")
    parent_hardware = _get(inputs.parent_summary, "hardware", {})
    hard_gates = _get(benchmark, "hard_gates", {}) or {}
    if not isinstance(hard_gates, Mapping):
        hard_gates = {}
    for key, label in (
        ("generation_valid", "generation_valid"),
        ("decode_success", "decode_success"),
        ("no_critical_temporal_collapse", "temporal_gate"),
    ):
        if hard_gates.get(key) is not True:
            violations.append(label)
    if quality is None:
        violations.append("quality_missing")
    quality_drop_limit = _target(inputs.target, "max_quality_drop")
    if quality is not None and parent_quality is not None and quality_drop_limit is not None:
        if float(parent_quality) - float(quality) > float(quality_drop_limit):
            violations.append("quality_gate")
    elif quality_drop_limit is not None:
        violations.append("quality_baseline_missing")

    reward_result = compute_reward(quality, hardware, parent_hardware, inputs.reward_weights)
    # Energy telemetry is optional for this phase.  Preserve the missing
    # field in the reward evidence, but do not turn ``energy_j=None`` into a
    # fake failure or invent a value just to make the scalar reward finite.
    for missing in reward_result.missing:
        if missing != "E":
            violations.append("metric_%s_missing" % missing)
    if inputs.research_grade:
        core_metrics = {
            "quality_score": quality,
            "latency_s": _get(hardware, "latency_s"),
            "peak_memory_gb": _get(hardware, "peak_memory_gb"),
        }
        if _target(inputs.target, "max_model_size_gb") is not None:
            core_metrics["model_size_gb"] = _get(hardware, "model_size_gb")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in core_metrics.values()):
            violations.append("research_grade_core_metrics_missing")

    efficiency_passed = False
    for name, threshold in inputs.efficiency_thresholds.items():
        before = _get(parent_hardware, name)
        after = _get(hardware, name)
        try:
            reduction = (float(before) - float(after)) / float(before) if float(before) > 0 else None
        except (TypeError, ValueError):
            reduction = None
        if reduction is not None and reduction >= float(threshold):
            efficiency_passed = True
    if not efficiency_passed:
        violations.append("efficiency_gate")

    for name in ("max_model_size_gb", "max_peak_memory_gb", "max_latency_s", "max_energy_j"):
        limit = _target(inputs.target, name)
        measured_name = {"max_model_size_gb": "model_size_gb", "max_peak_memory_gb": "peak_memory_gb", "max_latency_s": "latency_s", "max_energy_j": "energy_j"}[name]
        measured = _get(hardware, measured_name)
        if limit is not None and measured is not None and float(measured) > float(limit):
            violations.append("target_%s" % measured_name)
        elif limit is not None and measured is None:
            violations.append("metric_%s_missing" % measured_name)

    # Pareto/search evidence needs Q/L/M/S.  E is an optional objective when
    # the host cannot expose power telemetry; it remains None in the record.
    core_metric_values = [
        quality,
        _get(hardware, "latency_s"),
        _get(hardware, "peak_memory_gb"),
    ]
    if _target(inputs.target, "max_model_size_gb") is not None:
        core_metric_values.append(_get(hardware, "model_size_gb"))
    complete_metrics = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in core_metric_values
    )
    execution_violations = tuple(
        item
        for item in violations
        if item.startswith(("training_", "quantize_"))
    )
    if violations:
        status = "rejected" if execution_violations or any(item in {"quality_gate", "generation_valid", "decode_success", "temporal_gate", "efficiency_gate"} for item in violations) else "evaluated_candidate"
        accepted = False
    elif not inputs.research_grade:
        status = "evaluated_candidate"
        accepted = False
    else:
        status = "accepted"
        accepted = True
    pareto_eligible = bool(complete_metrics and not any(
        item.startswith(("training_", "quantize_"))
        or item in {"quality_gate", "generation_valid", "decode_success", "temporal_gate"}
        for item in violations
    ))
    if accepted:
        continuation_status = "final_accept"
        advance = True
    elif any(
        item.startswith(("training_", "quantize_"))
        or item in {"quality_gate", "generation_valid", "decode_success", "temporal_gate"}
        for item in violations
    ):
        continuation_status = "reject"
        advance = False
    elif inputs.on_pareto_front:
        continuation_status = "pareto_keep"
        advance = True
    elif complete_metrics:
        continuation_status = "exploratory_keep"
        advance = True
    else:
        continuation_status = "reject"
        advance = False
    return DecisionResult(
        status=status,
        accepted=accepted,
        violations=tuple(dict.fromkeys(violations)),
        reward=reward_result.reward,
        pareto_eligible=pareto_eligible,
        reward_terms=reward_result.terms,
        missing_metrics=reward_result.missing,
        continuation_status=continuation_status,
        advance=advance,
    )
