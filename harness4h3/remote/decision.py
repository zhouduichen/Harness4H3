"""Evidence gates for deciding whether a remote H3 child is promotable."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class DecisionResult:
    status: str
    accepted: bool
    violations: Tuple[str, ...]
    reward: Optional[float]
    pareto_eligible: bool
    reward_terms: Mapping[str, float] = field(default_factory=dict)
    missing_metrics: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _training_violations(metrics: Mapping[str, Any]) -> List[str]:
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
    for missing in reward_result.missing:
        violations.append("metric_%s_missing" % missing)
    if inputs.research_grade and reward_result.reward is None:
        violations.append("research_grade_metrics_missing")

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

    complete_metrics = reward_result.reward is not None
    if violations:
        status = "rejected" if any(item.startswith("training_") or item in {"quality_gate", "generation_valid", "decode_success", "temporal_gate", "efficiency_gate"} for item in violations) else "evaluated_candidate"
        accepted = False
    elif not inputs.research_grade:
        status = "evaluated_candidate"
        accepted = False
    else:
        status = "accepted"
        accepted = True
    pareto_eligible = bool(accepted and complete_metrics)
    return DecisionResult(
        status=status,
        accepted=accepted,
        violations=tuple(dict.fromkeys(violations)),
        reward=reward_result.reward,
        pareto_eligible=pareto_eligible,
        reward_terms=reward_result.terms,
        missing_metrics=reward_result.missing,
    )

