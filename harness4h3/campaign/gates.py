"""Hard feasibility and soft objective gates for campaign candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .proposals import CandidateEnvelope


@dataclass(frozen=True)
class MetricEvidence:
    metric_name: str
    metric_version: str
    input_reference: str
    value: Optional[float]
    confidence_or_validity: Any
    evidence_source: str
    device_profile_id: str
    hard: bool = False

    def __post_init__(self) -> None:
        for name in ("metric_name", "metric_version", "input_reference", "evidence_source", "device_profile_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("%s must be a non-empty string" % name)
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(float(self.value)):
                raise ValueError("metric value must be finite")
        if isinstance(self.confidence_or_validity, bool):
            return
        if isinstance(self.confidence_or_validity, (int, float)) and math.isfinite(float(self.confidence_or_validity)):
            return
        raise ValueError("confidence_or_validity must be boolean or finite numeric")

    @property
    def valid(self) -> bool:
        if isinstance(self.confidence_or_validity, bool):
            return self.confidence_or_validity
        return float(self.confidence_or_validity) > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    feasible: bool
    promotable: bool
    target_satisfied: bool
    violations: Tuple[str, ...]
    objective_values: Mapping[str, float]
    evidence_ids: Tuple[str, ...]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["violations"] = list(self.violations)
        value["evidence_ids"] = list(self.evidence_ids)
        return value


_ALIASES = {
    "quality_score": "quality",
    "quality": "quality",
    "latency": "latency_s",
    "latency_s": "latency_s",
    "memory": "peak_memory_gb",
    "peak_memory_gb": "peak_memory_gb",
    "size": "model_size_gb",
    "model_size_gb": "model_size_gb",
    "energy": "energy_j",
    "energy_j": "energy_j",
}


def _metric_name(value: str) -> str:
    return _ALIASES.get(value, value)


def _find_evidence(evidence: Mapping[str, MetricEvidence], name: str) -> Optional[MetricEvidence]:
    target = _metric_name(name)
    for key, item in evidence.items():
        if key == name or _metric_name(str(key)) == target or _metric_name(item.metric_name) == target:
            return item
    return None


def _constraint_metric(name: str) -> Tuple[str, Optional[str]]:
    if name.startswith("max_"):
        return _metric_name(name[4:]), "max"
    if name.startswith("min_"):
        return _metric_name(name[4:]), "min"
    return _metric_name(name), None


class AcceptanceGate:
    def evaluate(
        self,
        candidate: CandidateEnvelope,
        evidence: Mapping[str, MetricEvidence],
        *,
        hard_constraints: Mapping[str, Any],
        objectives: Mapping[str, str],
        min_rounds_met: bool,
    ) -> GateDecision:
        violations = []
        evidence_ids = []
        for item in evidence.values():
            evidence_ids.append(item.input_reference)
        for constraint, threshold in hard_constraints.items():
            metric, direction = _constraint_metric(str(constraint))
            item = _find_evidence(evidence, metric)
            if item is None or item.value is None or not item.valid:
                violations.append("%s:missing_or_invalid" % constraint)
                continue
            try:
                limit = float(threshold)
                value = float(item.value)
            except (TypeError, ValueError):
                violations.append("%s:invalid_threshold" % constraint)
                continue
            if direction == "max" and value > limit:
                violations.append(str(constraint))
            elif direction == "min" and value < limit:
                violations.append(str(constraint))
            elif direction is None and bool(threshold) != bool(value):
                violations.append(str(constraint))
        objective_values: Dict[str, float] = {}
        if not violations:
            for objective in objectives:
                item = _find_evidence(evidence, str(objective))
                if item is not None and item.value is not None and item.valid:
                    objective_values[str(objective)] = float(item.value)
        feasible = not violations
        promotable = feasible
        target_satisfied = feasible and promotable and bool(min_rounds_met)
        reason = "feasible" if feasible else "hard_constraint_failure"
        if feasible and not min_rounds_met:
            reason = "promotable_before_target_satisfied"
        return GateDecision(
            feasible=feasible,
            promotable=promotable,
            target_satisfied=target_satisfied,
            violations=tuple(dict.fromkeys(violations)),
            objective_values=objective_values,
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            reason=reason,
        )


def pareto_dominates(
    left: Mapping[str, MetricEvidence],
    right: Mapping[str, MetricEvidence],
    objectives: Mapping[str, str],
) -> bool:
    strictly_better = False
    for name, direction in objectives.items():
        left_item = _find_evidence(left, str(name))
        right_item = _find_evidence(right, str(name))
        if left_item is None or right_item is None or left_item.value is None or right_item.value is None:
            return False
        left_value = float(left_item.value)
        right_value = float(right_item.value)
        if direction == "maximize":
            if left_value < right_value:
                return False
            strictly_better = strictly_better or left_value > right_value
        elif direction == "minimize":
            if left_value > right_value:
                return False
            strictly_better = strictly_better or left_value < right_value
        else:
            raise ValueError("objective direction must be maximize or minimize")
    return strictly_better


__all__ = ["AcceptanceGate", "GateDecision", "MetricEvidence", "pareto_dominates"]
