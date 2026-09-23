"""Hard feasibility and soft objective gates for campaign candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
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
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
    "latency": "latency",
    "latency_s": "latency",
    "server_latency": "latency",
    "edge_latency": "latency",
    "memory": "memory",
    "peak_memory_gb": "memory",
    "training_peak_memory_gb": "memory",
    "server_memory": "memory",
    "edge_memory": "memory",
    "size": "model_size",
    "model_size_gb": "model_size",
    "edge_model_size": "model_size",
    "energy": "energy",
    "energy_j": "energy",
    "edge_energy": "energy",
}

_SERVER_METRICS = {
    "quality": ("quality",),
    "latency": ("server_latency", "latency_s"),
    "memory": ("server_memory", "peak_memory_gb", "training_peak_memory_gb"),
    "energy": ("energy_j",),
    "model_size": ("model_size_gb",),
}

_EDGE_METRICS = {
    "quality": ("quality",),
    "latency": ("edge_latency",),
    "memory": ("edge_memory",),
    "energy": ("edge_energy",),
    "model_size": ("edge_model_size",),
}

_EDGE_REQUIRED_EVIDENCE = (
    "edge_exported",
    "edge_quantized",
    "edge_runtime",
    "edge_device",
    "edge_latency",
    "edge_memory",
    "edge_energy",
    "edge_thermal",
    "edge_model_size",
)


def _metric_name(value: str) -> str:
    return _ALIASES.get(value, value)


def _find_evidence(evidence: Mapping[str, MetricEvidence], name: str) -> Optional[MetricEvidence]:
    target = _metric_name(name)
    # Prefer an exact metric before alias fallback. In particular, an edge
    # metric must never resolve to an earlier server proxy merely because both
    # share the same canonical optimization name.
    for key, item in evidence.items():
        if key == name or item.metric_name == name:
            return item
    for key, item in evidence.items():
        if _metric_name(str(key)) == target or _metric_name(item.metric_name) == target:
            return item
    return None


def canonical_metric_name(name: str) -> str:
    """Return the stable optimization name for server or edge aliases."""

    return _ALIASES.get(str(name), str(name))


def _constraint_metric(name: str) -> Tuple[str, Optional[str]]:
    if name.startswith("max_"):
        return _metric_name(name[4:]), "max"
    if name.startswith("min_"):
        return _metric_name(name[4:]), "min"
    return _metric_name(name), None


def _edge_evidence_complete(evidence: Mapping[str, MetricEvidence]) -> bool:
    explicit = _find_evidence(evidence, "edge_evidence_complete")
    if explicit is not None:
        if (
            explicit.value is None
            or not explicit.valid
            or float(explicit.value) <= 0
            or explicit.device_profile_id == "server"
        ):
            return False
    return all(
        (item := _find_evidence(evidence, name)) is not None
        and item.value is not None
        and item.valid
        and float(item.value) > 0
        and item.device_profile_id != "server"
        for name in _EDGE_REQUIRED_EVIDENCE
    )


def effective_metric_set(evidence: Mapping[str, MetricEvidence]) -> Mapping[str, MetricEvidence]:
    """Resolve one metric set for every optimization consumer.

    Server measurements remain proxies until the complete target-device tuple
    is present. Once it is present, every non-quality optimization metric is
    sourced from the target device, so a server-only latency can never decide a
    target-device parent.
    """

    candidates = _EDGE_METRICS if _edge_evidence_complete(evidence) else _SERVER_METRICS
    result: Dict[str, MetricEvidence] = {}
    for canonical, names in candidates.items():
        for name in names:
            item = _find_evidence(evidence, name)
            if item is not None and item.value is not None and item.valid:
                result[canonical] = item
                break
    return result


class AcceptanceGate:
    def evaluate(
        self,
        candidate: CandidateEnvelope,
        evidence: Mapping[str, MetricEvidence],
        *,
        hard_constraints: Mapping[str, Any],
        objectives: Mapping[str, str],
        min_rounds_met: bool,
        target_device_profile: Optional[Any] = None,
    ) -> GateDecision:
        violations = []
        evidence_ids = []
        edge_complete = _edge_evidence_complete(evidence)
        for item in evidence.values():
            evidence_ids.append(item.input_reference)
        for constraint, threshold in hard_constraints.items():
            metric, direction = _constraint_metric(str(constraint))
            item = _find_evidence(evidence, metric)
            if edge_complete:
                effective = effective_metric_set(evidence)
                item = effective.get(canonical_metric_name(metric), item)
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

        profile_passed = False
        profile_missing_or_incomplete = False
        if target_device_profile is None:
            profile_missing_or_incomplete = True
        elif not edge_complete:
            profile_missing_or_incomplete = True
        else:
            profile = target_device_profile
            edge_devices = {
                item.device_profile_id
                for item in evidence.values()
                if str(item.metric_name).startswith("edge_")
            }
            if edge_devices != {str(profile.id)}:
                violations.append("target_device_identity")
            profile_checks = (
                ("edge_latency", float(profile.max_latency_s), "max"),
                ("edge_memory", float(profile.max_edge_memory_gb), "max"),
                ("edge_energy", float(profile.max_energy_j), "max"),
                ("edge_thermal", float(profile.max_thermal_c), "max"),
                ("edge_model_size", float(profile.max_model_size_gb), "max"),
            )
            profile_passed = True
            for metric_name, limit, direction in profile_checks:
                item = _find_evidence(evidence, metric_name)
                if item is None or item.value is None or not item.valid:
                    violations.append("%s:missing_or_invalid" % metric_name)
                    profile_passed = False
                    continue
                if direction == "max" and float(item.value) > limit:
                    violations.append("target_%s" % metric_name)
                    profile_passed = False
            device_item = _find_evidence(evidence, "edge_device")
            metadata = dict(device_item.metadata) if device_item is not None else {}
            required_metadata = (
                "runtime_backend",
                "precision",
                "quantization",
                "resolution",
                "frames",
                "sampling_steps",
            )
            missing_metadata = [
                name for name in required_metadata
                if name not in metadata or metadata[name] is None or metadata[name] == ""
            ]
            for name in missing_metadata:
                violations.append("target_%s:missing_or_invalid" % name)
                profile_passed = False
            runtime_backend = metadata.get("runtime_backend")
            if runtime_backend != profile.runtime_backend:
                violations.append("target_runtime_backend")
                profile_passed = False
            precision = metadata.get("precision")
            if precision not in profile.supported_precision:
                violations.append("target_precision")
                profile_passed = False
            quantization = metadata.get("quantization")
            if quantization not in profile.supported_quantization:
                violations.append("target_quantization")
                profile_passed = False
            try:
                resolution = tuple(int(item) for item in metadata["resolution"])
            except (KeyError, TypeError, ValueError):
                resolution = None
            if resolution != tuple(profile.resolution):
                violations.append("target_resolution")
                profile_passed = False
            for name, expected in (("frames", profile.frames), ("sampling_steps", profile.sampling_steps)):
                try:
                    actual = int(metadata[name])
                except (KeyError, TypeError, ValueError):
                    actual = None
                if isinstance(metadata.get(name), bool) or actual != int(expected):
                    violations.append("target_%s" % name)
                    profile_passed = False
            candidate_deployment = getattr(candidate, "deployment_recipe", {})
            if isinstance(candidate_deployment, Mapping):
                precision = candidate_deployment.get("precision")
                quantization = candidate_deployment.get("quantization")
                if precision and str(precision) not in profile.supported_precision:
                    violations.append("target_precision")
                    profile_passed = False
                if quantization and str(quantization) not in profile.supported_quantization:
                    violations.append("target_quantization")
                    profile_passed = False
        objective_values: Dict[str, float] = {}
        if not violations:
            effective = effective_metric_set(evidence)
            for objective in objectives:
                canonical = canonical_metric_name(str(objective))
                item = effective.get(canonical)
                if item is not None and item.value is not None and item.valid:
                    objective_values[canonical] = float(item.value)
        feasible = not violations
        promotable = feasible
        target_satisfied = feasible and promotable and bool(min_rounds_met) and edge_complete and profile_passed
        reason = "feasible" if feasible else "hard_constraint_failure"
        if feasible and not min_rounds_met:
            reason = "promotable_before_target_satisfied"
        elif feasible and not edge_complete:
            reason = "promotable_for_edge_test"
        elif feasible and (profile_missing_or_incomplete or not profile_passed):
            reason = "target_device_profile_failure"
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
    left_effective = effective_metric_set(left)
    right_effective = effective_metric_set(right)
    for name, direction in objectives.items():
        canonical = canonical_metric_name(str(name))
        left_item = left_effective.get(canonical)
        right_item = right_effective.get(canonical)
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


__all__ = [
    "AcceptanceGate",
    "GateDecision",
    "MetricEvidence",
    "canonical_metric_name",
    "effective_metric_set",
    "pareto_dominates",
]
