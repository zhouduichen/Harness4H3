"""Continuous metric verifiers and reward aggregation for Student candidates.

Hard validity remains the responsibility of :mod:`student.evaluator`.  This
module only turns measured evidence into normalized, directional metrics and a
scalar reward suitable for ranking candidates.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


def _finite(value: Any, *, positive: bool = False) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        return None
    return parsed


def _nested(source: Mapping[str, Any], key: str) -> Any:
    value = source.get(key)
    return value


@dataclass(frozen=True)
class MetricRewardWeights:
    """Non-negative coefficients for ``alpha*Q-beta*L-gamma*M-delta*E-epsilon*S``."""

    alpha: float = 1.0
    beta: float = 0.2
    gamma: float = 0.2
    delta: float = 0.2
    epsilon: float = 0.2

    def __post_init__(self) -> None:
        values = (self.alpha, self.beta, self.gamma, self.delta, self.epsilon)
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
            raise ValueError("metric reward weights must be finite and non-negative")


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str
    direction: str
    reference: float
    weight_name: str
    required: bool = True

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("metric direction must be maximize or minimize")
        if not math.isfinite(float(self.reference)) or float(self.reference) <= 0:
            raise ValueError("metric reference must be finite and positive")


@dataclass(frozen=True)
class MetricObservation:
    name: str
    value: Optional[float]
    normalized: Optional[float]
    unit: str
    direction: str
    reference: float
    weight: float
    required: bool
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricBankResult:
    metrics: Mapping[str, MetricObservation]
    reward: Optional[float]
    reward_terms: Mapping[str, float] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": {name: observation.to_dict() for name, observation in self.metrics.items()},
            "reward": self.reward,
            "reward_terms": dict(self.reward_terms),
            "missing": list(self.missing),
            "invalid": list(self.invalid),
        }


@dataclass(frozen=True)
class TeacherRelativeMetrics:
    """Teacher-relative quality and incumbent-comparable efficiency evidence."""

    quality_ratio: float
    latency: float
    memory: float
    size: float
    energy: Optional[float] = None
    efficiency_deltas: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_ratio": float(self.quality_ratio),
            "latency": float(self.latency),
            "memory": float(self.memory),
            "size": float(self.size),
            "energy": float(self.energy) if self.energy is not None else None,
            "efficiency_deltas": dict(self.efficiency_deltas),
        }


@dataclass(frozen=True)
class ParetoDecision:
    """Strict promotion decision for a valid Student candidate."""

    promotable: bool
    reason: str
    reward: Optional[float]
    incumbent_reward: Optional[float]
    reward_delta: Optional[float]
    improved_metrics: tuple[str, ...] = ()
    regressed_metrics: tuple[str, ...] = ()
    quality_floor_ratio: float = 0.90

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotable": self.promotable,
            "reason": self.reason,
            "reward": self.reward,
            "incumbent_reward": self.incumbent_reward,
            "reward_delta": self.reward_delta,
            "improved_metrics": list(self.improved_metrics),
            "regressed_metrics": list(self.regressed_metrics),
            "quality_floor_ratio": self.quality_floor_ratio,
        }


def _bounded_log_delta(reference: float, value: float) -> float:
    if reference <= 0 or value <= 0:
        raise ValueError("metric references must be positive")
    return max(-1.0, min(1.0, math.log(float(reference) / float(value))))


def teacher_relative_metrics(
    student: Mapping[str, float],
    teacher: Mapping[str, float],
) -> TeacherRelativeMetrics:
    """Normalize a Student observation against the H3 teacher quality baseline."""

    quality = _finite(student.get("quality"))
    teacher_quality = _finite(teacher.get("quality"), positive=True)
    latency = _finite(student.get("latency"), positive=True)
    memory = _finite(student.get("memory"), positive=True)
    size = _finite(student.get("size"), positive=True)
    if quality is None or teacher_quality is None or latency is None or memory is None or size is None:
        raise ValueError("quality, latency, memory, and size are required for teacher-relative metrics")
    energy = _finite(student.get("energy"), positive=True)
    teacher_energy = _finite(teacher.get("energy"), positive=True)
    deltas = {}
    for name, value in (("latency", latency), ("memory", memory), ("size", size)):
        reference = _finite(teacher.get(name), positive=True)
        if reference is not None:
            deltas[name] = _bounded_log_delta(reference, value)
    if energy is not None and teacher_energy is not None:
        deltas["energy"] = _bounded_log_delta(teacher_energy, energy)
    return TeacherRelativeMetrics(
        quality_ratio=float(quality / teacher_quality),
        latency=float(latency),
        memory=float(memory),
        size=float(size),
        energy=float(energy) if energy is not None else None,
        efficiency_deltas=deltas,
    )


def _relative_score(
    quality_ratio: float,
    values: Mapping[str, float],
    incumbent: Mapping[str, float],
) -> float:
    terms = {"quality": 0.60 * float(quality_ratio)}
    weights = {"latency": 0.20, "memory": 0.10, "size": 0.05, "energy": 0.05}
    present = [name for name in weights if name in values and name in incumbent]
    scale = 1.0 / (0.60 + sum(weights[name] for name in present))
    score = terms["quality"]
    for name in present:
        score += weights[name] * _bounded_log_delta(float(incumbent[name]), float(values[name]))
    return float(score * scale)


def normalize_reward(
    relative: TeacherRelativeMetrics,
    *,
    incumbent: Mapping[str, float],
) -> float:
    """Compute a bounded quality-efficiency score against the current incumbent."""

    values = {
        "latency": relative.latency,
        "memory": relative.memory,
        "size": relative.size,
    }
    if relative.energy is not None:
        values["energy"] = relative.energy
    return _relative_score(relative.quality_ratio, values, incumbent)


def pareto_decision(
    *,
    candidate: Mapping[str, float],
    incumbent: Optional[Mapping[str, float]],
    quality_floor_ratio: float = 0.90,
    min_reward_delta: float = 0.02,
    material_efficiency_gain: float = 0.05,
    max_regression: float = 0.02,
) -> ParetoDecision:
    """Decide whether a valid Student improves the quality-efficiency frontier."""

    quality_ratio = _finite(candidate.get("quality_ratio"))
    if quality_ratio is None or quality_ratio < float(quality_floor_ratio):
        return ParetoDecision(False, "quality_floor_failed", None, None, None, quality_floor_ratio=quality_floor_ratio)
    if incumbent is None:
        return ParetoDecision(True, "initial_feasible", None, None, None, quality_floor_ratio=quality_floor_ratio)
    incumbent_quality = _finite(incumbent.get("quality_ratio"))
    if incumbent_quality is None:
        raise ValueError("incumbent quality_ratio is required")
    comparable = ("latency", "memory", "size", "energy")
    improved = []
    regressed = []
    for name in comparable:
        candidate_value = _finite(candidate.get(name), positive=True)
        incumbent_value = _finite(incumbent.get(name), positive=True)
        if candidate_value is None or incumbent_value is None:
            continue
        relative_change = (float(incumbent_value) - float(candidate_value)) / float(incumbent_value)
        if relative_change >= float(material_efficiency_gain):
            improved.append(name)
        if relative_change < -float(max_regression):
            regressed.append(name)
    candidate_values = {name: candidate[name] for name in comparable if name in candidate and name in incumbent}
    incumbent_values = {name: incumbent[name] for name in comparable if name in candidate and name in incumbent}
    reward = _relative_score(quality_ratio, candidate_values, incumbent_values)
    incumbent_reward = _relative_score(incumbent_quality, incumbent_values, incumbent_values)
    reward_delta = reward - incumbent_reward
    quality_improved = quality_ratio >= incumbent_quality + 0.01 and not regressed
    material_pareto = bool(improved) and not regressed and quality_ratio >= incumbent_quality - float(max_regression)
    if reward_delta >= float(min_reward_delta) or material_pareto or quality_improved:
        return ParetoDecision(
            True,
            "pareto_improvement",
            reward,
            incumbent_reward,
            reward_delta,
            tuple(improved),
            tuple(regressed),
            quality_floor_ratio,
        )
    return ParetoDecision(
        False,
        "pareto_rejected",
        reward,
        incumbent_reward,
        reward_delta,
        tuple(improved),
        tuple(regressed),
        quality_floor_ratio,
    )


class MetricVerifierBank:
    """Extract Q/L/M/E/S from evaluator evidence and aggregate a reward.

    References are explicit normalization scales, not claims about measured
    hardware.  Deployments can replace them with target-device values without
    changing the evidence schema.
    """

    DEFAULT_DEFINITIONS = (
        MetricDefinition("quality", "score", "maximize", 1.0, "alpha", True),
        MetricDefinition("latency", "ms", "minimize", 100.0, "beta", True),
        MetricDefinition("memory", "GB", "minimize", 8.0, "gamma", True),
        MetricDefinition("energy", "J", "minimize", 12.0, "delta", False),
        MetricDefinition("size", "GB", "minimize", 4.7, "epsilon", True),
    )

    def __init__(
        self,
        *,
        definitions: tuple[MetricDefinition, ...] = DEFAULT_DEFINITIONS,
        weights: MetricRewardWeights = MetricRewardWeights(),
    ):
        if not definitions:
            raise ValueError("MetricVerifierBank requires at least one definition")
        names = [definition.name for definition in definitions]
        if len(names) != len(set(names)):
            raise ValueError("metric names must be unique")
        self.definitions = tuple(definitions)
        self.weights = weights

    @staticmethod
    def _extract(name: str, quality: Mapping[str, Any], hardware: Mapping[str, Any]) -> tuple[Any, str]:
        if name == "quality":
            if "score" in quality:
                return quality["score"], "quality.score"
            return quality.get("quality_score"), "quality.quality_score"
        if name == "latency":
            if "latency_ms" in hardware:
                return hardware["latency_ms"], "hardware.latency_ms"
            value = _finite(hardware.get("latency_s"))
            return (value * 1000.0 if value is not None else None), "hardware.latency_s*1000"
        if name == "memory":
            for key in ("peak_memory_gb", "peak_vram_gb", "training_peak_memory_gb"):
                if hardware.get(key) is not None:
                    return hardware[key], "hardware.%s" % key
            return None, "hardware.peak_memory_gb"
        if name == "energy":
            return hardware.get("energy_j"), "hardware.energy_j"
        if name == "size":
            if hardware.get("model_size_gb") is not None:
                return hardware["model_size_gb"], "hardware.model_size_gb"
            checkpoint_bytes = _finite(hardware.get("checkpoint_bytes"))
            return (
                checkpoint_bytes / float(1024**3) if checkpoint_bytes is not None else None,
                "hardware.checkpoint_bytes/2^30",
            )
        return None, "unknown"

    def evaluate(self, quality: Mapping[str, Any] | None, hardware: Mapping[str, Any] | None) -> MetricBankResult:
        quality = dict(quality or {})
        hardware = dict(hardware or {})
        observations: dict[str, MetricObservation] = {}
        missing: list[str] = []
        invalid: list[str] = []
        terms: dict[str, float] = {}
        required_missing = False
        for definition in self.definitions:
            raw, source = self._extract(definition.name, quality, hardware)
            value = _finite(raw)
            if value is None:
                missing.append(definition.name)
                required_missing = required_missing or definition.required
                observations[definition.name] = MetricObservation(
                    definition.name, None, None, definition.unit, definition.direction,
                    definition.reference, float(getattr(self.weights, definition.weight_name)),
                    definition.required, source,
                )
                continue
            if value < 0 or (definition.name == "quality" and value > 1):
                invalid.append(definition.name)
                observations[definition.name] = MetricObservation(
                    definition.name, value, None, definition.unit, definition.direction,
                    definition.reference, float(getattr(self.weights, definition.weight_name)),
                    definition.required, source,
                )
                continue
            normalized = value / definition.reference
            weight = float(getattr(self.weights, definition.weight_name))
            observations[definition.name] = MetricObservation(
                definition.name, value, normalized, definition.unit, definition.direction,
                definition.reference, weight, definition.required, source,
            )
            signed = normalized if definition.direction == "maximize" else -normalized
            terms[definition.name] = weight * signed
        reward = None if required_missing or invalid else float(sum(terms.values()))
        return MetricBankResult(observations, reward, terms, tuple(missing), tuple(invalid))


__all__ = [
    "MetricBankResult",
    "MetricDefinition",
    "MetricObservation",
    "MetricRewardWeights",
    "MetricVerifierBank",
    "ParetoDecision",
    "TeacherRelativeMetrics",
    "normalize_reward",
    "pareto_decision",
    "teacher_relative_metrics",
]
