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
]
