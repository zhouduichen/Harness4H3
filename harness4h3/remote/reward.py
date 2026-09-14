"""Reward terms for measured H3 quality and hardware trade-offs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


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


@dataclass(frozen=True)
class RewardWeights:
    alpha: float
    beta: float
    gamma: float
    delta: float

    def __post_init__(self) -> None:
        values = (self.alpha, self.beta, self.gamma, self.delta)
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
            raise ValueError("reward weights must be finite and non-negative")


@dataclass(frozen=True)
class RewardResult:
    terms: Mapping[str, float]
    reward: Optional[float]
    missing: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_reward(
    quality: Any,
    hardware: Any,
    baseline: Any,
    weights: RewardWeights,
) -> RewardResult:
    quality_value = _finite(quality)
    if quality_value is None or quality_value < 0 or quality_value > 1:
        return RewardResult({}, None, ("Q",))
    values = {
        "L": (_finite(_value(hardware, "latency_s")), _finite(_value(baseline, "latency_s"), positive=True)),
        "M": (_finite(_value(hardware, "peak_memory_gb")), _finite(_value(baseline, "peak_memory_gb"), positive=True)),
        "E": (_finite(_value(hardware, "energy_j")), _finite(_value(baseline, "energy_j"), positive=True)),
    }
    missing = []
    terms: Dict[str, float] = {"Q": quality_value}
    for name, (candidate, parent) in values.items():
        if candidate is None or candidate < 0 or parent is None:
            missing.append(name)
            continue
        terms[name] = candidate / parent
    if missing:
        return RewardResult(terms, None, tuple(missing))
    reward = weights.alpha * terms["Q"] - weights.beta * terms["L"] - weights.gamma * terms["M"] - weights.delta * terms["E"]
    return RewardResult(terms, float(reward), ())

