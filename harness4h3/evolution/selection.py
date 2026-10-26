"""Cost-aware representatives from a verified candidate Pareto front."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ParetoRepresentative:
    candidate_id: str
    mode: str
    quality: float
    cost: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "quality": self.quality,
            "cost": self.cost,
            "selection_score": self.score,
        }


_WEIGHTS = {"lite": (1.0, 2.0), "medium": (1.0, 1.0), "heavy": (2.0, 1.0)}


def select_representatives(
    entries: Sequence[Mapping[str, Any]],
    *,
    baseline_quality: float,
    cost_metric: str = "inference_wall_time_s_mean",
) -> list[ParetoRepresentative]:
    """Choose up to three verified front entries using HarnessPro's rule.

    Inputs are flattened archive records containing ``candidate_id``,
    ``quality_score``, ``valid``, ``offline_simulation`` and the selected cost
    metric. Invalid evidence and candidates below the fixed quality floor are
    never eligible. Ties resolve by candidate ID.
    """

    floor = float(baseline_quality)
    if not math.isfinite(floor):
        raise ValueError("baseline_quality must be finite")
    eligible: dict[str, tuple[float, float]] = {}
    for item in entries:
        candidate_id = str(item.get("candidate_id", ""))
        hardware = item.get("hardware_metrics")
        hardware = hardware if isinstance(hardware, Mapping) else {}
        quality = item.get("quality_score")
        cost = item.get(cost_metric, hardware.get(cost_metric))
        if cost is None and cost_metric == "inference_wall_time_s_mean":
            cost = item.get("latency_s", hardware.get("latency_s"))
        if not candidate_id or item.get("valid") is not True or item.get("offline_simulation") is True:
            continue
        if isinstance(quality, bool) or isinstance(cost, bool):
            continue
        if not isinstance(quality, (int, float)) or not isinstance(cost, (int, float)):
            continue
        quality, cost = float(quality), float(cost)
        if not math.isfinite(quality) or not math.isfinite(cost) or cost < 0 or quality < floor:
            continue
        eligible[candidate_id] = (quality, cost)
    if not eligible:
        return []

    qmin = min(q for q, _ in eligible.values())
    qmax = max(q for q, _ in eligible.values())
    cmin = min(c for _, c in eligible.values())
    cmax = max(c for _, c in eligible.values())

    def distances(candidate_id: str) -> tuple[float, float]:
        quality, cost = eligible[candidate_id]
        dq = 0.0 if qmax == qmin else (qmax - quality) / (qmax - qmin)
        dc = 0.0 if cmax == cmin else (cost - cmin) / (cmax - cmin)
        return dq, dc

    chosen: list[ParetoRepresentative] = []
    selected: set[str] = set()
    for mode, (alpha, beta) in _WEIGHTS.items():
        ranked = sorted(
            (
                (alpha * distances(cid)[0] ** 2 + beta * distances(cid)[1] ** 2, cid)
                for cid in eligible if cid not in selected
            ),
            key=lambda pair: (pair[0], pair[1]),
        )
        if not ranked:
            break
        score, candidate_id = ranked[0]
        quality, cost = eligible[candidate_id]
        chosen.append(ParetoRepresentative(candidate_id, mode, quality, cost, score))
        selected.add(candidate_id)
    return chosen
