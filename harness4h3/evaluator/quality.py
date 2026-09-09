from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..h3.state import ModelState
from ..target.profile import TargetProfile


class FakeQualityEvaluator:
    def evaluate(self, state: ModelState, target: TargetProfile) -> Tuple[float, Mapping[str, Any]]:
        score = float(state.measured_metrics["quality_score"])
        return score, {"quality_score": score, "backend": "fake"}
