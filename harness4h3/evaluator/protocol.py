from __future__ import annotations

from typing import Any, Mapping, Protocol, Tuple

from ..controller.schemas import HardwareMetrics
from ..h3.state import ModelState
from ..target.profile import TargetProfile


class QualityEvaluator(Protocol):
    def evaluate(self, state: ModelState, target: TargetProfile) -> Tuple[float, Mapping[str, Any]]:
        ...


class HardwareEvaluator(Protocol):
    def evaluate(self, state: ModelState, target: TargetProfile) -> HardwareMetrics:
        ...
