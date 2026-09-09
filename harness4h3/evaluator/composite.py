from __future__ import annotations

from ..controller.schemas import EvaluationResult
from ..h3.state import ModelState
from ..target.profile import TargetProfile
from .constraints import ConstraintEvaluator
from .protocol import HardwareEvaluator, QualityEvaluator


class CompositeEvaluator:
    def __init__(
        self,
        quality: QualityEvaluator,
        hardware: HardwareEvaluator,
        constraints: ConstraintEvaluator,
    ):
        self.quality = quality
        self.hardware = hardware
        self.constraints = constraints

    def evaluate(self, state: ModelState, target: TargetProfile, baseline_quality: float) -> EvaluationResult:
        quality_score, quality_metrics = self.quality.evaluate(state, target)
        hardware = self.hardware.evaluate(state, target)
        feasible, violations, critical = self.constraints.evaluate(quality_score, hardware, target, baseline_quality)
        return EvaluationResult(
            quality_score=quality_score,
            quality_metrics=dict(quality_metrics),
            hardware=hardware,
            feasible=feasible,
            violations=violations,
            critical_regression=critical,
        )
