from __future__ import annotations

from typing import Mapping, Optional

from ..archive.system_candidate import SystemCandidate
from ..controller.schemas import EvaluationRecord
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

    def evaluate(
        self,
        state: ModelState,
        target: TargetProfile,
        baseline_quality: float,
        system: Optional[SystemCandidate] = None,
        device_id: Optional[str] = None,
        task_split: Optional[str] = None,
        benchmark_recipe: Optional[Mapping[str, object]] = None,
    ) -> EvaluationRecord:
        evaluation_state = state
        system_id = None
        if system is not None:
            if system.model_ref != state.model_id:
                raise ValueError(
                    "system %s references %s, but evaluator received model %s"
                    % (system.id, system.model_ref, state.model_id)
                )
            evaluation_state = system.evaluation_state(state)
            system_id = system.id
        quality_score, quality_metrics = self.quality.evaluate(evaluation_state, target)
        hardware = self.hardware.evaluate(evaluation_state, target)
        feasible, violations, critical = self.constraints.evaluate(quality_score, hardware, target, baseline_quality)
        metrics = {
            "quality_score": quality_score,
            "latency_s": hardware.latency_s,
            "peak_memory_gb": hardware.peak_memory_gb,
            "model_size_gb": hardware.model_size_gb,
            "energy_j": hardware.energy_j,
        }
        return EvaluationRecord(
            quality_score=quality_score,
            quality_metrics=dict(quality_metrics),
            hardware=hardware,
            feasible=feasible,
            violations=violations,
            critical_regression=critical,
            model_id=state.model_id,
            system_id=system_id,
            device_id=device_id,
            task_split=task_split,
            validity={"quality_measured": True, "hardware_measured": True},
            provenance={
                "evaluator": self.__class__.__name__,
                "benchmark_recipe": dict(benchmark_recipe or {}),
                "offline_simulation": bool(getattr(self.quality, "offline_simulation", True)),
            },
            search_score=target.objective_score(metrics),
        )
