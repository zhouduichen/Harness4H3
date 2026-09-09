from __future__ import annotations

from typing import List, Tuple

from ..controller.schemas import HardwareMetrics
from ..target.profile import TargetProfile


class ConstraintEvaluator:
    def evaluate(
        self,
        quality_score: float,
        hardware: HardwareMetrics,
        target: TargetProfile,
        baseline_quality: float,
    ) -> Tuple[bool, List[str], bool]:
        violations: List[str] = []
        limits = (
            ("max_model_size_gb", hardware.model_size_gb, target.max_model_size_gb),
            ("max_peak_memory_gb", hardware.peak_memory_gb, target.max_peak_memory_gb),
            ("max_latency_s", hardware.latency_s, target.max_latency_s),
            ("max_energy_j", hardware.energy_j, target.max_energy_j),
        )
        for name, value, limit in limits:
            if limit is not None and (value is None or value > limit):
                violations.append(name)
        quality_floor = target.min_quality_score
        if target.max_quality_drop is not None:
            drop_floor = baseline_quality - target.max_quality_drop
            quality_floor = drop_floor if quality_floor is None else max(quality_floor, drop_floor)
        critical = quality_floor is not None and quality_score < quality_floor
        if critical:
            violations.append("quality")
        return not violations, violations, critical
