from __future__ import annotations

from ..controller.schemas import HardwareMetrics
from ..h3.state import ModelState
from ..target.profile import TargetProfile


class FakeHardwareEvaluator:
    def evaluate(self, state: ModelState, target: TargetProfile) -> HardwareMetrics:
        metrics = state.measured_metrics
        return HardwareMetrics(
            latency_s=float(metrics["latency_s"]),
            peak_memory_gb=float(metrics["peak_memory_gb"]),
            model_size_gb=float(metrics["model_size_gb"]),
            energy_j=float(metrics["energy_j"]) if metrics.get("energy_j") is not None else None,
            throughput=float(metrics["throughput"]) if metrics.get("throughput") is not None else None,
            thermal=dict(metrics["thermal"]) if metrics.get("thermal") else None,
        )
