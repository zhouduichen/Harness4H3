from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..h3.state import ModelState
from ..harness.state import Task
from .h3 import BenchmarkSummary, H3BenchmarkRunner


@dataclass(frozen=True)
class MetricStatistics:
    count: int
    mean: Optional[float]
    median: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]
    stddev: Optional[float]
    ci95_low: Optional[float]
    ci95_high: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _statistics(values: Iterable[Optional[float]]) -> MetricStatistics:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not clean:
        return MetricStatistics(0, None, None, None, None, None, None, None)
    average = mean(clean)
    deviation = stdev(clean) if len(clean) >= 2 else None
    margin = 1.96 * deviation / math.sqrt(len(clean)) if deviation is not None else None
    return MetricStatistics(
        count=len(clean),
        mean=average,
        median=median(clean),
        minimum=min(clean),
        maximum=max(clean),
        stddev=deviation,
        ci95_low=average - margin if margin is not None else average,
        ci95_high=average + margin if margin is not None else average,
    )


def _summary_metrics(summary: BenchmarkSummary) -> Dict[str, Optional[float]]:
    black_rates = [
        float(run.quality_metrics["black_frame_ratio"])
        for run in summary.runs
        if isinstance(run.quality_metrics.get("black_frame_ratio"), (int, float))
    ]
    valid = [1.0 if run.semantic_generation_valid else 0.0 for run in summary.runs]
    return {
        "quality_score": summary.quality_score,
        "latency_s": summary.hardware.latency_s,
        "peak_memory_gb": summary.hardware.peak_memory_gb,
        "model_size_gb": summary.hardware.model_size_gb,
        "black_frame_rate": max(black_rates) if black_rates else None,
        "semantic_generation_valid_rate": mean(valid) if valid else None,
    }


def aggregate_summaries(summaries: Sequence[BenchmarkSummary]) -> Mapping[str, Mapping[str, Any]]:
    """Return reproducibility statistics without reducing failures to one scalar."""
    metrics = {name: [] for name in ("quality_score", "latency_s", "peak_memory_gb", "model_size_gb", "black_frame_rate", "semantic_generation_valid_rate")}
    for summary in summaries:
        for name, value in _summary_metrics(summary).items():
            metrics[name].append(value)
    return {name: _statistics(values).to_dict() for name, values in metrics.items()}


def _relative_reduction(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None or before <= 0:
        return None
    return (before - after) / before


@dataclass(frozen=True)
class M5ValidationResult:
    label: str
    baseline_model_id: str
    candidate_model_id: str
    target_profile_id: Optional[str]
    repetitions: int
    operator_attribution: Mapping[str, Any]
    conditions: Tuple[Mapping[str, Any], ...]
    aggregates: Mapping[str, Mapping[str, Mapping[str, Any]]]
    pairwise: Tuple[Mapping[str, Any], ...]
    validated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class M5ValidationRunner:
    """Run reproducibility/generalization checks around a fixed parent-child pair."""

    def __init__(self, benchmark: H3BenchmarkRunner):
        self.benchmark = benchmark

    def run(
        self,
        parent: ModelState,
        child: ModelState,
        tasks: Sequence[Task],
        *,
        label: str,
        repetitions: int = 2,
        target: Any = None,
        operator_attribution: Optional[Mapping[str, Any]] = None,
        black_frame_rate_threshold: float = 0.0,
        flip_order: bool = True,
    ) -> M5ValidationResult:
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if not tasks:
            raise ValueError("M5 validation requires at least one task")
        attribution = dict(operator_attribution or {})
        attribution.setdefault("primary_intervention", "quantization")
        attribution.setdefault("secondary_changes", [])
        attribution.setdefault("controlled_variables", [])
        conditions: List[Mapping[str, Any]] = []
        parent_summaries: List[BenchmarkSummary] = []
        child_summaries: List[BenchmarkSummary] = []
        pairwise: List[Mapping[str, Any]] = []

        for repetition in range(repetitions):
            parent_first = not flip_order or repetition % 2 == 0
            ordered = (("parent", parent), ("child", child)) if parent_first else (("child", child), ("parent", parent))
            summaries: Dict[str, BenchmarkSummary] = {}
            for order_index, (role, source_state) in enumerate(ordered):
                state = replace(source_state, model_id=f"{source_state.model_id}-m55-{label}-r{repetition}")
                summary = self.benchmark.run(
                    state,
                    tasks,
                    target=target,
                    operator_attribution={
                        **attribution,
                        "validation_label": label,
                        "validation_repetition": repetition,
                        "validation_role": role,
                    },
                    baseline_quality=(summaries["parent"].quality_score if role == "child" and "parent" in summaries else None),
                    baseline_hardware=(summaries["parent"].hardware if role == "child" and "parent" in summaries else None),
                    black_frame_rate_threshold=black_frame_rate_threshold,
                    reset_backend_before_run=True,
                )
                summaries[role] = summary
                conditions.append(
                    {
                        "label": label,
                        "repetition": repetition,
                        "order_index": order_index,
                        "role": role,
                        "model_id": source_state.model_id,
                        "summary": summary.to_dict(),
                    }
                )
            parent_summary = summaries["parent"]
            child_summary = summaries["child"]
            parent_summaries.append(parent_summary)
            child_summaries.append(child_summary)
            quality_delta = None
            if parent_summary.quality_score is not None and child_summary.quality_score is not None:
                quality_delta = child_summary.quality_score - parent_summary.quality_score
            quality_limit = getattr(target, "max_quality_drop", 0.05) if target is not None else 0.05
            quality_gate = quality_delta is not None and quality_delta >= -float(quality_limit)
            efficiency = {
                name: _relative_reduction(getattr(parent_summary.hardware, name), getattr(child_summary.hardware, name))
                for name in ("model_size_gb", "latency_s", "peak_memory_gb")
            }
            efficiency_gate = any(
                value is not None and value >= threshold
                for (name, value), threshold in zip(
                    efficiency.items(), (0.20, 0.15, 0.15)
                )
            )
            semantic_gate = bool(child_summary.hard_gates.get("generation_valid"))
            black_gate = child_summary.hard_gates.get("black_frame_gate") is True
            pairwise.append(
                {
                    "label": label,
                    "repetition": repetition,
                    "order": ["parent", "child"] if parent_first else ["child", "parent"],
                    "parent_quality": parent_summary.quality_score,
                    "child_quality": child_summary.quality_score,
                    "quality_delta": quality_delta,
                    "quality_drop_limit": float(quality_limit),
                    "quality_gate": quality_gate,
                    "efficiency_reductions": efficiency,
                    "efficiency_gate": efficiency_gate,
                    "semantic_generation_gate": semantic_gate,
                    "black_frame_gate": black_gate,
                    "target_feasible": child_summary.hard_gates.get("target_feasible"),
                    "validated": bool(quality_gate and efficiency_gate and semantic_gate and black_gate),
                }
            )

        aggregates = {
            "parent": aggregate_summaries(parent_summaries),
            "child": aggregate_summaries(child_summaries),
        }
        validated = bool(pairwise) and all(bool(item["validated"]) for item in pairwise)
        return M5ValidationResult(
            label=label,
            baseline_model_id=parent.model_id,
            candidate_model_id=child.model_id,
            target_profile_id=getattr(target, "id", None) if target is not None else None,
            repetitions=repetitions,
            operator_attribution=attribution,
            conditions=tuple(conditions),
            aggregates=aggregates,
            pairwise=tuple(pairwise),
            validated=validated,
        )
