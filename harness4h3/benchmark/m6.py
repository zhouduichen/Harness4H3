from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..h3.state import ModelState
from ..harness.state import Task
from .h3 import BenchmarkSummary, H3BenchmarkRunner


def _stats(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None, "stddev": None, "ci95_low": None, "ci95_high": None}
    average = mean(clean)
    deviation = stdev(clean) if len(clean) > 1 else None
    margin = 1.96 * deviation / math.sqrt(len(clean)) if deviation is not None else None
    return {
        "count": len(clean),
        "mean": average,
        "median": median(clean),
        "minimum": min(clean),
        "maximum": max(clean),
        "stddev": deviation,
        "ci95_low": average - margin if margin is not None else average,
        "ci95_high": average + margin if margin is not None else average,
    }


def _summary_metrics(summary: BenchmarkSummary) -> Dict[str, Optional[float]]:
    black = [
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
        "black_frame_rate": max(black) if black else None,
        "semantic_generation_valid_rate": mean(valid) if valid else None,
    }


def _aggregate(summaries: Sequence[BenchmarkSummary]) -> Mapping[str, Mapping[str, Any]]:
    names = ("quality_score", "latency_s", "peak_memory_gb", "model_size_gb", "black_frame_rate", "semantic_generation_valid_rate")
    result = {name: _stats(_summary_metrics(summary)[name] for summary in summaries) for name in names}
    # Keep the hardware-neutral key used by the existing benchmark and expose
    # the explicit VRAM vocabulary required by the M6 evidence contract.
    result["peak_vram_gb"] = dict(result["peak_memory_gb"])
    return result


def _reduction(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None or before <= 0:
        return None
    return (before - after) / before


@dataclass(frozen=True)
class M6ValidationResult:
    label: str
    operator: str
    parent_model_id: str
    branch_model_id: str
    target_profile_id: Optional[str]
    repetitions: int
    reference_metrics: Mapping[str, Any]
    conditions: Tuple[Mapping[str, Any], ...]
    aggregates: Mapping[str, Mapping[str, Mapping[str, Any]]]
    gates: Mapping[str, Any]
    validated: bool
    operator_attribution: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class M6ValidationRunner:
    """Validate one runtime-memory branch against the M5.5/reference recipe."""

    def __init__(self, benchmark: H3BenchmarkRunner):
        self.benchmark = benchmark

    def run(
        self,
        parent: ModelState,
        branch: ModelState,
        tasks: Sequence[Task],
        *,
        label: str,
        target: Any,
        reference_metrics: Mapping[str, Any],
        operator_attribution: Optional[Mapping[str, Any]] = None,
        repetitions: int = 1,
        black_frame_rate_threshold: float = 0.0,
        min_latency_reduction: float = 0.15,
        min_model_size_reduction: float = 0.20,
    ) -> M6ValidationResult:
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if not tasks:
            raise ValueError("M6 validation requires at least one task")
        if not 0.0 <= float(black_frame_rate_threshold) <= 1.0:
            raise ValueError("black_frame_rate_threshold must be between 0 and 1")
        parent_summaries: List[BenchmarkSummary] = []
        branch_summaries: List[BenchmarkSummary] = []
        conditions: List[Mapping[str, Any]] = []
        attribution = dict(operator_attribution or {})
        attribution.setdefault("primary_intervention", "runtime_memory")
        attribution.setdefault("secondary_changes", [])
        attribution.setdefault("controlled_variables", [])
        for repetition in range(repetitions):
            summaries: Dict[str, BenchmarkSummary] = {}
            for role, source in (("parent", parent), ("branch", branch)):
                state = replace(source, model_id=f"{source.model_id}-m6-{label}-r{repetition}")
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
                    baseline_quality=(summaries["parent"].quality_score if role == "branch" else None),
                    baseline_hardware=(summaries["parent"].hardware if role == "branch" else None),
                    black_frame_rate_threshold=black_frame_rate_threshold,
                    reset_backend_before_run=True,
                )
                summaries[role] = summary
                conditions.append({"label": label, "repetition": repetition, "role": role, "summary": summary.to_dict()})
            parent_summaries.append(summaries["parent"])
            branch_summaries.append(summaries["branch"])

        parent_aggregate = _aggregate(parent_summaries)
        branch_aggregate = _aggregate(branch_summaries)
        parent_quality = parent_aggregate["quality_score"]["mean"]
        branch_quality = branch_aggregate["quality_score"]["mean"]
        quality_delta = None if parent_quality is None or branch_quality is None else branch_quality - parent_quality
        quality_limit = float(getattr(target, "max_quality_drop", 0.05))
        peak_max = branch_aggregate["peak_memory_gb"]["maximum"]
        target_peak = getattr(target, "max_peak_memory_gb", None)
        reference_size = reference_metrics.get("model_size_gb")
        reference_latency = reference_metrics.get("latency_s")
        branch_size = branch_aggregate["model_size_gb"]["mean"]
        branch_latency = branch_aggregate["latency_s"]["mean"]
        size_reduction = _reduction(reference_size, branch_size)
        latency_reduction = _reduction(reference_latency, branch_latency)
        generation_valid = branch_aggregate["semantic_generation_valid_rate"]["minimum"] == 1.0
        black_ok = (branch_aggregate["black_frame_rate"]["maximum"] is not None and branch_aggregate["black_frame_rate"]["maximum"] <= black_frame_rate_threshold)
        decode_ok = all(summary.hard_gates.get("decode_success") is True for summary in branch_summaries)
        quality_ok = quality_delta is not None and quality_delta >= -quality_limit
        size_ok = size_reduction is not None and size_reduction >= float(min_model_size_reduction)
        latency_ok = latency_reduction is not None and latency_reduction >= float(min_latency_reduction)
        peak_ok = target_peak is not None and peak_max is not None and peak_max <= float(target_peak)
        gates: Dict[str, Any] = {
            "generation_valid": generation_valid,
            "decode_success": decode_ok,
            "black_frame_rate_max": branch_aggregate["black_frame_rate"]["maximum"],
            "black_frame_gate": black_ok,
            "quality_delta": quality_delta,
            "quality_drop_limit": quality_limit,
            "quality_gate": quality_ok,
            "model_size_reduction": size_reduction,
            "model_size_gate": size_ok,
            "latency_reduction": latency_reduction,
            "latency_gate": latency_ok,
            "peak_memory_max_gb": peak_max,
            "peak_memory_target_gb": target_peak,
            "peak_memory_gate": peak_ok,
            "peak_vram_mean_gb": branch_aggregate["peak_vram_gb"]["mean"],
            "peak_vram_median_gb": branch_aggregate["peak_vram_gb"]["median"],
            "peak_vram_min_gb": branch_aggregate["peak_vram_gb"]["minimum"],
            "peak_vram_max_gb": branch_aggregate["peak_vram_gb"]["maximum"],
            "peak_vram_std_gb": branch_aggregate["peak_vram_gb"]["stddev"],
            "peak_vram_ci95_low_gb": branch_aggregate["peak_vram_gb"]["ci95_low"],
            "peak_vram_ci95_high_gb": branch_aggregate["peak_vram_gb"]["ci95_high"],
        }
        validated = all(
            gates[name]
            for name in (
                "generation_valid", "decode_success", "black_frame_gate", "quality_gate",
                "model_size_gate", "latency_gate", "peak_memory_gate",
            )
        )
        return M6ValidationResult(
            label=label,
            operator=str((branch.runtime_state.get("runtime_policy") or {}).get("kind", "runtime_memory")),
            parent_model_id=parent.model_id,
            branch_model_id=branch.model_id,
            target_profile_id=getattr(target, "id", None),
            repetitions=repetitions,
            reference_metrics=dict(reference_metrics),
            conditions=tuple(conditions),
            aggregates={"parent": parent_aggregate, "branch": branch_aggregate},
            gates=gates,
            validated=bool(validated),
            operator_attribution=attribution,
        )
