"""Budget-aware ordering and evidence scoring for algorithm search."""

from __future__ import annotations

import math

from ..adapters.wan.contracts import stable_digest

SEARCH_RANKING_VERSION = "evidence-weighted-multifamily-v2"
from typing import Any, Mapping, Sequence


def candidate_family(plan: Mapping[str, Any], operator_specs: Mapping[str, Mapping[str, Any]]) -> str:
    """Map plans to coverage families used by the bounded evaluator."""

    name = str(plan.get("operator", ""))
    if name == "compose":
        return "hybrid"
    spec = operator_specs.get(name, {})
    granularity = str(spec.get("granularity", ""))
    if name == "sampling_step_skip" or granularity == "sampling_step":
        return "sampling"
    if name in {"timestep_block_gate", "timestep_attention_policy"} or "timestep" in granularity:
        return "timestep"
    if name in {"prune_attention_heads", "prune_heads", "head_gating"} or "head" in granularity:
        return "head"
    if name in {"attention_window", "sparse_runtime", "svoo_attention"} or granularity in {"attention", "sparse_runtime", "sparse_attention"}:
        return "attention"
    if name == "magcache" or granularity == "timestep_cache":
        return "cache"
    if name == "kv_token_select":
        return "kv"
    if name in {"prune_tokens", "token_merge"} or granularity == "token":
        return "token"
    if name == "prune_ffn_channels" or granularity == "channel":
        return "channel"
    if name in {"prune_blocks", "prune_layers"} or granularity in {"block", "layer"}:
        return "structure"
    return granularity or name or "other"


def budgeted_candidate_order(
    plans: Sequence[Mapping[str, Any]],
    rank: Sequence[int],
    operator_specs: Mapping[str, Mapping[str, Any]],
    budget: int,
    *,
    rotation: int = 0,
    family_counts: Mapping[str, int] | None = None,
    family_utilities: Mapping[str, float] | None = None,
) -> tuple[int, ...]:
    """Allocate the next slots to the least explored available family."""

    limit = max(0, int(budget))
    buckets: dict[str, list[int]] = {}
    family_order: list[str] = []
    seen: set[int] = set()
    for index in rank:
        if index < 0 or index >= len(plans) or index in seen:
            continue
        seen.add(index)
        family = candidate_family(plans[index], operator_specs)
        if family not in buckets:
            buckets[family] = []
            family_order.append(family)
        buckets[family].append(index)

    counts = {str(key): max(0, int(value)) for key, value in (family_counts or {}).items()}
    utilities = {
        str(key): float(value)
        for key, value in (family_utilities or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    }
    ordered: list[int] = []
    offsets = {family: 0 for family in family_order}
    for family in family_order:
        if len(ordered) >= limit:
            break
        if counts.get(family, 0) == 0 and buckets[family]:
            ordered.append(buckets[family][0])
            offsets[family] = 1
            counts[family] = 1
    while len(ordered) < min(limit, len(seen)):
        available = [family for family in family_order if offsets[family] < len(buckets[family])]
        if not available:
            break
        if any(name in utilities for name in available):
            family = min(available, key=lambda name: (-utilities.get(name, float("-inf")) / (1.0 + counts.get(name, 0)), counts.get(name, 0), family_order.index(name)))
        else:
            family = min(available, key=lambda name: (counts.get(name, 0), family_order.index(name)))
        ordered.append(buckets[family][offsets[family]])
        offsets[family] += 1
        counts[family] = counts.get(family, 0) + 1
    return tuple(ordered)


def efficiency_reduction_ratio(metrics: Mapping[str, Any]) -> float:
    """Return a bounded screening proxy, never an end-to-end speed estimate."""

    values: list[float] = []
    for key in ("parameter_reduction_ratio", "checkpoint_size_reduction_ratio"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    compute = metrics.get("compute_metrics")
    if isinstance(compute, Mapping):
        value = compute.get("reduction_ratio")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    runtime = metrics.get("runtime_execution")
    derived = runtime.get("derived") if isinstance(runtime, Mapping) else None
    if isinstance(derived, Mapping):
        for key, value in derived.items():
            if (str(key).endswith("_reduction_ratio") or str(key).endswith("_skip_ratio")) and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                values.append(float(value))
    return max(0.0, min(1.0, max(values, default=0.0)))


def measured_search_score(metrics: Mapping[str, Any]) -> float | None:
    """Score a valid probe using quality plus the measured compute/size signal."""

    if metrics.get("valid") is False:
        return None
    quality = metrics.get("quality_score")
    if not isinstance(quality, (int, float)) or isinstance(quality, bool) or not math.isfinite(float(quality)):
        return None
    return float(quality) + 0.10 * efficiency_reduction_ratio(metrics)


def measured_candidate_key(record: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    """Order valid measurements by score, quality, efficiency, then Agent rank."""

    if record.get("valid") is not True or record.get("quality_feasible", True) is not True:
        return None
    score = record.get("search_score")
    quality = record.get("quality_score")
    efficiency = record.get("efficiency_reduction_ratio")
    rank_position = record.get("rank_position", record.get("rank_index", 0))
    values = (score, quality, efficiency, rank_position)
    normalized: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            return None
        normalized.append(float(value))
    return normalized[0], normalized[1], normalized[2], -normalized[3]


def measured_pareto_key(record: Mapping[str, Any]) -> tuple[float, float, float, float, float] | None:
    """Comparable tie-break inside a Pareto front; quality remains the lead objective."""

    if record.get("valid") is not True:
        return None
    quality = record.get("quality_score")
    efficiency = record.get("efficiency_reduction_ratio")
    risk = record.get("risk_value", record.get("quality_error"))
    position = record.get("rank_position", record.get("rank_index", 0))
    if not isinstance(risk, (int, float)) or isinstance(risk, bool) or not math.isfinite(float(risk)):
        risk = 0.0
    values = (quality, efficiency, risk, position)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
        return None
    return (
        1.0 if record.get("quality_feasible") is True else 0.0,
        float(quality),
        float(efficiency),
        -float(risk),
        -float(position),
    )


def pareto_front_indices(records: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    """Return valid candidates non-dominated on quality, efficiency, and risk."""

    objectives: dict[int, tuple[float, float, float]] = {}
    for index, record in enumerate(records):
        if record.get("valid") is not True:
            continue
        quality = record.get("quality_score")
        efficiency = record.get("efficiency_reduction_ratio")
        risk = record.get("risk_value", record.get("quality_error"))
        if not isinstance(risk, (int, float)) or isinstance(risk, bool) or not math.isfinite(float(risk)):
            risk = 0.0
        values = (quality, efficiency, risk)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            continue
        objectives[index] = (float(quality), float(efficiency), -float(risk))

    front: list[int] = []
    for index, values in objectives.items():
        dominated = any(
            other != index
            and all(left >= right for left, right in zip(objectives[other], values))
            and any(left > right for left, right in zip(objectives[other], values))
            for other in objectives
        )
        if not dominated:
            front.append(index)
    return tuple(front)


def measured_recovery_key(record: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    """Rank valid near misses for choosing the next Block restore/replace step."""

    if record.get("valid") is not True:
        return None
    quality = record.get("quality_score")
    score = record.get("search_score")
    efficiency = record.get("efficiency_reduction_ratio")
    position = record.get("evaluation_position", record.get("rank_position", 0))
    values = (quality, score, efficiency, position)
    normalized: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            return None
        normalized.append(float(value))
    return normalized[0], normalized[1], normalized[2], -normalized[3]


def reusable_measured_trace(trace: Mapping[str, Any]) -> bool:
    """Never resume a legacy index fallback as a measured winner."""
    measured = trace.get("measured_search")
    selected = trace.get("selected_plan")
    if not isinstance(measured, Mapping) or not isinstance(selected, Mapping):
        return False
    if measured.get("ranking_version") != SEARCH_RANKING_VERSION or measured.get("complete") is not True:
        return False
    digest = stable_digest(selected)
    return measured.get("selected_plan_digest") == digest and any(
        isinstance(row, Mapping) and row.get("plan_digest") == digest and measured_candidate_key(row) is not None
        for row in measured.get("candidate_results", ())
    )
