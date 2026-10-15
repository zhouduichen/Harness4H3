"""Evidence-backed design records for recursive compression algorithms."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..adapters.wan.contracts import stable_digest


DESIGN_SCHEMA_VERSION = 1


def feedback_summary(history: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Extract bounded, explicit failure feedback without inventing outcomes."""

    values: list[str] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        decision = item.get("decision")
        if isinstance(decision, Mapping):
            promotion = decision.get("promotion")
            if isinstance(promotion, Mapping):
                reasons = promotion.get("reasons")
                if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)):
                    values.extend(str(reason) for reason in reasons if str(reason).strip())
            for key in ("failure_type", "failure_reason"):
                if decision.get(key):
                    values.append(str(decision[key]))
        evaluation = item.get("evaluation")
        if isinstance(evaluation, Mapping) and evaluation.get("failure_type"):
            values.append(str(evaluation["failure_type"]))
        if item.get("failure_reason"):
            values.append(str(item["failure_reason"]))
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique[:16])


def _available_records(profile: Mapping[str, Any], key: str, field: str) -> list[Mapping[str, Any]]:
    records = profile.get(key)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    return [
        item
        for item in records
        if isinstance(item, Mapping) and item.get(field) is not None
    ]


def _bottleneck(profile: Mapping[str, Any]) -> str:
    layers = _available_records(profile, "layers", "latency_ms")
    if any(isinstance(item.get("latency_ms"), (int, float)) and float(item["latency_ms"]) > 0.0 for item in layers):
        return "latency"
    flops = _available_records(profile, "layers", "estimated_flops")
    if any(isinstance(item.get("estimated_flops"), (int, float)) and float(item["estimated_flops"]) > 0.0 for item in flops):
        return "compute"
    channels = _available_records(profile, "channel_sensitivity", "available")
    if channels:
        return "ffn_sensitivity"
    heads = _available_records(profile, "head_sensitivity", "available")
    if heads:
        return "attention_head_sensitivity"
    tokens = _available_records(profile, "token_sensitivity", "available")
    if tokens:
        return "token_sensitivity"
    blocks = _available_records(profile, "block_sensitivity", "importance_score")
    return "block_sensitivity" if blocks else "unknown"


def _evidence_refs(profile: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    layers = profile.get("layers")
    if isinstance(layers, Sequence) and not isinstance(layers, (str, bytes)):
        if any(isinstance(item, Mapping) and item.get("latency_ms") is not None for item in layers):
            refs.append("profile.layers.latency_ms")
        if any(isinstance(item, Mapping) and item.get("estimated_flops") is not None for item in layers):
            refs.append("profile.layers.estimated_flops")
    block_sensitivity = profile.get("block_sensitivity")
    if isinstance(block_sensitivity, Sequence) and not isinstance(block_sensitivity, (str, bytes)):
        if any(isinstance(item, Mapping) and item.get("importance_score") is not None for item in block_sensitivity):
            refs.append("profile.block_sensitivity.importance_score")
    channel_sensitivity = profile.get("channel_sensitivity")
    if isinstance(channel_sensitivity, Sequence) and not isinstance(channel_sensitivity, (str, bytes)):
        if any(isinstance(item, Mapping) and item.get("available") is True for item in channel_sensitivity):
            refs.append("profile.channel_sensitivity")
    timestep_sensitivity = profile.get("timestep_sensitivity")
    if isinstance(timestep_sensitivity, Mapping) and timestep_sensitivity:
        refs.append("profile.timestep_sensitivity")
    for key in ("head_sensitivity", "attention_sensitivity", "token_sensitivity"):
        values = profile.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and any(isinstance(item, Mapping) and item.get("available") is True for item in values):
            refs.append("profile.%s" % key)
    sampling = profile.get("sampling_step_sensitivity")
    if isinstance(sampling, Mapping) and sampling.get("available") is True:
        refs.append("profile.sampling_step_sensitivity")
    return refs


def design_fingerprint(record_without_fingerprint: Mapping[str, Any]) -> str:
    """Return a stable digest for the design decision, not its file location."""

    return stable_digest(dict(record_without_fingerprint))


def build_design_record(
    *,
    algorithm_id: str,
    parent_algorithm_id: str | None,
    search_choice: str,
    profile: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    hypothesis: str,
    change_summary: str = "",
    change_rationale: str = "",
    source_sha256: str | None = None,
    parent_source_sha256: str | None = None,
    change_taxonomy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable evidence context for one executable algorithm."""

    history_ids = [
        str(item.get("experience_id", item.get("round_id")))
        for item in history
        if isinstance(item, Mapping) and (item.get("experience_id") or item.get("round_id"))
    ][:16]
    feedback = feedback_summary(history)
    strategy = "composed" if str(plan.get("operator", "")) == "compose" else str(plan.get("operator", "unknown"))
    body: dict[str, Any] = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "algorithm_id": str(algorithm_id),
        "parent_algorithm_id": parent_algorithm_id,
        "search_choice": str(search_choice),
        "strategy": strategy,
        "bottleneck": _bottleneck(profile),
        "evidence_refs": _evidence_refs(profile),
        "history_ids": history_ids,
        "feedback": list(feedback),
        "hypothesis": str(hypothesis),
        "change_summary": str(change_summary),
        "change_rationale": str(change_rationale),
        "source_sha256": source_sha256,
        "parent_source_sha256": parent_source_sha256,
        "change_taxonomy": dict(change_taxonomy or {}),
        "plan": dict(plan),
    }
    return {**body, "fingerprint": design_fingerprint(body)}
