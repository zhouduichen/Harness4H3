"""Validation helpers for persistent runtime timestep gating."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TimestepGateRule:
    block_id: str
    min_timestep: float
    max_timestep: float

    def __post_init__(self) -> None:
        if not str(self.block_id).strip():
            raise ValueError("timestep_gate_block_id_required")
        for value in (self.min_timestep, self.max_timestep):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError("timestep_gate_bounds_must_be_finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("timestep_gate_bounds_must_be_normalized")
        if float(self.min_timestep) > float(self.max_timestep):
            raise ValueError("timestep_gate_min_exceeds_max")

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": str(self.block_id),
            "min_timestep": float(self.min_timestep),
            "max_timestep": float(self.max_timestep),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TimestepGateRule":
        if not isinstance(raw, Mapping) or set(raw) != {"block_id", "min_timestep", "max_timestep"}:
            raise ValueError("timestep_gate_rule_schema_invalid")
        return cls(str(raw["block_id"]), float(raw["min_timestep"]), float(raw["max_timestep"]))


@dataclass(frozen=True)
class TimestepAttentionRule:
    """Per-timestep self-attention K/V budget.

    The rule deliberately changes the actual K/V sequence passed to the
    attention implementation.  It is separate from block gating so a
    timestep policy can be composed with a structural/runtime block policy
    without turning a skipped block into a misleading token statistic.
    """

    min_timestep: float
    max_timestep: float
    keep_ratio: float
    block_ids: tuple[str, ...] = ()
    score: str = "l2_norm"

    def __post_init__(self) -> None:
        for value in (self.min_timestep, self.max_timestep):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("timestep_attention_bounds_must_be_finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("timestep_attention_bounds_must_be_normalized")
        if float(self.min_timestep) > float(self.max_timestep):
            raise ValueError("timestep_attention_min_exceeds_max")
        if isinstance(self.keep_ratio, bool) or not isinstance(self.keep_ratio, (int, float)) or not math.isfinite(float(self.keep_ratio)):
            raise ValueError("timestep_attention_keep_ratio_invalid")
        if not 0.0 < float(self.keep_ratio) <= 1.0:
            raise ValueError("timestep_attention_keep_ratio_invalid")
        if not isinstance(self.block_ids, tuple) or not all(isinstance(item, str) and item for item in self.block_ids):
            raise ValueError("timestep_attention_block_ids_invalid")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("timestep_attention_duplicate_block")
        if self.score not in {"l2_norm", "mean_abs"}:
            raise ValueError("timestep_attention_score_invalid")

    def applies_to(self, block_id: str) -> bool:
        return not self.block_ids or str(block_id) in self.block_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_timestep": float(self.min_timestep),
            "max_timestep": float(self.max_timestep),
            "keep_ratio": float(self.keep_ratio),
            "block_ids": list(self.block_ids),
            "score": str(self.score),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TimestepAttentionRule":
        if not isinstance(raw, Mapping):
            raise ValueError("timestep_attention_rule_schema_invalid")
        allowed = {"min_timestep", "max_timestep", "keep_ratio", "block_ids", "score"}
        if set(raw) - allowed or not {"min_timestep", "max_timestep", "keep_ratio"} <= set(raw):
            raise ValueError("timestep_attention_rule_schema_invalid")
        block_ids = raw.get("block_ids", [])
        if not isinstance(block_ids, (list, tuple)) or not all(isinstance(item, str) for item in block_ids):
            raise ValueError("timestep_attention_block_ids_invalid")
        return cls(
            float(raw["min_timestep"]),
            float(raw["max_timestep"]),
            float(raw["keep_ratio"]),
            tuple(str(item) for item in block_ids),
            str(raw.get("score", "l2_norm")),
        )


def validate_timestep_gate_plan(plan: Mapping[str, Any], block_ids: Sequence[str]) -> tuple[TimestepGateRule, ...]:
    if not isinstance(plan, Mapping) or set(plan) != {"rules", "policy"}:
        raise ValueError("timestep_gate_plan_schema_invalid")
    if plan.get("policy") != "skip_when_in_range":
        raise ValueError("timestep_gate_policy_invalid")
    raw_rules = plan.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("timestep_gate_rules_required")
    known = {str(item) for item in block_ids}
    rules = tuple(TimestepGateRule.from_mapping(item) for item in raw_rules)
    if any(rule.block_id not in known for rule in rules):
        raise ValueError("timestep_gate_unknown_block")
    if len({rule.block_id for rule in rules}) != len(rules):
        raise ValueError("timestep_gate_duplicate_block")
    return rules


def validate_timestep_attention_plan(plan: Mapping[str, Any], block_ids: Sequence[str]) -> tuple[TimestepAttentionRule, ...]:
    if not isinstance(plan, Mapping) or set(plan) != {"rules", "policy"}:
        raise ValueError("timestep_attention_plan_schema_invalid")
    if plan.get("policy") != "select_kv_tokens":
        raise ValueError("timestep_attention_policy_invalid")
    raw_rules = plan.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("timestep_attention_rules_required")
    known = {str(item) for item in block_ids}
    rules = tuple(TimestepAttentionRule.from_mapping(item) for item in raw_rules)
    if any(block_id not in known for rule in rules for block_id in rule.block_ids):
        raise ValueError("timestep_attention_unknown_block")
    for left_index, left in enumerate(rules):
        for right in rules[left_index + 1:]:
            shared_blocks = not left.block_ids or not right.block_ids or bool(set(left.block_ids) & set(right.block_ids))
            if shared_blocks and max(float(left.min_timestep), float(right.min_timestep)) <= min(float(left.max_timestep), float(right.max_timestep)):
                raise ValueError("timestep_attention_overlapping_rules")
    return rules


def normalized_timestep(value: Any) -> float:
    """Convert scalar/tensor-like timestep values to the [0, 1] convention."""

    raw = value
    if hasattr(raw, "detach"):
        raw = raw.detach().float().reshape(-1)[0].item()
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError("timestep_gate_runtime_timestep_invalid")
    number = float(raw)
    return number / 1000.0 if number > 1.0 else number


def normalized_timesteps(value: Any, batch_size: int) -> tuple[float, ...]:
    """Normalize one timestep per batch element without silently taking item 0."""

    size = int(batch_size)
    if size <= 0:
        raise ValueError("timestep_gate_batch_size_invalid")
    raw = value
    if hasattr(raw, "detach"):
        raw = raw.detach().float().reshape(-1).tolist()
    elif isinstance(raw, (list, tuple)):
        raw = list(raw)
    else:
        raw = [raw]
    if len(raw) == 1:
        raw = raw * size
    if len(raw) != size:
        raise ValueError("timestep_gate_batch_shape_invalid")
    values = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError("timestep_gate_runtime_timestep_invalid")
        number = float(item)
        values.append(number / 1000.0 if number > 1.0 else number)
    return tuple(values)


def rule_matches(rule: TimestepGateRule, timestep: Any) -> bool:
    value = normalized_timestep(timestep)
    return float(rule.min_timestep) <= value <= float(rule.max_timestep)


def rules_to_dict(rules: Sequence[TimestepGateRule]) -> list[Mapping[str, Any]]:
    return [dict(asdict(rule)) for rule in rules]


__all__ = [
    "TimestepAttentionRule",
    "TimestepGateRule",
    "normalized_timestep",
    "normalized_timesteps",
    "rule_matches",
    "rules_to_dict",
    "validate_timestep_attention_plan",
    "validate_timestep_gate_plan",
]
