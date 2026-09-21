"""Immutable, bounded search policy for one remote H3 optimization round."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Tuple


_FIELDS = {
    "schema_version",
    "round_id",
    "substrate_digest",
    "search_mode",
    "allowed_operators",
    "axis_budget",
    "objective",
    "fixed_evaluation",
    "resource_policy",
    "stop_conditions",
    "source_observation_ids",
    "created_at",
}


class RoundPolicyValidationError(ValueError):
    """Raised when an LLM-produced round policy cannot be trusted."""


def normalize_round_policy_progress(policy: "RoundPolicy", raw: Any = None) -> dict[str, Any]:
    """Return a small restart-safe progress cursor for one policy round.

    The cursor is deliberately separate from the policy itself.  A policy is
    immutable input from the Controller; progress is mutable campaign state
    produced only after a candidate has completed evaluation.
    """

    if not isinstance(policy, RoundPolicy):
        raise RoundPolicyValidationError("policy must be a RoundPolicy")
    value = raw if isinstance(raw, Mapping) else {}
    if str(value.get("round_id", "")) != policy.round_id:
        return {
            "round_id": policy.round_id,
            "trials_completed": 0,
            "gpu_hours_used": 0.0,
            "completed_experiment_ids": [],
            "stop_reason": None,
        }
    try:
        trials = int(value.get("trials_completed", 0))
    except (TypeError, ValueError):
        trials = 0
    try:
        gpu_hours = float(value.get("gpu_hours_used", 0.0))
    except (TypeError, ValueError):
        gpu_hours = 0.0
    ids = value.get("completed_experiment_ids", [])
    if not isinstance(ids, (list, tuple)):
        ids = []
    completed_ids = []
    for item in ids:
        text = str(item).strip()
        if text and text not in completed_ids:
            completed_ids.append(text)
    stop_reason = value.get("stop_reason")
    stop_reason = str(stop_reason).strip() if stop_reason else None
    return {
        "round_id": policy.round_id,
        "trials_completed": max(0, trials),
        "gpu_hours_used": max(0.0, gpu_hours) if math.isfinite(gpu_hours) else 0.0,
        "completed_experiment_ids": completed_ids,
        "stop_reason": stop_reason,
    }


def round_policy_budget_status(policy: "RoundPolicy", progress: Any = None) -> str | None:
    """Return the hard stop reason when a round policy has no budget left."""

    cursor = normalize_round_policy_progress(policy, progress)
    if cursor["stop_reason"]:
        return str(cursor["stop_reason"])
    max_trials = int(policy.axis_budget["max_trials"])
    if int(cursor["trials_completed"]) >= max_trials:
        return "budget_exhausted"
    max_gpu_hours = float(policy.axis_budget["max_gpu_hours"])
    used_gpu_hours = float(cursor["gpu_hours_used"])
    if (max_gpu_hours <= 0.0 and used_gpu_hours > 0.0) or (
        max_gpu_hours > 0.0 and used_gpu_hours >= max_gpu_hours
    ):
        return "budget_exhausted"
    return None


def record_round_policy_trial(
    policy: "RoundPolicy",
    progress: Any,
    experiment_id: Any,
    gpu_hours: Any = 0.0,
) -> tuple[dict[str, Any], bool]:
    """Count one newly evaluated candidate exactly once.

    ``gpu_hours`` is measured worker cost when available.  The function is
    tolerant of legacy records that omit it, because trial counting remains
    authoritative even when old telemetry is incomplete.
    """

    cursor = normalize_round_policy_progress(policy, progress)
    experiment = str(experiment_id or "").strip()
    if not experiment or experiment in cursor["completed_experiment_ids"]:
        return cursor, False
    try:
        cost = float(gpu_hours)
    except (TypeError, ValueError):
        cost = 0.0
    if not math.isfinite(cost) or cost < 0.0:
        cost = 0.0
    cursor["trials_completed"] += 1
    cursor["gpu_hours_used"] += cost
    cursor["completed_experiment_ids"].append(experiment)
    return cursor, True


def mark_round_policy_stop(policy: "RoundPolicy", progress: Any, reason: str) -> dict[str, Any]:
    """Persist an explicit policy stop without changing immutable policy data."""

    cursor = normalize_round_policy_progress(policy, progress)
    text = str(reason or "").strip()
    if text:
        cursor["stop_reason"] = text
    return cursor


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RoundPolicyValidationError("%s must be a non-empty string" % field)
    return value.strip()


def _string_tuple(value: Any, field: str, *, allow_empty: bool = False) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RoundPolicyValidationError("%s must be a list of strings" % field)
    result = tuple(_non_empty_string(item, "%s item" % field) for item in value)
    if not allow_empty and not result:
        raise RoundPolicyValidationError("%s must not be empty" % field)
    if len(result) != len(set(result)):
        raise RoundPolicyValidationError("%s must not contain duplicates" % field)
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoundPolicyValidationError("%s must be an object" % field)
    return copy.deepcopy(dict(value))


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoundPolicyValidationError("%s must be an integer" % field)
    return int(value)


def _non_negative_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoundPolicyValidationError("%s must be numeric" % field)
    number = float(value)
    if number < 0:
        raise RoundPolicyValidationError("%s must be non-negative" % field)
    return number


@dataclass(frozen=True)
class RoundPolicy:
    """Controller-owned search policy whose safety fields cannot drift."""

    schema_version: int
    round_id: str
    substrate_digest: str
    search_mode: str
    allowed_operators: Tuple[str, ...]
    axis_budget: Mapping[str, Any]
    objective: Mapping[str, Any]
    fixed_evaluation: Mapping[str, Any]
    resource_policy: Mapping[str, Any]
    stop_conditions: Tuple[str, ...]
    source_observation_ids: Tuple[str, ...]
    created_at: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RoundPolicy":
        if not isinstance(raw, Mapping):
            raise RoundPolicyValidationError("round policy must be an object")
        unknown = sorted(set(raw) - _FIELDS)
        if unknown:
            raise RoundPolicyValidationError("unknown round policy field(s): %s" % ", ".join(map(str, unknown)))
        missing = sorted(_FIELDS - set(raw))
        if missing:
            raise RoundPolicyValidationError("missing round policy field(s): %s" % ", ".join(missing))

        schema_version = _integer(raw["schema_version"], "schema_version")
        if schema_version != 1:
            raise RoundPolicyValidationError("unsupported round policy schema_version: %s" % schema_version)

        axis_budget = _mapping(raw["axis_budget"], "axis_budget")
        max_trials = _integer(axis_budget.get("max_trials"), "axis_budget.max_trials")
        if max_trials <= 0:
            raise RoundPolicyValidationError("axis_budget.max_trials must be positive")
        max_gpu_hours = _non_negative_number(axis_budget.get("max_gpu_hours"), "axis_budget.max_gpu_hours")
        axis_budget["max_trials"] = max_trials
        axis_budget["max_gpu_hours"] = max_gpu_hours

        fixed_evaluation = _mapping(raw["fixed_evaluation"], "fixed_evaluation")
        _non_empty_string(fixed_evaluation.get("split"), "fixed_evaluation.split")
        _non_empty_string(fixed_evaluation.get("recipe_digest"), "fixed_evaluation.recipe_digest")

        resource_policy = _mapping(raw["resource_policy"], "resource_policy")
        min_training_gpus = _integer(
            resource_policy.get("min_training_gpus"), "resource_policy.min_training_gpus"
        )
        controller_overlap_gpus = _integer(
            resource_policy.get("controller_overlap_gpus"), "resource_policy.controller_overlap_gpus"
        )
        if min_training_gpus < 2:
            raise RoundPolicyValidationError("resource_policy.min_training_gpus must be at least 2")
        if controller_overlap_gpus < 0:
            raise RoundPolicyValidationError("resource_policy.controller_overlap_gpus must be non-negative")
        resource_policy["min_training_gpus"] = min_training_gpus
        resource_policy["controller_overlap_gpus"] = controller_overlap_gpus

        return cls(
            schema_version=schema_version,
            round_id=_non_empty_string(raw["round_id"], "round_id"),
            substrate_digest=_non_empty_string(raw["substrate_digest"], "substrate_digest"),
            search_mode=_non_empty_string(raw["search_mode"], "search_mode"),
            allowed_operators=_string_tuple(raw["allowed_operators"], "allowed_operators"),
            axis_budget=axis_budget,
            objective=_mapping(raw["objective"], "objective"),
            fixed_evaluation=fixed_evaluation,
            resource_policy=resource_policy,
            stop_conditions=_string_tuple(raw["stop_conditions"], "stop_conditions"),
            source_observation_ids=_string_tuple(
                raw["source_observation_ids"], "source_observation_ids", allow_empty=True
            ),
            created_at=_non_empty_string(raw["created_at"], "created_at"),
        )

    def to_dict(self) -> Mapping[str, Any]:
        value = asdict(self)
        value["allowed_operators"] = list(self.allowed_operators)
        value["stop_conditions"] = list(self.stop_conditions)
        value["source_observation_ids"] = list(self.source_observation_ids)
        return copy.deepcopy(value)


def validate_round_policy(
    policy: RoundPolicy,
    *,
    registered_operators: Iterable[str],
    substrate_digest: str,
    evaluation_digest: str,
    gpu_count: int,
) -> None:
    """Validate policy fields against immutable runtime facts."""

    if not isinstance(policy, RoundPolicy):
        raise RoundPolicyValidationError("policy must be a RoundPolicy")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count <= 0:
        raise RoundPolicyValidationError("gpu_count must be a positive integer")
    if policy.substrate_digest != _non_empty_string(substrate_digest, "substrate_digest"):
        raise RoundPolicyValidationError("round policy substrate digest does not match runtime substrate")
    if policy.fixed_evaluation.get("recipe_digest") != _non_empty_string(
        evaluation_digest, "evaluation_digest"
    ):
        raise RoundPolicyValidationError("round policy evaluation digest does not match fixed evaluator")

    registered = {_non_empty_string(name, "registered operator") for name in registered_operators}
    missing = sorted(set(policy.allowed_operators) - registered)
    if missing:
        raise RoundPolicyValidationError("round policy contains unregistered operator(s): %s" % ", ".join(missing))

    minimum = _integer(policy.resource_policy.get("min_training_gpus"), "resource_policy.min_training_gpus")
    overlap = _integer(
        policy.resource_policy.get("controller_overlap_gpus"), "resource_policy.controller_overlap_gpus"
    )
    if minimum < 2:
        raise RoundPolicyValidationError("resource_policy.min_training_gpus must be at least 2")
    if overlap < 0:
        raise RoundPolicyValidationError("resource_policy.controller_overlap_gpus must be non-negative")
    if overlap >= gpu_count:
        raise RoundPolicyValidationError("resource_policy.controller_overlap_gpus exceeds GPU count")
    if minimum + overlap > gpu_count:
        raise RoundPolicyValidationError("round policy GPU allocation exceeds GPU count")


__all__ = [
    "RoundPolicy",
    "RoundPolicyValidationError",
    "mark_round_policy_stop",
    "normalize_round_policy_progress",
    "record_round_policy_trial",
    "round_policy_budget_status",
    "validate_round_policy",
]
