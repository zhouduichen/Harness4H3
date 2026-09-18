from dataclasses import replace

import pytest

from harness4h3.controller.round_policy import (
    RoundPolicy,
    RoundPolicyValidationError,
    validate_round_policy,
)


def policy_payload():
    return {
        "schema_version": 1,
        "round_id": "R0012",
        "substrate_digest": "sha256:substrate",
        "search_mode": "runtime_efficiency",
        "allowed_operators": ["step_distill", "lpl"],
        "axis_budget": {"max_trials": 2, "max_gpu_hours": 8.0},
        "objective": {"quality_floor": 0.82},
        "fixed_evaluation": {"split": "heldout", "recipe_digest": "sha256:eval"},
        "resource_policy": {"min_training_gpus": 2, "controller_overlap_gpus": 1},
        "stop_conditions": ["critical_regression", "budget_exhausted"],
        "source_observation_ids": ["obs-1"],
        "created_at": "2026-09-18T00:00:00+00:00",
    }


def test_round_policy_round_trips_and_rejects_unknown_fields():
    policy = RoundPolicy.from_dict(policy_payload())
    assert policy.to_dict()["allowed_operators"] == ["step_distill", "lpl"]
    with pytest.raises(RoundPolicyValidationError, match="unknown"):
        RoundPolicy.from_dict({**policy_payload(), "unexpected": True})


def test_round_policy_cannot_drift_substrate_or_overclaim_gpu_overlap():
    policy = RoundPolicy.from_dict(policy_payload())
    with pytest.raises(RoundPolicyValidationError, match="substrate"):
        validate_round_policy(
            policy,
            registered_operators={"step_distill", "lpl"},
            substrate_digest="sha256:other",
            evaluation_digest="sha256:eval",
            gpu_count=4,
        )
    unsafe = replace(policy, resource_policy={"min_training_gpus": 1, "controller_overlap_gpus": 4})
    with pytest.raises(RoundPolicyValidationError, match="at least 2"):
        validate_round_policy(
            unsafe,
            registered_operators={"step_distill", "lpl"},
            substrate_digest="sha256:substrate",
            evaluation_digest="sha256:eval",
            gpu_count=4,
        )


def test_round_policy_rejects_unregistered_operator_and_gpu_sum():
    policy = RoundPolicy.from_dict(policy_payload())
    with pytest.raises(RoundPolicyValidationError, match="unregistered"):
        validate_round_policy(
            policy,
            registered_operators={"step_distill"},
            substrate_digest="sha256:substrate",
            evaluation_digest="sha256:eval",
            gpu_count=4,
        )
    policy = replace(policy, resource_policy={"min_training_gpus": 3, "controller_overlap_gpus": 2})
    with pytest.raises(RoundPolicyValidationError, match="exceeds GPU count"):
        validate_round_policy(
            policy,
            registered_operators={"step_distill", "lpl"},
            substrate_digest="sha256:substrate",
            evaluation_digest="sha256:eval",
            gpu_count=4,
        )
