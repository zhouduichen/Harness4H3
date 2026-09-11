from dataclasses import replace

from research.experiments.m6_campaign import (
    build_research_report,
    classify_outcome,
    experiment_fingerprint,
    state_digest,
    validate_novelty,
)
from harness4h3.h3.state import ModelState


def _state(policy=None):
    state = ModelState.fake_baseline("M0001")
    return replace(state, architecture_name="MiniMax-H3", runtime_state=policy or {})


def test_fingerprint_is_order_independent_but_changes_with_parent_or_args():
    first = experiment_fingerprint(
        "runtime_offload", {"mode": "aggressive", "x": 1}, _state(), "rtx"
    )
    second = experiment_fingerprint(
        "runtime_offload", {"x": 1, "mode": "aggressive"}, _state(), "rtx"
    )
    changed_parent = experiment_fingerprint(
        "runtime_offload",
        {"mode": "aggressive", "x": 1},
        _state({"runtime_policy": {"kind": "cache_release"}}),
        "rtx",
    )
    changed_args = experiment_fingerprint(
        "runtime_offload", {"mode": "balanced", "x": 1}, _state(), "rtx"
    )
    assert first == second
    assert first != changed_parent
    assert first != changed_args
    assert len(state_digest(_state())) == 64


def test_same_operator_with_new_args_requires_prior_experiment_reference():
    recent = [{"experiment_id": "exp_0001", "operator": "vae_tiling", "operator_args": {"tile_size": 256}}]
    assert (
        validate_novelty(
            "vae_tiling",
            "try exp_0001 with lower temporary activations",
            "new",
            set(),
            recent,
        )
        is None
    )
    assert (
        validate_novelty("vae_tiling", "try another tiling", "newer", set(), recent)
        == "same_operator_requires_prior_evidence"
    )
    assert validate_novelty("runtime_offload", "new offload hypothesis", "new", set(), recent) is None
    assert (
        validate_novelty("vae_tiling", "new", "old", {"old"}, recent)
        == "duplicate_experiment_fingerprint"
    )


def test_rejection_does_not_consume_failure_budget_classification():
    assert classify_outcome(True, True, {"dev": {"validated": False}}, {}, False) == "rejected_candidate"
    assert (
        classify_outcome(
            True,
            True,
            {"dev": {"validated": True}},
            {"heldout": {"failure_type": "timeout"}},
            False,
        )
        == "failed_experiment"
    )
    assert classify_outcome(False, False, {}, {}, False) == "failed_experiment"
    assert (
        classify_outcome(
            True,
            True,
            {"dev": {"validated": True}, "heldout": {"validated": True}},
            {},
            True,
        )
        == "accepted_candidate"
    )


def test_research_report_splits_rejection_from_failure_and_records_zero_intervention():
    payload = {
        "harness": {"version": "Harness4H3-v1.0"},
        "target_profile_id": "rtx5080_h3_v1",
        "status": "experiment_budget_exhausted",
        "iterations": [
            {"experiment_id": "exp_0001", "outcome": "rejected_candidate", "system_parent_id": "C0000"},
            {
                "experiment_id": "exp_0002",
                "outcome": "failed_experiment",
                "failure_type": "duplicate_experiment_fingerprint",
                "operator_executed": False,
            },
        ],
        "system_candidates": [{"id": "C0000", "status": "baseline"}],
        "duplicate_fingerprints_blocked": 1,
    }
    report = build_research_report(payload, "start", "end")
    assert report["failed_experiment_count"] == 1
    assert report["rejected_candidate_count"] == 1
    assert report["human_intervention_count"] == 0
    assert report["repeated_failure_avoidance"]["rejected_parent_reuse"] is False
