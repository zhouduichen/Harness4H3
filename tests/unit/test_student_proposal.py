from __future__ import annotations

import copy

import pytest

from harness4h3.student.proposal import ProposalValidationError, StudentProposal, StudentTarget


def valid_payload(*, hidden_size: int = 2048, depth: int = 24):
    temporal_layers = list(range(0, depth, max(1, depth // 6)))[:6]
    return {
        "schema_version": 1,
        "proposal_id": "student_0001",
        "parent_proposal_id": None,
        "teacher": {"checkpoint": "/data/teacher.safetensors", "adapter": "minimax_h3"},
        "architecture": {
            "family": "video_latent_dit",
            "latent_channels": 24,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": 32 if hidden_size % 32 == 0 else 16,
            "mlp_ratio": 4.0,
            "spatial_patch": 2,
            "temporal_patch": 1,
            "temporal_layers": temporal_layers,
            "conditioning": "ada_norm_zero",
            "norm": "rmsnorm",
            "activation": "silu",
        },
        "training": {
            "method": "dmd2",
            "source_steps": 32,
            "target_steps": 8,
            "max_steps": 1000,
            "learning_rate": 1e-6,
            "critic_learning_rate": 1e-6,
            "batch_size": 1,
        },
        "deployment": {"precision": "bf16", "quantization": "int8"},
    }


def test_two_different_proposals_are_valid_and_digest_stable():
    first = StudentProposal.from_dict(valid_payload(hidden_size=2048, depth=24))
    second = StudentProposal.from_dict(valid_payload(hidden_size=1792, depth=32))
    target = StudentTarget(min_params=1_000_000_000, max_params=2_000_000_000)
    first_report = first.validate(target)
    second_report = second.validate(target)
    assert first_report.errors == ()
    assert second_report.errors == ()
    assert first.digest != second.digest
    assert StudentProposal.from_dict(first.to_dict()).digest == first.digest


def test_unknown_fields_and_invalid_head_divisibility_are_rejected():
    payload = valid_payload(hidden_size=2000, depth=24)
    payload["architecture"]["unknown"] = 1
    with pytest.raises(ProposalValidationError, match="unknown"):
        StudentProposal.from_dict(payload)
    payload = valid_payload(hidden_size=2048, depth=24)
    payload["architecture"]["num_heads"] = 30
    assert any("divisible" in error for error in StudentProposal.from_dict(payload).validate(StudentTarget()).errors)


def test_training_and_deployment_bounds_are_strict():
    payload = valid_payload()
    payload["training"]["learning_rate"] = float("nan")
    with pytest.raises(ProposalValidationError, match="finite"):
        StudentProposal.from_dict(payload)


def test_round_trip_does_not_mutate_input():
    payload = valid_payload()
    original = copy.deepcopy(payload)
    proposal = StudentProposal.from_dict(payload)
    expected = copy.deepcopy(original)
    expected["training"].pop("max_steps")
    expected["training"].pop("batch_size")
    assert proposal.to_dict() == expected
    assert payload == original
