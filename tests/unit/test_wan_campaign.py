from pathlib import Path

import pytest

from harness4h3.student.wan_campaign import (
    WanProposalError,
    WanTrainingProposal,
    candidate_decision,
    sha256_file,
)


def _proposal(parent: str, **overrides):
    value = {
        "schema_version": 1,
        "proposal_id": "wan_0001",
        "parent_checkpoint_sha256": parent,
        "steps": 8,
        "teacher_reuse_steps": 2,
        "learning_rate": 5e-7,
        "focus": "semantic",
        "hypothesis": "Increase semantic alignment while preserving continuation stability.",
    }
    value.update(overrides)
    return value


def _v2_proposal(parent: str, *, method: str = "progressive_distillation", **overrides):
    value = {
        "schema_version": 2,
        "proposal_id": "wan_0002",
        "parent_checkpoint_sha256": parent,
        "training": {
            "method": method,
            "source_steps": 32,
            "target_steps": 8,
            "train_steps": 256,
            "learning_rate": 1e-4,
            "critic_learning_rate": 1e-4,
            "velocity_weight": 1.0,
            "clean_anchor_weight": 0.5,
            "rollout_steps": 4,
            "rollout_every": 2,
            "rollout_endpoint_weight": 1.0,
            "rollout_clean_weight": 0.5,
            "trainable_scope": "all",
            "fp32_optimizer_state": False,
        },
        "initialization": {"mode": "inherit"},
        "precision": {"loss_reduction": "fp32", "optimizer_state": "bf16"},
        "curriculum": {"fidelity_schedule": ["F1", "F2", "F3"]},
        "data": {"manifest_digest": "c" * 64, "prompt_policy": "balanced", "seed_policy": "fixed"},
        "continuation": {"overlap_latent_frames": 7, "anchor_schedule": "linear"},
        "focus": "semantic",
        "hypothesis": "Use shared progressive rollout to improve semantic alignment.",
    }
    for key, value_override in overrides.items():
        if key in value["training"]:
            value["training"][key] = value_override
        else:
            value[key] = value_override
    return value


def _progressive_capabilities():
    return {"capabilities": [{"name": "progressive_distillation", "available": True}]}


def test_wan_proposal_is_strict_and_binds_to_parent_digest():
    parent = "a" * 64
    proposal = WanTrainingProposal.from_mapping(_proposal(parent), expected_parent_sha256=parent)
    assert proposal.steps == 8
    assert proposal.teacher_reuse_steps == 2
    assert proposal.to_dict()["parent_checkpoint_sha256"] == parent

    with pytest.raises(WanProposalError, match="unknown"):
        WanTrainingProposal.from_mapping(_proposal(parent, shell="rm -rf /"), expected_parent_sha256=parent)
    with pytest.raises(WanProposalError, match="parent_checkpoint_sha256"):
        WanTrainingProposal.from_mapping(_proposal("b" * 64), expected_parent_sha256=parent)


def test_wan_proposal_rejects_unsafe_training_ranges():
    parent = "a" * 64
    with pytest.raises(WanProposalError, match="steps"):
        WanTrainingProposal.from_mapping(_proposal(parent, steps=0), expected_parent_sha256=parent)
    with pytest.raises(WanProposalError, match="teacher_reuse_steps"):
        WanTrainingProposal.from_mapping(_proposal(parent, teacher_reuse_steps=9), expected_parent_sha256=parent)
    with pytest.raises(WanProposalError, match="learning_rate"):
        WanTrainingProposal.from_mapping(_proposal(parent, learning_rate=0.1), expected_parent_sha256=parent)


def test_candidate_selection_requires_contract_and_semantic_improvement():
    parent = {
        "status": "success",
        "valid": True,
        "promotable": True,
        "offline_simulation": False,
        "continuation_verified": True,
        "quality_score": 0.65,
        "quality_metrics": {"semantic": 0.61},
    }
    better = dict(parent, quality_score=0.67, quality_metrics={"semantic": 0.63})
    decision = candidate_decision(
        parent,
        better,
        teacher_semantic=0.635,
        semantic_floor_ratio=0.95,
        quality_floor_ratio=0.90,
        min_semantic_delta=0.005,
    )
    assert decision["accepted"] is True
    assert decision["reason"] == "semantic_improved"

    broken = dict(better, continuation_verified=False)
    rejected = candidate_decision(
        parent,
        broken,
        teacher_semantic=0.635,
        semantic_floor_ratio=0.95,
        quality_floor_ratio=0.90,
        min_semantic_delta=0.005,
    )
    assert rejected["accepted"] is False
    assert rejected["reason"] == "contract_invalid"


def test_sha256_file_is_stable(tmp_path: Path):
    path = tmp_path / "checkpoint.safetensors"
    path.write_bytes(b"checkpoint")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_v2_proposal_normalizes_nested_training_policy():
    parent = "a" * 64
    proposal = WanTrainingProposal.from_mapping(
        _v2_proposal(parent), expected_parent_sha256=parent, capabilities=_progressive_capabilities()
    )
    assert proposal.schema_version == 2
    assert proposal.training.method == "progressive_distillation"
    assert proposal.training.rollout_steps == 4
    assert proposal.continuation.overlap_latent_frames == 7
    assert len(proposal.fingerprint()) == 64


def test_v1_proposal_remains_readable_as_flat_policy():
    parent = "a" * 64
    proposal = WanTrainingProposal.from_mapping(_proposal(parent), expected_parent_sha256=parent)
    assert proposal.schema_version == 1
    assert proposal.training.method == "progressive_distillation"
    assert proposal.steps == 8
    assert proposal.teacher_reuse_steps == 2


def test_v2_rejects_unavailable_dmd2_and_unknown_shell():
    parent = "a" * 64
    with pytest.raises(WanProposalError, match="capability_unavailable"):
        WanTrainingProposal.from_mapping(
            _v2_proposal(parent, method="dmd2"),
            expected_parent_sha256=parent,
            capabilities=_progressive_capabilities(),
        )
    with pytest.raises(WanProposalError, match="unknown"):
        WanTrainingProposal.from_mapping(
            _v2_proposal(parent, shell="python evil.py"),
            expected_parent_sha256=parent,
            capabilities=_progressive_capabilities(),
        )
