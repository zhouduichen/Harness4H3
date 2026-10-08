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
