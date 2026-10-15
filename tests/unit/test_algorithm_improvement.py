from __future__ import annotations

import pytest

from harness4h3.evolution.agent import AlgorithmAgent
from harness4h3.evolution.candidate import AlgorithmCandidate
from harness4h3.evolution.improvement import AlgorithmValidator
from harness4h3.evolution.store import AlgorithmStore
from harness4h3.sandbox import AlgorithmSandbox


def fake_profile():
    return {
        "model": {"architecture": "WanFakeDiT", "block_count": 4, "ffn_width": 16},
        "layers": [
            {"block_id": "block.0", "estimated_flops": 10.0, "latency_ms": 1.0},
            {"block_id": "block.1", "estimated_flops": 40.0, "latency_ms": 4.0},
            {"block_id": "block.2", "estimated_flops": 20.0, "latency_ms": 2.0},
            {"block_id": "block.3", "estimated_flops": 30.0, "latency_ms": 3.0},
        ],
        "block_sensitivity": [
            {"block_id": "block.%d" % index, "importance_score": 0.1 + index * 0.1, "available": True}
            for index in range(4)
        ],
        "channel_sensitivity": [
            {
                "block_id": "block.0",
                "width": 16,
                "available": True,
                "channels": [{"channel": index, "importance_score": float(index + 1), "available": True} for index in range(16)],
            }
        ],
    }


def parent_source():
    return "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n"


def quality_failure():
    return {"experience_id": "R0001", "status": "rejected", "decision": {"promotion": {"reasons": ["quality_floor_failed"]}}}


def test_deterministic_improvement_changes_parent_source_and_records_reason(tmp_path):
    parent = AlgorithmCandidate.from_source("A0001", parent_source(), hypothesis="baseline")
    child = AlgorithmAgent().improve(
        "A0002",
        parent=parent,
        parent_id="M0000",
        profile=fake_profile(),
        history=[quality_failure()],
        failures=["quality_floor_failed"],
        search_space={"version_id": "S0001"},
    )
    assert child.parent_algorithm_id == parent.algorithm_id
    assert child.parent_source_sha256 == parent.source_sha256
    assert child.metadata["parent_source_sha256"] == parent.source_sha256
    assert child.source != parent.source
    assert child.change_rationale
    result = AlgorithmSandbox(tmp_path / "sandbox").run(child.source, fake_profile(), [])
    assert result.status == "success"


def test_algorithm_store_rejects_wrong_parent_hash(tmp_path):
    store = AlgorithmStore(tmp_path)
    parent = store.create(AlgorithmCandidate.from_source("A0001", parent_source()))
    child = AlgorithmAgent().improve(
        "A0002", parent=parent, parent_id="M0000", profile=fake_profile(), history=[], failures=["efficiency_plateau"]
    )
    with pytest.raises(ValueError, match="parent_source_hash_mismatch"):
        store.create_child(child, "0" * 64)


def test_algorithm_validator_compares_parent_and_child_under_same_conditions(tmp_path):
    parent = AlgorithmCandidate.from_source("A0001", parent_source())
    child = AlgorithmAgent().improve(
        "A0002", parent=parent, parent_id="M0000", profile=fake_profile(), history=[], failures=["efficiency_plateau"]
    )
    sandbox = AlgorithmSandbox(tmp_path / "sandbox")

    def evaluate(result):
        return {
            "valid": True,
            "offline_simulation": True,
            "search_score": 0.8 if result.plan.operator == "prune_blocks" else 0.9,
            "model_size_gb": 1.0 if result.plan.operator == "prune_blocks" else 0.7,
        }

    record = AlgorithmValidator().validate(parent, child, sandbox, fake_profile(), [], evaluate, budget={"steps": 1})
    assert record.valid is True
    assert record.evidence_level == "cpu_fake"
    assert record.parent_source_sha256 == parent.source_sha256
    assert record.candidate_plan["operator"] == "prune_ffn_channels"


def test_agent_can_propose_untrusted_search_space_extension():
    proposal = AlgorithmAgent().propose_search_space_change(
        search_space={"version_id": "S0001", "operators": [{"name": "prune_blocks"}]},
        history=[{"round_id": "R0001", "status": "rejected"}],
        failures=["channel_importance_plateau"],
    )
    assert proposal["spec"]["name"] == "prune_ffn_by_weight_threshold"
    assert "not_real_wan_validated" in proposal["spec"]["limitations"]
