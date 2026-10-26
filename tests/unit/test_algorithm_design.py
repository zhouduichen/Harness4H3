from __future__ import annotations

from harness4h3.evolution.design import build_design_record, feedback_summary


def _profile():
    return {
        "model": {"architecture": "WanFakeDiT"},
        "layers": [
            {"block_id": "block.0", "estimated_flops": 10.0, "latency_ms": 1.0},
            {"block_id": "block.1", "estimated_flops": 40.0, "latency_ms": 4.0},
        ],
        "block_sensitivity": [
            {"block_id": "block.0", "importance_score": 0.1, "available": True},
            {"block_id": "block.1", "importance_score": 0.2, "available": True},
        ],
        "channel_sensitivity": [{"block_id": "block.1", "available": True, "channels": []}],
    }


def test_design_record_is_stable_and_changes_when_plan_changes():
    common = {
        "algorithm_id": "A0001",
        "parent_algorithm_id": None,
        "search_choice": "explore",
        "profile": _profile(),
        "history": [],
        "hypothesis": "target a high-cost insensitive block",
    }
    first = build_design_record(
        **common,
        plan={"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
    )
    repeated = build_design_record(
        **common,
        plan={"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
    )
    changed = build_design_record(
        **common,
        plan={"operator": "prune_blocks", "args": {"block_ids": ["block.1"]}},
    )

    assert first["fingerprint"] == repeated["fingerprint"]
    assert first["fingerprint"] != changed["fingerprint"]
    assert "profile.layers.latency_ms" in first["evidence_refs"]
    assert "profile.block_sensitivity.importance_score" in first["evidence_refs"]
    assert first["bottleneck"] == "latency"


def test_feedback_summary_distinguishes_quality_and_efficiency_failures():
    history = [
        {
            "experience_id": "R0001",
            "status": "rejected",
            "decision": {"promotion": {"reasons": ["quality_floor_failed"]}},
        },
        {
            "experience_id": "R0002",
            "status": "rejected",
            "decision": {"promotion": {"reasons": ["no_substantial_efficiency_improvement"]}},
        },
    ]

    feedback = feedback_summary(history)

    assert "quality_floor_failed" in feedback
    assert "no_substantial_efficiency_improvement" in feedback


def test_design_record_keeps_history_ids_and_parent_lineage():
    record = build_design_record(
        algorithm_id="A0003",
        parent_algorithm_id="A0002",
        search_choice="exploit",
        profile={"model": {"architecture": "WanFakeDiT"}, "layers": []},
        history=[{"experience_id": "R0002", "status": "accepted"}],
        plan={"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
        hypothesis="reuse the last accepted structure",
    )

    assert record["parent_algorithm_id"] == "A0002"
    assert record["history_ids"] == ["R0002"]
    assert record["feedback"] == []
