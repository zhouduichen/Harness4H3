from __future__ import annotations

import pytest

from harness4h3.controller.reviewer import ReviewDecision, review_json_schema, review_prompt


def valid_review():
    return {
        "action": "continue",
        "reason": "telemetry is healthy",
        "evidence_ids": ["evt-1"],
        "confidence": 0.8,
        "next_review_after_s": 60,
        "risks": [],
    }


def test_review_decision_round_trips_and_rejects_unknown_fields():
    decision = ReviewDecision.from_dict(valid_review())
    assert decision.action == "continue"
    assert decision.evidence_ids == ("evt-1",)
    assert decision.to_dict()["next_review_after_s"] == 60.0
    with pytest.raises(ValueError, match="unknown"):
        ReviewDecision.from_dict({**valid_review(), "extra": True})


def test_review_decision_rejects_unsafe_or_invalid_values():
    with pytest.raises(ValueError, match="action"):
        ReviewDecision.from_dict({**valid_review(), "action": "approve"})
    with pytest.raises(ValueError, match="confidence"):
        ReviewDecision.from_dict({**valid_review(), "confidence": 2})
    with pytest.raises(ValueError, match="next_review_after_s"):
        ReviewDecision.from_dict({**valid_review(), "next_review_after_s": 0})
    with pytest.raises(ValueError, match="evidence_ids"):
        ReviewDecision.from_dict({**valid_review(), "evidence_ids": "evt-1"})


def test_review_schema_is_strict_and_has_only_safe_actions():
    schema = review_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["action"]["enum"] == [
        "continue",
        "stop",
        "replan",
        "review_only",
    ]
    assert set(schema["required"]) == set(schema["properties"])


def test_review_prompt_contains_phase_trigger_and_gpu_isolation():
    prompt = review_prompt(
        {
            "phase": "training",
            "trigger": "heartbeat",
            "telemetry": {"gpu_utilization": {"2": 91}},
            "recent_events": [{"event_type": "worker_started"}],
        },
        review_json_schema(),
    )
    assert "training" in prompt
    assert "heartbeat" in prompt
    assert "GPU0" in prompt
    assert "live gpu_isolation snapshot" in prompt
    assert "single transient controller_review_unavailable" in prompt
