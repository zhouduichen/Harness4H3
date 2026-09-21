from __future__ import annotations

from harness4h3.campaign.failures import FailureAttributor


def test_decode_failure_is_attributed_without_llm_category():
    report = FailureAttributor().attribute(
        "generation",
        {"failure_code": "video_decode_failed", "message": "invalid mp4"},
        ("evidence-v3",),
    )
    assert report.category == "generation_decode"
    assert report.deterministic_fix
    assert "video_decode_failed" in report.prohibited_changes
    assert report.confidence == 1.0


def test_unknown_failure_keeps_evidence_without_inventing_fix():
    report = FailureAttributor().attribute(
        "training",
        {"failure_code": "new_unknown_failure", "message": "unclear"},
        ("evidence-x",),
    )
    assert report.category == "unclassified_experimental_failure"
    assert report.deterministic_fix is None
    assert report.confidence < 1.0
    assert report.evidence_ids == ("evidence-x",)
