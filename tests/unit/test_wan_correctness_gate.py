import hashlib
import json

import pytest

from harness4h3.student.wan_campaign import candidate_decision
from tools.wan_vllm_campaign import correctness_gate_decision, validate_correctness_gate


def test_pending_correctness_gate_rejects_candidate_before_quality_logic():
    decision = candidate_decision(
        {"quality_score": 0.1, "quality_metrics": {"semantic": 0.1}},
        {
            "status": "success",
            "valid": True,
            "offline_simulation": False,
            "continuation_verified": True,
            "quality_score": 1.0,
            "quality_metrics": {"semantic": 1.0},
        },
        teacher_semantic=1.0,
        semantic_floor_ratio=0.1,
        quality_floor_ratio=0.1,
        min_semantic_delta=0.0,
        correctness_gate={"passed": False, "legacy_r2_promotion_frozen": True},
    )
    assert decision["accepted"] is False
    assert decision["reason"] == "correctness_gate_pending"


def test_correctness_gate_requires_existing_digest_bound_evidence(tmp_path):
    config = {
        "correctness_audit_passed": True,
        "correctness_evidence_path": str(tmp_path / "missing.json"),
        "correctness_evidence_sha256": "0" * 64,
        "legacy_r2_promotion_frozen": True,
        "legacy_r2_checkpoint_digests": [],
    }
    with pytest.raises(ValueError, match="correctness evidence"):
        validate_correctness_gate(config)


def test_passed_correctness_gate_freezes_configured_legacy_r2_parent(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"status": "passed"}) + "\n", encoding="utf-8")
    evidence_digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    legacy_digest = "a" * 64
    config = {
        "correctness_audit_passed": True,
        "correctness_evidence_path": str(evidence),
        "correctness_evidence_sha256": evidence_digest,
        "legacy_r2_promotion_frozen": True,
        "legacy_r2_checkpoint_digests": [legacy_digest],
    }
    gate = correctness_gate_decision(config, {"student_checkpoint_sha256": legacy_digest}, {})
    assert gate["passed"] is True
    assert gate["legacy_r2_frozen"] is True
    decision = candidate_decision(
        {},
        {"status": "success", "valid": True, "offline_simulation": False, "continuation_verified": True},
        teacher_semantic=0.0,
        semantic_floor_ratio=0.1,
        quality_floor_ratio=0.1,
        min_semantic_delta=0.0,
        correctness_gate=gate,
    )
    assert decision == {"accepted": False, "reason": "legacy_r2_frozen"}
