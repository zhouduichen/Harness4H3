import hashlib
import json

import pytest

from harness4h3.student.wan_campaign import candidate_decision
from tools.wan_vllm_campaign import correctness_gate_decision, validate_correctness_gate
from h3_training.wan_native_gate import evaluate_native_promotion


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


def test_native_pure_color_case_vetoes_promotion(tmp_path):
    result = evaluate_native_promotion(
        {"quality_score": 0.66, "visual_cases": [{"passed": False, "reason": "pure_color_block"}], "nfe": 192, "pure_seconds": 8.0, "peak_vram_gib": 14.0},
        stage="F3",
        base_wan32_quality=0.6633,
        frozen_r2_quality=0.6552,
        record_valid=True,
        scheduler_parity_ok=True,
        failure_memory_path=tmp_path / "failure-memory.jsonl",
    )
    assert result["all_visual_pass"] is False
    assert result["promotion_allowed"] is False
    assert "pure_color_block" in result["failure_memory"][0]["reason"]
    assert (tmp_path / "failure-memory.jsonl").read_text(encoding="utf-8").count("pure_color_block") == 1


def test_native_f3_requires_quality_and_frozen_r2_margin():
    passing = evaluate_native_promotion(
        {"quality_score": 0.660, "visual_cases": [{"passed": True}], "nfe": 192, "pure_seconds": 8.0, "peak_vram_gib": 14.0},
        stage="F3", base_wan32_quality=0.6633, frozen_r2_quality=0.6552,
        record_valid=True, scheduler_parity_ok=True,
    )
    failing = evaluate_native_promotion(
        {"quality_score": 0.657, "visual_cases": [{"passed": True}], "nfe": 192, "pure_seconds": 8.0, "peak_vram_gib": 14.0},
        stage="F3", base_wan32_quality=0.6633, frozen_r2_quality=0.6552,
        record_valid=True, scheduler_parity_ok=True,
    )
    assert passing["promotion_allowed"] is True
    assert failing["promotion_allowed"] is False


def test_native_partial_stage_never_enables_controller():
    result = evaluate_native_promotion(
        {"quality_score": 0.660, "visual_cases": [{"passed": True}], "nfe": 192, "pure_seconds": 8.0, "peak_vram_gib": 14.0},
        stage="F2", base_wan32_quality=0.6633, frozen_r2_quality=0.6552,
        record_valid=True, scheduler_parity_ok=True,
    )
    assert result["promotion_allowed"] is False
    assert result["controller_search_enabled"] is False


def test_native_f3_requires_p0_artifacts():
    result = evaluate_native_promotion(
        {"quality_score": 0.660, "visual_cases": [{"passed": True}], "nfe": 192, "pure_seconds": 8.0, "peak_vram_gib": 14.0},
        stage="F3", base_wan32_quality=0.6633, frozen_r2_quality=0.6552,
        record_valid=True, scheduler_parity_ok=True,
        required_artifacts=[{"path": "missing-h3-rgb.mp4", "present": False}],
    )
    assert result["promotion_allowed"] is False
    assert result["failure_memory"][0]["reason"] == "required_artifact_missing"
