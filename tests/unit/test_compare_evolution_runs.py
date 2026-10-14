from __future__ import annotations

import json

from tools.compare_evolution_runs import compare_runs, summarize_run


def _write_active(root, *, active_id="M0000", metrics=None):
    candidates = root / "models" / "candidates"
    candidates.mkdir(parents=True)
    (root / "models" / "active.json").write_text(json.dumps({"id": active_id}), encoding="utf-8")
    (candidates / (active_id + ".json")).write_text(
        json.dumps({"id": active_id, "state": {"measured_metrics": metrics or {}}}),
        encoding="utf-8",
    )


def _write_run(
    root,
    *,
    status="rejected",
    recipe="recipe-a",
    quality=0.9,
    case_count=2,
    source_revision="revision-a",
    round_id="R0001",
    parent_model_id="M0000",
    child_model_id="M0001",
    baseline_quality=None,
    baseline_hardware=None,
    search_score=None,
):
    round_dir = root / "rounds" / "R0001"
    round_dir = root / "rounds" / round_id
    round_dir.mkdir(parents=True)
    evaluation = {
        "status": "success",
        "quality_score": quality,
        "quality_delta": -0.01,
        "evaluator_recipe_sha256": recipe,
        "source_revision": source_revision,
        "hardware_metrics": {
            "inference_wall_time_s_mean": 12.0,
            "peak_memory_gb_max": 2.0,
            "model_size_gb": 8.0,
            "parameter_count": 100,
            "case_count": case_count,
            "gpu_runtime_cost_s": 36.0,
        },
    }
    if baseline_quality is not None:
        evaluation["baseline_quality_score"] = baseline_quality
    if baseline_hardware is not None:
        evaluation["baseline_hardware_metrics"] = baseline_hardware
    if search_score is not None:
        evaluation["search_score"] = search_score
    (round_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    (round_dir / "decision.json").write_text(
        json.dumps(
            {
                "status": status,
                "parent_model_id": parent_model_id,
                "child_model_id": child_model_id,
                "evaluation": evaluation,
            }
        ),
        encoding="utf-8",
    )


def test_compare_preserves_missing_active_metrics_and_sums_cost(tmp_path):
    _write_run(tmp_path / "fixed")
    result = compare_runs(["fixed=%s" % (tmp_path / "fixed")], expected_cases=2)
    item = result["runs"][0]
    assert item["active_model_metrics_available"] is False
    assert item["gpu_runtime_cost_s"] == 36.0
    assert item["final_source"] == "missing"
    assert item["final"]["quality_score"] is None
    assert result["protocol_comparable"] is False


def test_compare_reads_active_id_and_does_not_replace_it_with_rejected_child(tmp_path):
    root = tmp_path / "run"
    _write_active(root, active_id="M0001", metrics={"quality_score": 0.98, "hardware_metrics": {"model_size_gb": 8.0}})
    _write_run(root, parent_model_id="M0001", child_model_id="M0002", quality=0.70)
    result = summarize_run("run", root)
    assert result["active_model_id"] == "M0001"
    assert result["final_source"] == "active_model"
    assert result["final"]["quality_score"] == 0.98
    assert result["last_candidate_model_id"] == "M0002"
    assert result["last_candidate_status"] == "rejected"
    assert result["rejected_model_ids"] == ["M0002"]


def test_compare_keeps_last_and_historical_best_candidates_separate(tmp_path):
    root = tmp_path / "run"
    _write_active(root, active_id="M0000", metrics={"quality_score": 1.0})
    _write_run(root, round_id="R0001", child_model_id="M0001", quality=0.91, search_score=0.91)
    _write_run(root, round_id="R0002", child_model_id="M0002", quality=0.94, search_score=0.94)
    result = summarize_run("run", root)
    assert result["last_candidate_model_id"] == "M0002"
    assert result["historical_best_candidate_model_id"] == "M0002"
    assert result["rejected_model_ids"] == ["M0001", "M0002"]
    assert result["final"]["quality_score"] == 1.0


def test_compare_recovers_only_an_associated_parent_baseline(tmp_path):
    root = tmp_path / "run"
    _write_active(root, active_id="M0000", metrics={})
    _write_run(
        root,
        parent_model_id="M0000",
        child_model_id="M0001",
        baseline_quality=1.0,
        baseline_hardware={"inference_wall_time_s_mean": 10.0, "model_size_gb": 8.0},
    )
    result = summarize_run("run", root)
    assert result["final_source"] == "associated_parent_baseline"
    assert result["final_source_round_id"] == "R0001"
    assert result["final"]["quality_score"] == 1.0
    assert result["final"]["latency_s"] == 10.0


def test_compare_marks_missing_baseline_metrics_instead_of_using_unrelated_candidate(tmp_path):
    root = tmp_path / "run"
    _write_active(root, active_id="M0000", metrics={})
    _write_run(root, parent_model_id="M9999", child_model_id="M0001", quality=0.91)
    result = summarize_run("run", root)
    assert result["final_source"] == "missing"
    assert result["final"]["quality_score"] is None


def test_compare_warns_when_recipes_differ(tmp_path):
    _write_run(tmp_path / "a", recipe="recipe-a")
    _write_run(tmp_path / "b", recipe="recipe-b")
    result = compare_runs(["a=%s" % (tmp_path / "a"), "b=%s" % (tmp_path / "b")])
    assert result["protocol_comparable"] is False
    assert "evaluator_recipe_sha256 differs across runs" in result["warnings"]


def test_compare_marks_case_count_mismatch(tmp_path):
    _write_run(tmp_path / "run", case_count=1)
    result = compare_runs(["run=%s" % (tmp_path / "run")], expected_cases=16)
    assert result["protocol_comparable"] is False
    assert any("expected 16" in warning for warning in result["warnings"])


def test_compare_rejects_missing_or_mismatched_source_revision(tmp_path):
    _write_run(tmp_path / "missing", source_revision="unknown")
    missing = compare_runs(["missing=%s" % (tmp_path / "missing")])
    assert missing["protocol_comparable"] is False
    assert any("source_revision missing" in warning for warning in missing["warnings"])

    _write_run(tmp_path / "a", source_revision="revision-a")
    _write_run(tmp_path / "b", source_revision="revision-b")
    result = compare_runs(["a=%s" % (tmp_path / "a"), "b=%s" % (tmp_path / "b")])
    assert result["protocol_comparable"] is False
    assert "source_revision differs across runs" in result["warnings"]
