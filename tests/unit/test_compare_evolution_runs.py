from __future__ import annotations

import json

from tools.compare_evolution_runs import compare_runs


def _write_run(root, *, status="rejected", recipe="recipe-a", quality=0.9, case_count=2, source_revision="revision-a"):
    round_dir = root / "rounds" / "R0001"
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
    (round_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    (round_dir / "decision.json").write_text(json.dumps({"status": status, "evaluation": evaluation}), encoding="utf-8")


def test_compare_preserves_missing_active_metrics_and_sums_cost(tmp_path):
    _write_run(tmp_path / "fixed")
    result = compare_runs(["fixed=%s" % (tmp_path / "fixed")], expected_cases=2)
    item = result["runs"][0]
    assert item["active_model_metrics_available"] is False
    assert item["gpu_runtime_cost_s"] == 36.0
    assert item["final"]["quality_score"] == 0.9
    assert result["protocol_comparable"] is True


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
