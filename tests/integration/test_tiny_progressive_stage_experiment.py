import json

from research.experiments.tiny_progressive_stage_manager import run_tiny_progressive_stage_experiment


def test_formal_tiny_progressive_experiment_promotes_4_to_2_to_1(tmp_path):
    result = run_tiny_progressive_stage_experiment(tmp_path, seed=37)
    assert [item["teacher_nfe"] for item in result.stages] == [4, 2]
    assert [item["student_nfe"] for item in result.stages] == [2, 1]
    assert [item["teacher_model_id"] for item in result.stages] == ["M0000", "M0001"]
    assert [item["student_model_id"] for item in result.stages] == ["M0001", "M0002"]
    assert all(item["accepted"] for item in result.stages)
    assert result.final_model_id == "M0002"
    manifest = json.loads((tmp_path / "progressive-stage-manifest.json").read_text(encoding="utf-8"))
    assert manifest["manager"]["teacher_model_id"] == "M0002"
    assert len(manifest["manager"]["promotions"]) == 2
