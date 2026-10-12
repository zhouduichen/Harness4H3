from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.evaluator import WanVideoEvaluationConfig, WanVideoEvaluator
from harness4h3.student.quality import QualityEvidence


def test_real_evaluator_config_requires_trusted_inputs(tmp_path):
    try:
        WanVideoEvaluationConfig.from_mapping({}, tmp_path / "out")
    except ValueError as exc:
        assert "wan_config" in str(exc)
        assert "evaluation_manifest" in str(exc)
    else:
        raise AssertionError("missing real evaluator inputs were accepted")


def test_real_evaluator_records_video_and_quality_evidence(tmp_path, monkeypatch):
    wan_config = tmp_path / "wan.yaml"
    wan_config.write_text(
        "width: 8\nheight: 8\nnum_frames: 4\nnative_fps: 8\n"
        "num_inference_steps: 1\nguidance_scale: 1.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": "smoke", "caption": "a test video", "seeds": [7]}]}), encoding="utf-8")
    config = WanVideoEvaluationConfig(
        wan_config=wan_config,
        evaluation_manifest=manifest,
        clip_model_path=tmp_path / "clip",
        output_root=tmp_path / "evaluation",
    )
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    baseline = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "baseline")
    child = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "child")
    quality = QualityEvidence(
        semantic=0.8,
        temporal=0.9,
        motion=0.2,
        aggregate=0.75,
        backend="test",
        model_path="test",
    )
    evaluator = WanVideoEvaluator(config, baseline, quality_backend=SimpleNamespace(evaluate=lambda path, caption: quality), pipeline_loader=lambda checkpoint: (object(), object()))
    monkeypatch.setattr(
        evaluator,
        "_generate",
        lambda pipeline, torch, caption, seed: np.linspace(0.0, 1.0, 4 * 8 * 8 * 3, dtype=np.float32).reshape(1, 4, 8, 8, 3),
    )

    result = evaluator(child, {})

    assert result["offline_simulation"] is False
    assert result["feasible"] is True
    assert result["quality_metrics"]["score_type"] == "clip_temporal"
    assert Path(result["cases"][0]["video_path"]).is_file()
