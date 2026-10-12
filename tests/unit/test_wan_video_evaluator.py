from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.evaluator import WanEvaluationError, WanVideoEvaluationConfig, WanVideoEvaluator
from harness4h3.student.quality import QualityEvidence


def test_real_evaluator_config_requires_trusted_inputs(tmp_path):
    try:
        WanVideoEvaluationConfig.from_mapping({}, tmp_path / "out")
    except ValueError as exc:
        assert "wan_config" in str(exc)
        assert "evaluation_manifest" in str(exc)
    else:
        raise AssertionError("missing real evaluator inputs were accepted")


def test_real_evaluator_defaults_quality_device_to_generation_device(tmp_path):
    config = WanVideoEvaluationConfig.from_mapping(
        {
            "wan_config": str(tmp_path / "wan.yaml"),
            "evaluation_manifest": str(tmp_path / "manifest.json"),
            "clip_model_path": str(tmp_path / "clip"),
            "device": "cuda:3",
            "source_revision": "test-revision",
        },
        tmp_path / "out",
    )
    assert config.quality_device == "cuda:3"
    assert config.source_revision == "test-revision"


def test_real_evaluator_rejects_manifest_smaller_than_requested_budget(tmp_path):
    wan_config = tmp_path / "wan.yaml"
    wan_config.write_text(
        "width: 8\nheight: 8\nnum_frames: 4\nnative_fps: 8\n"
        "num_inference_steps: 1\nguidance_scale: 1.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": "only", "caption": "test", "seeds": [7]}]}), encoding="utf-8")
    config = WanVideoEvaluationConfig(
        wan_config=wan_config,
        evaluation_manifest=manifest,
        clip_model_path=tmp_path / "clip",
        output_root=tmp_path / "evaluation",
        max_cases=2,
    )
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    baseline = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "baseline")
    try:
        WanVideoEvaluator(config, baseline)
    except WanEvaluationError as exc:
        assert "insufficient_cases" in str(exc)
    else:
        raise AssertionError("undersized manifest was accepted for a larger budget")


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
    assert len(result["cases"][0]["latency_samples_s"]) == 3
    assert result["hardware_metrics"]["gpu_runtime_cost_s"] >= result["hardware_metrics"]["inference_wall_time_s_median"]
    assert len(result["evaluator_recipe_sha256"]) == 64
    assert result["checkpoint_sha256"] == child.sha256
    assert len(result["evaluation_manifest_sha256"]) == 64
    assert result["evaluator_version"]
    assert result["source_revision"] == "unknown"
    assert list((tmp_path / "evaluation" / "evaluations" / child.sha256).iterdir())


def test_real_evaluator_caches_prompt_embeddings_for_latency_repeats(tmp_path, monkeypatch):
    wan_config = tmp_path / "wan.yaml"
    wan_config.write_text(
        "width: 8\nheight: 8\nnum_frames: 4\nnative_fps: 8\n"
        "num_inference_steps: 1\nguidance_scale: 1.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": "cached", "caption": "a test video", "seeds": [7]}]}), encoding="utf-8")
    config = WanVideoEvaluationConfig(
        wan_config=wan_config,
        evaluation_manifest=manifest,
        clip_model_path=tmp_path / "clip",
        output_root=tmp_path / "evaluation",
    )
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    baseline = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "baseline")
    child_model = adapter.load_model("fake://baseline")
    adapter.remove_blocks(child_model, ["block.0"])
    child = adapter.save_model(child_model, tmp_path / "child")
    quality = QualityEvidence(
        semantic=0.8,
        temporal=0.9,
        motion=0.2,
        aggregate=0.75,
        backend="test",
        model_path="test",
    )

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = "bfloat16"

        @staticmethod
        def device(value):
            return value

    class Pipeline:
        text_encoder = type("TextEncoder", (), {"dtype": "bfloat16"})()

        def __init__(self):
            self.embed_calls = []

        def _get_t5_prompt_embeds(self, caption, *, device, dtype):
            self.embed_calls.append((caption, device, dtype))
            return type("Embedding", (), {"to": lambda self, _device: self})()

    pipeline = Pipeline()
    evaluator = WanVideoEvaluator(
        config,
        baseline,
        quality_backend=SimpleNamespace(evaluate=lambda path, caption: quality),
        pipeline_loader=lambda checkpoint: (pipeline, FakeTorch()),
    )
    generate_calls = []

    def generate(_pipeline, _torch, caption, seed, **kwargs):
        generate_calls.append((caption, seed, kwargs))
        return np.linspace(0.0, 1.0, 4 * 8 * 8 * 3, dtype=np.float32).reshape(1, 4, 8, 8, 3)

    monkeypatch.setattr(evaluator, "_generate", generate)
    result = evaluator(child, {})

    assert result["hardware_metrics"]["latency_measurement_method"] == "denoising+vae_decode_with_case_cached_prompt_embeddings"
    assert result["hardware_metrics"]["prompt_encoding_included"] is False
    assert len(pipeline.embed_calls) == 4  # positive and negative text, once for each artifact
    assert len(generate_calls) == 8  # baseline and child, warmup plus three repeats
    assert all(call[2]["prompt_embeds"] is not None for call in generate_calls)


def test_real_evaluator_rejects_visual_quality_collapse(tmp_path, monkeypatch):
    wan_config = tmp_path / "wan.yaml"
    wan_config.write_text(
        "width: 8\nheight: 8\nnum_frames: 4\nnative_fps: 8\n"
        "num_inference_steps: 1\nguidance_scale: 1.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": "collapse", "caption": "test", "seeds": [7]}]}), encoding="utf-8")
    config = WanVideoEvaluationConfig(
        wan_config=wan_config,
        evaluation_manifest=manifest,
        clip_model_path=tmp_path / "clip",
        output_root=tmp_path / "evaluation",
    )
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    baseline = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "baseline")
    child = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "child")
    collapsed = QualityEvidence(
        semantic=0.9,
        temporal=0.9,
        motion=0.0,
        aggregate=0.9,
        backend="test",
        model_path="test",
        visual_quality_gate={"enabled": True, "passed": False, "failures": ["low_texture_frame_ratio_above_ceiling"]},
    )
    evaluator = WanVideoEvaluator(config, baseline, quality_backend=SimpleNamespace(evaluate=lambda path, caption: collapsed), pipeline_loader=lambda checkpoint: (object(), object()))
    monkeypatch.setattr(
        evaluator,
        "_generate",
        lambda pipeline, torch, caption, seed: np.zeros((1, 4, 8, 8, 3), dtype=np.float32),
    )

    result = evaluator(child, {})

    assert result["valid"] is False
    assert result["promotable"] is False
    assert result["cases"][0]["validity"]["visual_quality_gate_passed"] is False


def test_real_evaluator_resumes_completed_case_from_progress(tmp_path, monkeypatch):
    wan_config = tmp_path / "wan.yaml"
    wan_config.write_text(
        "width: 8\nheight: 8\nnum_frames: 4\nnative_fps: 8\n"
        "num_inference_steps: 1\nguidance_scale: 1.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "first", "caption": "first", "seeds": [7]},
                    {"case_id": "second", "caption": "second", "seeds": [7]},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = WanVideoEvaluationConfig(
        wan_config=wan_config,
        evaluation_manifest=manifest,
        clip_model_path=tmp_path / "clip",
        output_root=tmp_path / "evaluation",
        max_cases=2,
    )
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    baseline = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "baseline")
    child_model = adapter.load_model("fake://baseline")
    adapter.remove_blocks(child_model, ["block.0"])
    child = adapter.save_model(child_model, tmp_path / "child")
    quality = QualityEvidence(
        semantic=0.8,
        temporal=0.9,
        motion=0.2,
        aggregate=0.75,
        backend="test",
        model_path="test",
    )
    evaluator = WanVideoEvaluator(
        config,
        baseline,
        quality_backend=SimpleNamespace(evaluate=lambda path, caption: quality),
        pipeline_loader=lambda checkpoint: (checkpoint, object()),
    )
    completed_video = tmp_path / "completed.mp4"
    completed_video.write_bytes(b"already generated")
    progress_dir = tmp_path / "evaluation" / "evaluations" / child.sha256 / evaluator.recipe_digest
    progress_dir.mkdir(parents=True)
    (progress_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "in_progress",
                "checkpoint_sha256": child.sha256,
                "evaluator_recipe_sha256": evaluator.recipe_digest,
                "cases": [
                    {
                        "case_id": "first",
                        "seed": 7,
                        "caption": "first",
                        "video_path": str(completed_video),
                        "generation_wall_time_s": 1.0,
                        "latency_samples_s": [1.0, 1.0, 1.0],
                        "peak_memory_gb": 1.0,
                        "validity": {"valid": True},
                        "quality": quality.to_dict(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def generate(pipeline, torch, caption, seed):
        calls.append((pipeline, caption, seed))
        return np.linspace(0.0, 1.0, 4 * 8 * 8 * 3, dtype=np.float32).reshape(1, 4, 8, 8, 3)

    monkeypatch.setattr(evaluator, "_generate", generate)
    result = evaluator(child, {})

    # Parent: 2 cases x (warmup + 3 repeats); Child: only the unfinished case.
    assert len(calls) == 12
    assert result["resumed_case_count"] == 1
    assert result["hardware_metrics"]["case_count"] == 2
