from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.evaluator import WanEvaluationError, WanVideoEvaluationConfig, WanVideoEvaluator
from harness4h3.student.quality import QualityEvidence


def test_checkpoint_size_deduplicates_hardlinks_and_symlinks(tmp_path):
    root = tmp_path / "pipeline"
    root.mkdir()
    source = root / "weights.bin"
    source.write_bytes(b"1234")
    (root / "hardlink.bin").hardlink_to(source)
    (root / "symlink.bin").symlink_to(source)

    assert WanVideoEvaluator._checkpoint_size_gb(str(root)) == pytest.approx(4 / 1024**3)


def test_unavailable_peak_memory_is_none():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    evaluator = object.__new__(WanVideoEvaluator)
    evaluator.config = SimpleNamespace(device="cuda:7")

    assert evaluator._peak_memory_gb(Torch) is None


def test_candidate_external_policy_installation_preserves_order_and_provenance(monkeypatch):
    installed = []

    class Handle:
        def summary(self):
            return {"calls": 12}

    sparsevideo = SimpleNamespace(
        default_method_config=lambda method, **kwargs: {"method": method, **kwargs},
        apply=lambda pipeline, **kwargs: (installed.append("svoo"), Handle())[1],
    )
    diffusers = SimpleNamespace(__version__="test-diffusers")

    class MagCacheConfig:
        def __init__(self, **kwargs):
            self.values = kwargs

    hooks = SimpleNamespace(MagCacheConfig=MagCacheConfig)
    monkeypatch.setitem(sys.modules, "sparsevideo", sparsevideo)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)

    class Transformer:
        def enable_cache(self, config):
            installed.append("magcache")

    pipeline = SimpleNamespace(
        transformer=Transformer(),
        _harness4h3_external_runtime_policies=[
            {"operator": "svoo_attention", "args": {"method": "svoo", "model_key": "wan21-t2v-1.3b", "config": {}}},
            {"operator": "magcache", "args": {"method": "magcache", "config": {"threshold": 0.12}}},
        ],
    )
    evaluator = object.__new__(WanVideoEvaluator)
    evaluator.config = SimpleNamespace(runtime_attention={}, runtime_cache={})
    evaluator._wan_spec = {"num_inference_steps": 16}
    returned = evaluator._install_persisted_external_policies(pipeline)

    assert returned is pipeline
    assert installed == ["svoo", "magcache"]
    assert pipeline._harness4h3_runtime_policy_provenance["installation_order"] == ["svoo_attention", "magcache"]
    assert WanVideoEvaluator._runtime_attention_summary(pipeline)["handle"] == {"calls": 12}
    assert pipeline._harness4h3_runtime_cache["diffusers_version"] == "test-diffusers"


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
    assert config.resolved_pipeline_device == "cuda:3"
    assert config.source_revision == "test-revision"


def test_real_evaluator_replays_raw_runtime_manifest_before_device_move(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    transformer_dir = checkpoint / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "harness4h3_runtime_operations": [
                    {"operator": "sampling_step_skip", "args": {"target_steps": 4, "strategy": "uniform"}}
                ]
            }
        ),
        encoding="utf-8",
    )

    class Transformer:
        def __init__(self):
            self.config = SimpleNamespace()
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1
            return self

    class Pipeline:
        def __init__(self):
            self.transformer = Transformer()
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

        def set_progress_bar_config(self, **kwargs):
            self.progress_kwargs = kwargs

    pipeline = Pipeline()

    class DiffusionPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return pipeline

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(DiffusionPipeline=DiffusionPipeline))
    evaluator = object.__new__(WanVideoEvaluator)
    evaluator._pipeline_loader = None
    evaluator.config = SimpleNamespace(
        local_files_only=True,
        dtype="float32",
        device="cpu",
        resolved_pipeline_device="cpu",
    )

    loaded, _torch = evaluator._load_pipeline(str(checkpoint))

    assert loaded is pipeline
    assert pipeline.transformer._harness4h3_sampling_step_policy["target_steps"] == 4
    assert [item["operator"] for item in pipeline.transformer._harness4h3_runtime_operations] == ["sampling_step_skip"]
    assert pipeline.to_calls == ["cpu"]


def test_real_evaluator_allows_explicit_pipeline_device_and_binds_recipe(tmp_path):
    (tmp_path / "wan.yaml").write_text(
        "width: 8\nheight: 8\nnum_frames: 4\nnative_fps: 8\n"
        "num_inference_steps: 1\nguidance_scale: 1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"cases": [{"case_id": "one", "caption": "test", "seeds": [7]}]}),
        encoding="utf-8",
    )
    raw = {
        "wan_config": str(tmp_path / "wan.yaml"),
        "evaluation_manifest": str(tmp_path / "manifest.json"),
        "clip_model_path": str(tmp_path / "clip"),
        "device": "cuda:3",
        "pipeline_device": "cpu",
    }
    explicit = WanVideoEvaluationConfig.from_mapping(raw, tmp_path / "explicit")
    assert explicit.resolved_pipeline_device == "cpu"

    raw["pipeline_device"] = None
    inherited = WanVideoEvaluationConfig.from_mapping(raw, tmp_path / "inherited")
    assert inherited.resolved_pipeline_device == "cuda:3"

    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    baseline = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "baseline")
    explicit_digest = WanVideoEvaluator(explicit, baseline).recipe_digest
    inherited_digest = WanVideoEvaluator(inherited, baseline).recipe_digest
    assert explicit_digest != inherited_digest


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


def test_real_evaluator_moves_complete_pipeline_and_keeps_transformer_in_eval_mode():
    class Component:
        def __init__(self):
            self.devices = []
            self.eval_calls = 0

        def to(self, device):
            self.devices.append(device)
            return self

        def eval(self):
            self.eval_calls += 1
            return self

    class Pipeline:
        def __init__(self):
            self.transformer = Component()
            self.vae = Component()
            self.text_encoder = Component()
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

    pipeline = Pipeline()
    moved = WanVideoEvaluator._move_pipeline_to_device(pipeline, "cuda:2")
    assert moved is pipeline
    assert pipeline.to_calls == ["cuda:2"]
    assert pipeline.transformer.eval_calls == 1

    class FallbackPipeline:
        def __init__(self):
            self.transformer = Component()
            self.vae = Component()
            self.text_encoder = Component()

    fallback = FallbackPipeline()
    WanVideoEvaluator._move_pipeline_to_device(fallback, "cpu")
    assert fallback.transformer.devices == ["cpu"]
    assert fallback.vae.devices == ["cpu"]
    assert fallback.text_encoder.devices == ["cpu"]
    assert fallback.transformer.eval_calls == 1


def test_real_evaluator_uses_vae_parameter_device_for_decode():
    class Torch:
        @staticmethod
        def device(value):
            return "device:%s" % value

    class Parameter:
        device = "device:cuda:4"

    class Vae:
        @staticmethod
        def parameters():
            yield Parameter()

    assert WanVideoEvaluator._module_device(Vae(), Torch, "cpu") == "device:cuda:4"

    class NoParameters:
        pass

    assert WanVideoEvaluator._module_device(NoParameters(), Torch, "cpu") == "device:cpu"


def test_real_evaluator_decodes_latents_on_vae_device(tmp_path):
    torch = __import__("torch")

    class Vae(torch.nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.decode_devices = []

        def decode(self, latents, return_dict=False):
            self.decode_devices.append(latents.device)
            return (latents,)

    class Pipeline:
        text_encoder = SimpleNamespace(dtype=torch.float32)

        def __init__(self):
            self.vae = Vae()
            self.video_processor = SimpleNamespace(postprocess_video=lambda value, output_type: value.detach().numpy())

        def _get_t5_prompt_embeds(self, caption, *, device, dtype):
            del caption
            return torch.zeros((1, 1), device=device, dtype=dtype)

        def __call__(self, **kwargs):
            assert kwargs["prompt_embeds"].device.type == "cpu"
            return SimpleNamespace(frames=torch.ones((1, 1, 1, 1, 1)))

    evaluator = object.__new__(WanVideoEvaluator)
    evaluator.config = SimpleNamespace(
        device="cpu",
        resolved_pipeline_device="cpu",
        dtype="float32",
    )
    evaluator._wan_spec = {
        "height": 1,
        "width": 1,
        "num_frames": 1,
        "num_inference_steps": 1,
        "guidance_scale": 1.0,
    }
    pipeline = Pipeline()
    evaluator._generate(pipeline, torch, "test", 7)
    assert pipeline.vae.decode_devices == [torch.device("cpu")]


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
    assert result["generation_device"] == "cuda:0"
    assert result["pipeline_device"] == "cuda:0"
    assert result["hardware_metrics"]["input_shape"] == {"height": 8, "width": 8, "num_frames": 4}
    assert result["hardware_metrics"]["peak_memory_gb_max"] is None
    assert result["hardware_metrics"]["peak_memory_available"] is False
    assert result["hardware_metrics"]["parameter_count_scope"] == "transformer"
    assert result["hardware_metrics"]["transformer_parameter_count"] == result["hardware_metrics"]["parameter_count"]
    assert result["hardware_metrics"]["pipeline_model_size_gb"] == result["hardware_metrics"]["model_size_gb"]
    assert result["hardware_metrics"]["transformer_model_size_gb"] is None
    assert list((tmp_path / "evaluation" / "evaluations" / child.sha256).iterdir())


def test_real_evaluator_caches_prompt_embeddings_for_latency_repeats(tmp_path, monkeypatch):
    wan_config = tmp_path / "wan.yaml"
    wan_config.write_text(
        "width: 8\nheight: 8\nnum_frames: 4\nnative_fps: 8\n"
        "num_inference_steps: 1\nguidance_scale: 1.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": "cached", "caption": "a test video", "seeds": [7, 8]}]}), encoding="utf-8")
    config = WanVideoEvaluationConfig(
        wan_config=wan_config,
        evaluation_manifest=manifest,
        clip_model_path=tmp_path / "clip",
        output_root=tmp_path / "evaluation",
        seed_count=2,
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
        reset_devices = []

        @staticmethod
        def is_available():
            return True

        @classmethod
        def reset_peak_memory_stats(cls, device):
            cls.reset_devices.append(device)

        @staticmethod
        def max_memory_allocated(device):
            assert device == "cuda:0"
            return 2 * 1024**3

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
    assert result["hardware_metrics"]["peak_memory_gb_max"] == 2.0
    assert FakeCuda.reset_devices
    assert len(pipeline.embed_calls) == 4  # positive and negative text, once per artifact/case
    assert all(call[1] == "cuda:0" for call in pipeline.embed_calls)
    assert len(generate_calls) == 16  # two seeds per artifact, warmup plus three repeats
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
