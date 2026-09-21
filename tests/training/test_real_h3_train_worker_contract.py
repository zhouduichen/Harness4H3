import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from tools import h3_real_train_worker as worker
from tools import h3_model_worker as model_worker


def _request(tmp_path, operator_args):
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / "parent.safetensors"
    save_file(
        {"placeholder": torch.ones(1)},
        str(parent),
        metadata={
            "config": json.dumps(
                {"transformer": {"image_model": "minimax_h3", "num_layers": 2}},
                separators=(",", ":"),
            )
        },
    )
    teacher = tmp_path / "teacher.safetensors"
    save_file(
        {"placeholder": torch.ones(1)},
        str(teacher),
        metadata={
            "config": json.dumps(
                {"transformer": {"image_model": "minimax_h3", "num_layers": 2}},
                separators=(",", ":"),
            )
        },
    )
    return {
        "operator": "distill",
        "operator_args": operator_args,
        "parent": {"checkpoint_path": str(parent)},
        "artifacts_dir": str(tmp_path / "artifacts"),
    }, parent


def test_distill_contract_requires_the_configured_full_cache(tmp_path):
    request, parent = _request(tmp_path, {"dataset_fraction": 1.0, "training_steps": 4})
    paths = worker.validate_config(
        {
            "comfyui_root": str(tmp_path),
            "model_checkpoint": str(parent),
            "source_parent_checkpoint": str(tmp_path / "teacher.safetensors"),
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "out"),
            "world_size": 4,
            "max_steps": 4,
        },
        request,
    )
    assert paths["operator"] == "distill"
    assert paths["dataset_fraction"] == 1.0

    bad_request, _ = _request(tmp_path / "bad", {"dataset_fraction": 0.5, "training_steps": 4})
    with pytest.raises(ValueError, match="dataset_fraction=1.0"):
        worker.validate_config(
            {
                "comfyui_root": str(tmp_path),
                "model_checkpoint": str((tmp_path / "bad" / "parent.safetensors")),
                "source_parent_checkpoint": str(tmp_path / "bad" / "teacher.safetensors"),
                "cache_dir": str(tmp_path / "bad" / "cache"),
                "output_dir": str(tmp_path / "bad" / "out"),
                "world_size": 4,
                "max_steps": 4,
            },
            bad_request,
        )


def test_distributed_preflight_accepts_two_gpu_world_size(tmp_path):
    request, parent = _request(tmp_path, {"dataset_fraction": 1.0, "training_steps": 1})
    paths = worker.validate_config(
        {
            "comfyui_root": str(tmp_path),
            "model_checkpoint": str(parent),
            "source_parent_checkpoint": str(tmp_path / "teacher.safetensors"),
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "out"),
            "world_size": 2,
            "max_steps": 1,
        },
        request,
    )
    assert paths["world_size"] == 2


def test_distributed_preflight_rejects_single_gpu_world_size(tmp_path):
    request, parent = _request(tmp_path, {"dataset_fraction": 1.0, "training_steps": 1})
    with pytest.raises(ValueError, match=r"world_size in \[2, 4\]"):
        worker.validate_config(
            {
                "comfyui_root": str(tmp_path),
                "model_checkpoint": str(parent),
                "source_parent_checkpoint": str(tmp_path / "teacher.safetensors"),
                "cache_dir": str(tmp_path / "cache"),
                "output_dir": str(tmp_path / "out"),
                "world_size": 1,
                "max_steps": 1,
            },
            request,
        )


def test_dmd2_preflight_requires_a_bounded_generator_update_interval(tmp_path):
    request, parent = _request(tmp_path, {"training_steps": 4, "generator_update_interval": 2})
    request = {**request, "operator": "dmd2"}
    paths = worker.validate_config(
        {
            "comfyui_root": str(tmp_path),
            "model_checkpoint": str(parent),
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "out"),
            "world_size": 4,
            "max_steps": 4,
        },
        request,
    )
    assert paths["operator"] == "dmd2"
    assert paths["generator_update_interval"] == 2

    bad = {**request, "operator_args": {"training_steps": 4, "generator_update_interval": 5}}
    with pytest.raises(ValueError, match="generator_update_interval"):
        worker.validate_config(
            {
                "comfyui_root": str(tmp_path),
                "model_checkpoint": str(parent),
                "cache_dir": str(tmp_path / "cache"),
                "output_dir": str(tmp_path / "out"),
                "world_size": 4,
                "max_steps": 4,
            },
            bad,
        )


def test_dmd2_latent_critic_has_video_and_audio_outputs():
    critic = worker.H3DMD2Critic()
    video = torch.randn(1, 24, 5, 16, 16)
    audio = torch.randn(1, 32, 2, 37)
    predicted_video, predicted_audio = critic(video, audio, 0.5)
    assert predicted_video.shape == video.shape
    assert predicted_audio.shape == audio.shape
    assert torch.isfinite(predicted_video).all()
    assert torch.isfinite(predicted_audio).all()


def test_nonzero_rank_head_placeholders_are_materialized_for_fsdp_sync():
    class Placeholder(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.register_parameter("weight", None)
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))

    class FinalLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.video_out = Placeholder(5376, 96)
            self.audio_out = Placeholder(5376, 32)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.final_layer = FinalLayer()

    model = Model()
    assert worker._replace_trainable_output_heads(model) == 2
    assert isinstance(model.final_layer.video_out, worker._TrainableLinear)
    assert isinstance(model.final_layer.audio_out, worker._TrainableLinear)
    assert model.final_layer.video_out.weight.shape == (96, 5376)
    assert model.final_layer.audio_out.weight.shape == (32, 5376)
    assert model.final_layer.video_out.weight.dtype == torch.float32
    assert model.final_layer.audio_out.weight.dtype == torch.float32
    assert all(parameter.requires_grad for parameter in model.final_layer.video_out.parameters())
    assert all(parameter.requires_grad for parameter in model.final_layer.audio_out.parameters())


def test_external_worker_requires_complete_training_evidence():
    metrics = {
        "initial_loss": 1.0,
        "final_loss": 0.9,
        "gradient_norm": 0.5,
        "optimizer_steps": 1,
        "trainable_parameter_count": 4,
        "parent_sha256": "a" * 64,
        "parent_sha256_before": "a" * 64,
        "parent_sha256_after": "a" * 64,
        "child_sha256": "b" * 64,
        "changed_trainable_tensors": 1,
        "unchanged_frozen_tensors": 2,
        "child_reloaded": True,
        "peak_vram_per_rank": {"0": 1024},
        "gpu_time_s_per_rank": [1.0],
        "wall_time_s": 2.0,
    }
    assert model_worker._validate_training_evidence(
        "recovery_finetune", {"metrics": metrics}, "a" * 64
    )["gradient_norm"] == 0.5
    for missing in (
        "peak_vram_per_rank",
        "gpu_time_s_per_rank",
        "wall_time_s",
        "child_reloaded",
    ):
        incomplete = dict(metrics)
        incomplete.pop(missing)
        with pytest.raises(ValueError, match="missing measured training metric"):
            model_worker._validate_training_evidence("recovery_finetune", {"metrics": incomplete}, "a" * 64)
