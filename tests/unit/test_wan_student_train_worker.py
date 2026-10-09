from pathlib import Path

import pytest

import torch

from tools.wan_student_train_worker import (
    _apply_initialization_mode,
    _configure_trainable_scope,
    _encode_training_prompt,
    _parse_devices,
    _trainable_parameter_probe,
    _training_cache_paths,
    _training_spec,
)


def test_worker_schedule_provider_uses_native_32_and_coarsened_16():
    from diffusers import FlowMatchEulerDiscreteScheduler
    from tools.wan_student_train_worker import _wan_schedule_provider

    provider = _wan_schedule_provider(
        FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0), source_steps=32
    )
    source = provider(32)
    target = provider(16)
    assert source.mode == "native"
    assert target.mode == "coarsened_from_source"
    assert target.sigmas == source.sigmas[::2]


def test_parse_devices_requires_three_distinct_cuda_devices():
    assert _parse_devices(["cuda:0", "cuda:1", "cuda:2"]) == ("cuda:0", "cuda:1", "cuda:2")
    with pytest.raises(ValueError, match="exactly three"):
        _parse_devices(["cuda:0", "cuda:1"])
    with pytest.raises(ValueError, match="distinct"):
        _parse_devices(["cuda:0", "cuda:0", "cuda:1"])


def test_training_spec_reads_the_existing_wan_model_config(tmp_path: Path):
    wan_config = tmp_path / "wan.yaml"
    wan_config.write_text(
        "wan_student:\n"
        "  model_id: Wan-AI/Wan2.1-T2V-1.3B-Diffusers\n"
        "  pretrained_path: /models/wan\n"
        "  lineage_root: W0000\n"
        "  width: 256\n"
        "  height: 256\n"
        "  num_frames: 81\n"
        "  native_fps: 16.0\n"
        "  dtype: bf16\n",
        encoding="utf-8",
    )
    spec = _training_spec(
        {"learning_rate": 1.0e-6, "h3_latent_frames": 36, "h3_latent_height": 16, "h3_latent_width": 16, "h3_latent_channels": 24},
        wan_config,
    )
    assert spec.model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert spec.lineage_root == "W0000"
    assert spec.num_frames == 81
    assert spec.h3_latent_frames == 36


def test_training_prompt_is_normal_autograd_tensor():
    class FakePipeline:
        @staticmethod
        def encode_prompt(*args, **kwargs):
            with torch.inference_mode():
                return torch.ones(1, 4, 8), None

    class Spec:
        max_sequence_length = 512

    result = _encode_training_prompt(FakePipeline(), "test", Spec(), torch.device("cpu"), torch.float32)
    assert not result.is_inference()
    assert result.requires_grad is False


def test_training_cache_paths_select_unique_captions(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for index, caption in enumerate(("red plane", "red plane", "orange cat", "yellow train")):
        torch.save({"caption": caption}, cache_dir / ("%08d.pt" % index))

    selected = _training_cache_paths({"h3_cache_dir": str(cache_dir)}, limit=3)

    assert [path.name for path in selected] == ["00000000.pt", "00000002.pt", "00000003.pt"]


def test_training_cache_paths_enforces_native_5s_contract(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save(
        {
            "prompt": torch.zeros(2, 5120),
            "video": torch.zeros(24 * 64, 96),
            "audio": torch.zeros(404, 32),
            "caption": "native",
            "latent_frames": 24,
            "num_frames": 81,
            "height": 256,
            "width": 256,
            "teacher_generated": True,
            "synthetic_latent": False,
            "offline_simulation": False,
        },
        cache_dir / "00000000.pt",
    )
    selected = _training_cache_paths({"h3_cache_dir": str(cache_dir), "cache_contract": "native_5s"}, limit=1)
    assert [path.name for path in selected] == ["00000000.pt"]


def test_trainable_scope_can_keep_only_text_conditioning_paths():
    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].attn2 = torch.nn.Module()
            self.blocks[0].attn2.to_k = torch.nn.Linear(2, 2)
            self.blocks[0].attn2.to_q = torch.nn.Linear(2, 2)
            self.condition_embedder = torch.nn.Module()
            self.condition_embedder.text_embedder = torch.nn.Linear(2, 2)

    transformer = FakeTransformer()
    scope = _configure_trainable_scope(
        transformer,
        "blocks.*.attn2.to_k.*,condition_embedder.text_embedder.*",
    )

    assert scope.startswith("blocks.*.attn2.to_k.*")
    assert transformer.blocks[0].attn2.to_k.weight.requires_grad
    assert not transformer.blocks[0].attn2.to_q.weight.requires_grad
    assert transformer.condition_embedder.text_embedder.weight.requires_grad


def test_trainable_parameter_probe_ignores_frozen_prefix_parameters():
    transformer = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    list(transformer.parameters())[0].requires_grad_(False)
    list(transformer.parameters())[1].requires_grad_(False)
    list(transformer.parameters())[2].requires_grad_(True)

    name, snapshot = _trainable_parameter_probe(transformer)

    assert name == "1.weight"
    assert snapshot.dtype == torch.float32
    assert snapshot.shape == transformer[1].weight.shape


def test_initialization_mode_resets_wan_output_projection():
    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_out = torch.nn.Linear(2, 2)

    transformer = FakeTransformer()
    assert _apply_initialization_mode(transformer, "inherit") == "inherit"
    assert _apply_initialization_mode(transformer, "generation_head_reset") == "generation_head_reset"
    assert torch.count_nonzero(transformer.proj_out.weight) == 0
    assert torch.count_nonzero(transformer.proj_out.bias) == 0
