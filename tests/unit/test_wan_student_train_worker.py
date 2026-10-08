from pathlib import Path

import pytest

import torch

from tools.wan_student_train_worker import _encode_training_prompt, _parse_devices, _training_spec


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
