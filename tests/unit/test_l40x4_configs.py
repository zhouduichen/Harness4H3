from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_l40x4_device_profile_is_pending_and_safe():
    raw = load_yaml("configs/devices/l40x4-server.yaml")
    assert raw["verification"]["status"] == "pending"
    assert raw["hardware"]["gpu_count"] == 4
    assert raw["hardware"]["gpu_name_contains"] == "NVIDIA L40"
    assert raw["hardware"]["min_vram_gib_per_gpu"] == 40
    assert raw["hardware"]["min_system_ram_gib"] == 128
    assert raw["hardware"]["distributed_backend"] == "nccl"
    assert raw["capabilities"]["recovery_finetune"]["enabled"] is False
    assert raw["capabilities"]["prune"]["enabled"] is False
    assert raw["capabilities"]["distill"]["enabled"] is False


def test_l40x4_smoke_recipe_is_bounded_and_sharded():
    raw = load_yaml("configs/experiments/a1-t0-l40x4.yaml")
    assert raw["operator"] == "recovery_finetune"
    assert raw["distributed"] == {
        "launcher": "torchrun",
        "strategy": "fsdp_full_shard",
        "world_size": 4,
        "backend": "nccl",
        "use_orig_params": True,
    }
    assert raw["model"]["dtype"] == "bfloat16"
    assert raw["model"]["trainable_scope"] == "heads"
    assert raw["memory"]["micro_batch_size"] == 1
    assert raw["optimization"]["max_steps"] == 1
    assert raw["data"]["sample_count"] == 1
    assert raw["checkpoint"]["require_diffusers_reload"] is True
    assert raw["checkpoint"]["require_comfyui_reload"] is True


def test_l40x4_worker_uses_fixed_four_rank_linux_argv():
    raw = json.loads((ROOT / "configs/a1-worker.l40x4.example.json").read_text(encoding="utf-8"))
    command = raw["trainer_command"]
    assert command[0] == "/opt/h3-training/.venv/bin/torchrun"
    assert "--nproc_per_node=4" in command
    assert "/opt/h3-training/train_worker.py" in command
    assert "/opt/Harness4H3/configs/experiments/a1-t0-l40x4.yaml" in command
    assert not any("fixture" in item or "mock" in item for item in command)
    assert raw["deploy_model_dir"] == "/opt/ComfyUI/models/diffusion_models"
