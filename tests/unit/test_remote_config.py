import json
from pathlib import Path

import pytest

from harness4h3.remote.config import RemoteConfigError, load_remote_campaign_config


def test_remote_workflow_has_linux_model_and_benchmark_targets():
    workflow = json.loads(Path("examples/remote_linux_h3_workflow_api.json").read_text())
    assert workflow["135"]["inputs"]["unet_name"] == "minimax_h3_fl2va_bf16.safetensors"
    assert workflow["136"]["inputs"]["clip_name"] == "qwen3vl_32b_minimax_h3_bf16.safetensors"
    assert workflow["132"]["inputs"]["steps"] == 32
    assert workflow["139"]["inputs"]["length"] == 22


def test_remote_config_validates_host_roots_and_reward_weights():
    config = load_remote_campaign_config(Path("configs/remote-l40-h3.yaml"))
    assert config.remote.host == "Jiayu-intern"
    assert config.reward.alpha > 0
    assert config.remote.model_root == "/data/models/MiniMax-H3"
    assert {"prune_blocks", "distill", "recovery_finetune", "step_distill", "dmd2", "quantize"}.issubset(
        set(config.worker.allowed_operators)
    )
    assert config.worker.operator_launchers["dmd2"] == "torchrun"
    assert config.worker.operator_launchers["quantize"] == "python"
    assert config.worker.quantized_bits == (8,)


def test_remote_config_has_continuous_review_defaults():
    overnight = load_remote_campaign_config(Path("configs/remote-l40-h3-rsi-overnight.yaml"))
    assert overnight.review_interval_s == 60.0
    assert overnight.max_review_calls == 512
    assert overnight.max_context_observations == 24
    assert overnight.controller_max_iterations == 64
    assert overnight.benchmark_task_timeout_s == 1200.0
    assert overnight.resource_wait_replan_after == 3

    base = load_remote_campaign_config(Path("configs/remote-l40-h3.yaml"))
    assert base.review_interval_s == 60.0
    assert base.max_review_calls == 120
    assert base.controller_max_iterations == 64
    assert overnight.checkpoint_retention_policy == "rejected_candidate_v1"
    assert overnight.max_retained_checkpoints == 3
    assert {"prune_blocks", "distill", "recovery_finetune", "step_distill", "dmd2", "quantize"}.issubset(
        set(overnight.worker.allowed_operators)
    )
    assert overnight.worker.operator_launchers["prune_blocks"] == "python"
    assert overnight.worker.operator_launchers["quantize"] == "python"
    assert [worker.gpu_index for worker in overnight.comfyui_workers] == [0, 1, 2, 3]


def test_overnight_config_enables_pipeline_and_on_demand_comfyui():
    config = load_remote_campaign_config(Path("configs/remote-l40-h3-rsi-overnight.yaml"))
    assert config.pipeline_enabled is True
    assert config.pipeline_max_inflight == 2
    assert config.controller_overlap_gpus == 1
    assert config.prefetch_before_full_training is True
    assert config.comfyui_process_policy == "on_demand"
    assert config.comfyui_idle_shutdown_s > 0
    assert config.comfyui_lease_max_age_s > config.comfyui_idle_shutdown_s
    assert config.power_target_w == 300.0


def test_remote_config_defaults_to_idle_comfyui_cache_release():
    config = load_remote_campaign_config(Path("configs/remote-l40-h3.yaml"))
    assert config.comfyui_cache_policy == "idle_release"


def test_remote_config_rejects_unknown_comfyui_cache_policy(tmp_path):
    source = Path("configs/remote-l40-h3.yaml").read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(
        source.replace(
            "  template: ../examples/remote_linux_h3_workflow_api.json\n",
            "  template: %s\n" % Path("examples/remote_linux_h3_workflow_api.json").resolve(),
        ).replace(
            "  comfyui_cache_policy: idle_release\n",
            "  comfyui_cache_policy: invalid\n",
        )
    )
    with pytest.raises(RemoteConfigError, match="comfyui_cache_policy"):
        load_remote_campaign_config(path)


def test_remote_config_rejects_nonpositive_checkpoint_retention_cap(tmp_path):
    source = Path("configs/remote-l40-h3.yaml").read_text()
    path = tmp_path / "bad-retention.yaml"
    path.write_text(
        source.replace(
            "  template: ../examples/remote_linux_h3_workflow_api.json\n",
            "  template: %s\n" % Path("examples/remote_linux_h3_workflow_api.json").resolve(),
        ).replace("  max_retained_checkpoints: 3\n", "  max_retained_checkpoints: 0\n")
    )
    with pytest.raises(RemoteConfigError, match="max_retained_checkpoints"):
        load_remote_campaign_config(path)
