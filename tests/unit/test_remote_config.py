import json
from pathlib import Path

from harness4h3.remote.config import load_remote_campaign_config


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

