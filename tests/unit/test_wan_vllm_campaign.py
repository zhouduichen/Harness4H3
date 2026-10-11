import json
from pathlib import Path

import pytest

from tools.wan_vllm_campaign import (
    _campaign_capabilities,
    _memory_lessons,
    build_training_command,
    extract_json_content,
    load_loop_config,
    proposal_json_schema,
)
from harness4h3.student.wan_campaign import WanTrainingProposal


def test_extract_json_content_accepts_openai_and_fenced_json():
    payload = {"choices": [{"message": {"content": "```json\n{\"steps\": 8}\n```"}}]}
    assert extract_json_content(payload) == {"steps": 8}


def test_extract_json_content_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        extract_json_content({"choices": [{"message": {"content": "[]"}}]})


def test_training_command_contains_only_trusted_paths_and_validated_knobs(tmp_path: Path):
    parent = "a" * 64
    proposal = WanTrainingProposal.from_mapping(
        {
            "schema_version": 1,
            "proposal_id": "wan_0001",
            "parent_checkpoint_sha256": parent,
            "steps": 8,
            "teacher_reuse_steps": 2,
            "learning_rate": 5e-7,
            "focus": "semantic",
            "hypothesis": "semantic",
        },
        expected_parent_sha256=parent,
    )
    command = build_training_command(
        {
            "worker_python": "/opt/python",
            "training_entrypoint": "/srv/tools/wan_student_train_worker.py",
            "training_config": "/srv/config.yaml",
        },
        proposal,
        output_dir=tmp_path / "round",
        result_path=tmp_path / "round" / "training-result.json",
        parent_checkpoint=tmp_path / "parent.safetensors",
    )
    assert command[:2] == ("/opt/python", "/srv/tools/wan_student_train_worker.py")
    assert "--steps" in command and "8" in command
    assert "--learning-rate" in command and "5e-07" in command
    assert "--endpoint-weight" in command and "1.0" in command
    assert "--parent-checkpoint" in command
    assert "--training-checkpoint" in command
    assert "--checkpoint-every-steps" in command


def test_native_training_command_binds_rollout_state_contract(tmp_path: Path):
    parent = "a" * 64
    proposal = WanTrainingProposal.from_mapping(
        {
            "schema_version": 2,
            "proposal_id": "wan_0001",
            "parent_checkpoint_sha256": parent,
            "training": {
                "method": "progressive_distillation",
                "source_steps": 32,
                "target_steps": 16,
                "train_steps": 8,
                "learning_rate": 5e-7,
                "critic_learning_rate": 1e-6,
                "endpoint_weight": 0.0,
                "velocity_weight": 0.0,
                "clean_anchor_weight": 0.0,
                "rollout_steps": 2,
                "rollout_every": 1,
                "rollout_endpoint_weight": 1.0,
                "rollout_clean_weight": 1.0,
                "trainable_scope": "all",
                "fp32_optimizer_state": False,
            },
            "initialization": {"mode": "inherit"},
            "precision": {"loss_reduction": "fp32", "optimizer_state": "bf16"},
            "curriculum": {"fidelity_schedule": ["F1", "F2", "F3"]},
            "data": {"manifest_digest": "c" * 64, "prompt_policy": "fixed_manifest", "seed_policy": "fixed"},
            "continuation": {"overlap_latent_frames": 6, "anchor_schedule": "constant"},
            "focus": "stability",
            "hypothesis": "Native rollout-state stability.",
        },
        expected_parent_sha256=parent,
    )
    command = build_training_command(
        {
            "worker_python": "/opt/python",
            "training_entrypoint": "/srv/tools/wan_student_train_worker.py",
            "training_config": "/srv/config.yaml",
            "training_mode": "native_rollout_state",
            "teacher_forced_intervals": 16,
            "trajectory_record": "/data/trajectory-record",
        },
        proposal,
        output_dir=tmp_path / "round",
        result_path=tmp_path / "round" / "training-result.json",
        parent_checkpoint=tmp_path / "parent.safetensors",
    )
    assert command[command.index("--endpoint-weight") + 1] == "0.0"
    assert command[command.index("--velocity-weight") + 1] == "0.0"
    assert command[command.index("--clean-anchor-weight") + 1] == "0.0"
    assert command[command.index("--mode") + 1] == "native_rollout_state"
    assert command[command.index("--teacher-forced-intervals") + 1] == "16"
    assert command[command.index("--trajectory-record") + 1] == "/data/trajectory-record"


def test_loop_config_requires_absolute_trusted_paths(tmp_path: Path):
    config = tmp_path / "loop.yaml"
    config.write_text(
        "wan_loop:\n"
        "  training_config: relative.yaml\n"
        "  wan_config: /srv/wan.yaml\n"
        "  evaluation_manifest: /srv/manifest.json\n"
        "  clip_model_path: /srv/clip\n"
        "  teacher_baseline_result: /srv/teacher.json\n"
        "  parent_checkpoint: /srv/parent.safetensors\n"
        "  output_root: /data/campaign\n"
        "  worker_python: /opt/python\n"
        "  training_entrypoint: /srv/worker.py\n"
        "  vllm_base_url: http://127.0.0.1:8000/v1\n"
        "  vllm_model: qwen3.5-controller\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="training_config"):
        load_loop_config(config)


def test_wan_loop_config_exposes_unattended_service_defaults():
    config = load_loop_config(Path("configs/student-campaign.wan-vllm-loop.yaml"))
    assert int(config["watchdog_restart_limit"]) >= 1
    assert int(config["watchdog_backoff_seconds"]) >= 0
    assert str(config["watchdog_stop_file"]).startswith(str(config["output_root"]))
    assert str(config["watchdog_lock_file"]).startswith(str(config["output_root"]))
    assert str(config["watchdog_child_pid_file"]).startswith(str(config["output_root"]))
    assert str(config["watchdog_log"]).startswith(str(config["output_root"]))


def test_native_5s_config_enables_visual_information_gate():
    config = load_loop_config(Path("configs/student-campaign.wan-5s-native-20261009.yaml"))
    gate = config["visual_quality_gate"]
    assert gate["enabled"] is True
    assert gate["version"] == "visual-structure-v1"
    assert gate["min_spatial_gradient"] > 0
    assert gate["max_low_texture_frame_ratio"] < 1


def test_native_rollout_state_loop_requires_fixed_trajectory(tmp_path: Path):
    config = tmp_path / "loop.yaml"
    config.write_text(
        "wan_loop:\n"
        "  training_config: /srv/training.yaml\n"
        "  wan_config: /srv/wan.yaml\n"
        "  evaluation_manifest: /srv/manifest.json\n"
        "  clip_model_path: /srv/clip\n"
        "  teacher_baseline_result: /srv/teacher.json\n"
        "  parent_checkpoint: /srv/parent.safetensors\n"
        "  output_root: /data/campaign\n"
        "  worker_python: /opt/python\n"
        "  training_entrypoint: /srv/worker.py\n"
        "  vllm_base_url: http://127.0.0.1:8000/v1\n"
        "  vllm_model: qwen3.5-controller\n"
        "  continuation_contract: 5s\n"
        "  training_mode: native_rollout_state\n"
        "  teacher_forced_intervals: 16\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trajectory_record"):
        load_loop_config(config)


def test_native_rollout_state_loop_rejects_non_native_training_cache(tmp_path: Path):
    training = tmp_path / "training.yaml"
    training.write_text("wan_training:\n  cache_contract: long_video_30s\n", encoding="utf-8")
    config = tmp_path / "loop.yaml"
    config.write_text(
        "wan_loop:\n"
        f"  training_config: {training}\n"
        "  wan_config: /srv/wan.yaml\n"
        "  evaluation_manifest: /srv/manifest.json\n"
        "  clip_model_path: /srv/clip\n"
        "  teacher_baseline_result: /srv/teacher.json\n"
        "  parent_checkpoint: /srv/parent.safetensors\n"
        "  output_root: /data/campaign\n"
        "  worker_python: /opt/python\n"
        "  training_entrypoint: /srv/worker.py\n"
        "  vllm_base_url: http://127.0.0.1:8000/v1\n"
        "  vllm_model: qwen3.5-controller\n"
        "  continuation_contract: 5s\n"
        "  training_mode: native_rollout_state\n"
        "  teacher_forced_intervals: 16\n"
        "  trajectory_record: /data/trajectory-record\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cache_contract=native_5s"):
        load_loop_config(config)


def test_visual_collapse_is_persisted_as_a_negative_memory_lesson():
    lessons = _memory_lessons(
        {"controller_memory_limit": 3, "memory_seed_paths": []},
        [
            {
                "round": 1,
                "decision": {"accepted": False, "reason": "visual_collapse"},
                "evaluation": {"quality_metrics": {"semantic": 0.2}, "quality_score": 0.3},
                "proposal": {"training": {"train_steps": 1}},
            }
        ],
    )
    assert lessons[-1]["decision"] == "visual_collapse"
    assert "Avoid repeating" in lessons[-1]["lesson"]


def test_proposal_schema_disallows_code_and_paths():
    schema = proposal_json_schema()
    assert schema["additionalProperties"] is False
    assert "shell" not in schema["properties"]
    assert "parent_checkpoint_sha256" in schema["properties"]


def test_v2_schema_contains_bounded_algorithm_policy_and_capability_snapshot():
    schema = proposal_json_schema()
    assert schema["properties"]["schema_version"]["const"] == 2
    assert "training" in schema["properties"]
    assert "method" in schema["properties"]["training"]["properties"]
    assert "source_steps" in schema["properties"]["training"]["properties"]
    assert "endpoint_weight" in schema["properties"]["training"]["required"]
    assert "initialization" in schema["properties"]
    assert "continuation" in schema["properties"]


def test_dmd2_capability_requires_adapter_and_memory_admission():
    disabled = _campaign_capabilities({"algorithm_capabilities": ["dmd2"], "dmd2_memory_admitted": True})
    dmd2 = next(item for item in disabled["capabilities"] if item["name"] == "dmd2")
    assert dmd2["available"] is False
    enabled = _campaign_capabilities(
        {
            "algorithm_capabilities": ["dmd2"],
            "dmd2_memory_admitted": True,
            "wan_dmd2_adapter_ready": True,
        }
    )
    assert next(item for item in enabled["capabilities"] if item["name"] == "dmd2")["available"] is True
