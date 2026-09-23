from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness4h3.student.config import StudentConfigError, load_student_campaign_config


def _write_config(tmp_path: Path, **student_overrides):
    root = {
        "student": {
            "goal": "test goal",
            "teacher_checkpoint": "/srv/models/teacher.safetensors",
            "h3_cache_dir": "/srv/models/cache",
            "worker_entrypoint": "/srv/harness/tools/student_train_worker.py",
            "worker_python": "/opt/venv/bin/python",
            "remote_campaign_root": "/srv/harness/work/student",
            "controller": {"provider": "ollama", "model": "local", "base_url": "http://127.0.0.1:11434"},
            "evaluation_command": ["/opt/venv/bin/python", "/srv/harness/tools/student_evaluate_worker.py"],
            **student_overrides,
        },
        "remote": {
            "host": "test-host",
            "ssh_port": 22,
            "harness_root": "/srv/harness",
            "model_root": "/srv/models",
            "comfyui_root": "/srv/comfy",
            "comfyui_port": 8188,
        },
    }
    path = tmp_path / "student.yaml"
    path.write_text(yaml.safe_dump(root), encoding="utf-8")
    return path


def test_student_campaign_config_requires_fixed_worker_and_real_evaluator(tmp_path):
    config = load_student_campaign_config(_write_config(tmp_path))
    assert config.worker_entrypoint.endswith("student_train_worker.py")
    assert config.worker_device == "auto"
    assert config.worker_gpu_wait_s == 1800
    assert config.worker_min_free_memory_gb == 44.3
    assert config.worker_student_min_free_memory_gb == 20.0
    assert config.worker_student_memory_safety_margin_gb == 2.0
    assert config.target.min_params == 1_000_000_000
    with pytest.raises(StudentConfigError, match="worker_entrypoint"):
        load_student_campaign_config(_write_config(tmp_path, worker_entrypoint="relative.py"))


def test_student_campaign_example_config_loads_with_shared_model_root():
    config_path = Path(__file__).parents[2] / "configs" / "student-campaign.example.yaml"
    config = load_student_campaign_config(config_path)
    assert config.remote.model_root == "/data/models"
    assert config.clip_model_path == "/data/models/clip-vit-large-patch14-336"
    assert config.target_device is not None


def test_remote_paths_cannot_escape_configured_roots(tmp_path):
    with pytest.raises(StudentConfigError, match="escapes"):
        load_student_campaign_config(_write_config(tmp_path, teacher_checkpoint="/tmp/teacher.safetensors"))


def test_worker_device_must_be_a_cuda_index_or_auto(tmp_path):
    with pytest.raises(StudentConfigError, match="worker_device"):
        load_student_campaign_config(_write_config(tmp_path, worker_device="cuda:bad"))


def test_clip_quality_backend_requires_local_model_path(tmp_path):
    with pytest.raises(StudentConfigError, match="clip_model_path"):
        load_student_campaign_config(_write_config(tmp_path, quality_backend="clip_temporal"))
    config = load_student_campaign_config(
        _write_config(tmp_path, quality_backend="clip_temporal", clip_model_path="/srv/models/clip")
    )
    assert config.evaluation_manifest == "/srv/harness/work/student/evaluation-manifest.json"


def test_target_device_and_review_models_are_explicitly_configurable(tmp_path):
    config = load_student_campaign_config(
        _write_config(
            tmp_path,
            target_device_command=["/opt/edge/run"],
            target_device_id="phone-a",
            advocate_model="advocate-v1",
            critical_model="critical-v1",
            revision_model="revision-v1",
        )
    )
    assert config.target_device_command == ("/opt/edge/run",)
    assert config.target_device_id == "phone-a"
    assert (config.advocate_model, config.critical_model, config.revision_model) == (
        "advocate-v1",
        "critical-v1",
        "revision-v1",
    )


def test_target_device_profile_is_the_primary_identity_with_command(tmp_path):
    config = load_student_campaign_config(
        _write_config(
            tmp_path,
            target_device_command=["/opt/edge/run"],
            target_device={
                "id": "phone-a",
                "runtime_backend": "fake-runtime",
                "max_latency_s": 1.0,
                "max_memory_gb": 1.0,
                "max_energy_j": 1.0,
                "max_thermal_c": 70.0,
                "max_model_size_gb": 1.0,
                "supported_precision": ["bf16"],
                "supported_quantization": ["none"],
                "resolution": [512, 512],
                "frames": 5,
                "sampling_steps": 1,
            },
        )
    )
    assert config.target_device is not None
    assert config.target_device.id == "phone-a"
    assert config.target_device_id == "phone-a"


def test_target_device_command_requires_device_identity(tmp_path):
    with pytest.raises(StudentConfigError, match="configured together"):
        load_student_campaign_config(_write_config(tmp_path, target_device_command=["/opt/edge/run"]))
