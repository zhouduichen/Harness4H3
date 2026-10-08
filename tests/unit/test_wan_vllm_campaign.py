import json
from pathlib import Path

import pytest

from tools.wan_vllm_campaign import (
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
    assert "--parent-checkpoint" in command


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


def test_proposal_schema_disallows_code_and_paths():
    schema = proposal_json_schema()
    assert schema["additionalProperties"] is False
    assert "shell" not in schema["properties"]
    assert "parent_checkpoint_sha256" in schema["properties"]
