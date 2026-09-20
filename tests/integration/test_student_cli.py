from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from tests.unit.test_student_proposal import valid_payload


ROOT = Path(__file__).resolve().parents[2]


def write_config(tmp_path: Path) -> Path:
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
        },
        "remote": {
            "host": "test-host",
            "harness_root": "/srv/harness",
            "model_root": "/srv/models",
            "comfyui_root": "/srv/comfy",
        },
    }
    path = tmp_path / "student.yaml"
    path.write_text(yaml.safe_dump(root), encoding="utf-8")
    return path


def test_validate_and_compile_cli_are_offline(tmp_path):
    config = write_config(tmp_path)
    validate = subprocess.run(
        [sys.executable, "-m", "harness4h3", "student-campaign", "--student-config", str(config), "validate", "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    assert json.loads(validate.stdout)["valid"] is True

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps(valid_payload()), encoding="utf-8")
    compile_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "harness4h3",
            "student-campaign",
            "--student-config",
            str(config),
            "compile",
            "--proposal",
            str(proposal),
            "--output",
            str(tmp_path / "compile"),
            "--json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    assert json.loads(compile_result.stdout)["graph_status"] == "compiled"


def test_detach_is_not_claimed_before_supervisor_exists(tmp_path):
    config = write_config(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "harness4h3", "student-campaign", "--student-config", str(config), "run", "--detach", "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "remote supervisor" in json.loads(result.stdout)["error"]
