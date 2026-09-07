from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_cli_runs_fake_comfyui_through_external_evaluator(fake_comfyui, tmp_path, workflow):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    tasks_path = tmp_path / "tasks.yaml"
    tasks_path.write_text(
        yaml.safe_dump({"tasks": [{"id": "dev-one", "split": "dev", "prompt": "move", "seed": 7}]}),
        encoding="utf-8",
    )
    evaluator_code = (
        "import json,sys; json.load(sys.stdin); "
        "print(json.dumps({'score':0.8,'metrics':{'external':1.0},'critical_regression':False,'failure_type':None}))"
    )
    config = {
        "backend": {"base_url": fake_comfyui.url, "poll_interval_s": 0, "task_timeout_s": 2},
        "workflow": {
            "template": str(workflow_path),
            "prompt_target": {"node_id": "prompt", "input": "text"},
            "seed_target": {"node_id": "seed", "input": "seed"},
            "mutable": {"steps": {"node_id": "sampler", "input": "steps"}},
        },
        "runtime": {
            "output_dir": str(tmp_path / "outputs"),
            "trajectory_path": str(tmp_path / "runs.jsonl"),
            "archive_dir": str(tmp_path / "archive"),
            "tasks_path": str(tasks_path),
        },
        "evaluator": {"command": [sys.executable, "-c", evaluator_code]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "harness4h3", "--config", str(config_path), "run", "--split", "dev", "--json"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mean_score"] == 0.8
    records = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["harness_version"] == "H0"
    assert records[0]["evaluation"]["external"] == 1.0
    assert (tmp_path / "archive/candidates/H0.json").exists()

