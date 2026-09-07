from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness4h3.config import ConfigError, load_config


def write_config(tmp_path: Path, workflow, mutable=None):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    mutable = mutable or {"steps": {"node_id": "sampler", "input": "steps"}}
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(
        yaml.safe_dump(
            {
                "backend": {"base_url": "http://localhost:8188"},
                "workflow": {
                    "template": "workflow.json",
                    "prompt_target": {"node_id": "prompt", "input": "text"},
                    "seed_target": {"node_id": "seed", "input": "seed"},
                    "mutable": mutable,
                },
                "runtime": {"tasks_path": "tasks.yaml"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_config_resolves_paths_relative_to_config(tmp_path, workflow):
    config = load_config(write_config(tmp_path, workflow))
    assert config.workflow.template == tmp_path / "workflow.json"
    assert config.runtime.archive_dir == tmp_path / "var/archive"


def test_config_rejects_non_allowlisted_workflow_key(tmp_path, workflow):
    path = write_config(tmp_path, workflow, {"model_name": {"node_id": "sampler", "input": "steps"}})
    with pytest.raises(ConfigError, match="unsupported mutable workflow key"):
        load_config(path)


def test_config_rejects_frontend_graph(tmp_path, workflow):
    path = write_config(tmp_path, {"nodes": [], "links": []})
    with pytest.raises(ConfigError, match="API format"):
        load_config(path)


def test_task_manifest_rejects_path_traversal_id(tmp_path):
    from harness4h3.harness.state import load_tasks

    path = tmp_path / "tasks.yaml"
    path.write_text("tasks:\n  - id: ../../escape\n    split: dev\n    prompt: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task requires"):
        load_tasks(path)
