import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from harness4h3.h3.state import ModelState
from h3_training.tiny.factory import create_tiny_checkpoint, load_tiny_checkpoint


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools" / "tiny_training_worker.py"


def request(tmp_path, operator, operator_args, nfe=4):
    parent_path = create_tiny_checkpoint(tmp_path / "M0000.pt", "M0000", sampling_nfe=nfe, seed=4)
    state = ModelState(
        model_id="M0000",
        parent_model_id=None,
        checkpoint_path=str(parent_path),
        architecture_name="TinyH3",
        sampling_steps=nfe,
        provenance={"kind": "tiny_reference", "offline_simulation": False},
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    payload = {
        "operator": operator,
        "parent": {"id": "M0000", "checkpoint_path": str(parent_path), "state": state.to_dict()},
        "child_model_id": "M0001",
        "operator_args": operator_args,
        "artifacts_dir": str(artifacts),
    }
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    return parent_path, request_path, result_path


@pytest.mark.parametrize(
    ("operator", "operator_args", "expected_nfe"),
    [("recovery_finetune", {"training_steps": 3}, 4), ("step_distill", {"target_steps": 2}, 2)],
)
def test_tiny_worker_returns_real_child_and_metrics(tmp_path, operator, operator_args, expected_nfe):
    parent_path, request_path, result_path = request(tmp_path, operator, operator_args)
    parent_bytes = parent_path.read_bytes()
    completed = subprocess.run(
        [sys.executable, str(WORKER), "--request", str(request_path), "--result", str(result_path), "--max-training-steps", "3"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + result_path.read_text(encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    child_path = Path(result["output_state"]["checkpoint_path"])
    _, metadata = load_tiny_checkpoint(child_path)
    assert result["status"] == "success"
    assert result["metrics"]["real_worker"] is True
    assert result["metrics"]["optimizer_steps"] > 0
    assert result["metrics"]["parent_sha256"] != result["metrics"]["child_sha256"]
    assert result["metrics"]["child_reloaded"] is True
    assert metadata["sampling_nfe"] == expected_nfe
    assert parent_path.read_bytes() == parent_bytes


def test_tiny_worker_rejects_non_binary_distillation(tmp_path):
    _, request_path, result_path = request(tmp_path, "step_distill", {"target_steps": 3})
    completed = subprocess.run(
        [sys.executable, str(WORKER), "--request", str(request_path), "--result", str(result_path)],
        cwd=tmp_path,
        check=False,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert completed.returncode != 0
    assert result["failure_type"] == "invalid_training_config"


def test_tiny_worker_rejects_unsupported_operator(tmp_path):
    _, request_path, result_path = request(tmp_path, "dmd2", {"training_steps": 1})
    completed = subprocess.run(
        [sys.executable, str(WORKER), "--request", str(request_path), "--result", str(result_path)],
        cwd=tmp_path,
        check=False,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert completed.returncode != 0
    assert result["failure_type"] == "unsupported_training_operator"
