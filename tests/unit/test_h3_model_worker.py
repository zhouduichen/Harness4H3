from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.controller.schemas import CostEstimate
from harness4h3.executor.local import LocalProcessExecutor
from harness4h3.h3.state import ModelState
from harness4h3.operators.base import ExecutionContext
from harness4h3.operators.external import ExternalScriptOperator
from harness4h3.target.profile import TargetProfile


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools" / "h3_model_worker.py"
TRAINER = ROOT / "tests" / "fixtures" / "h3_model_trainer_fixture.py"
INVALID_TRAINER = ROOT / "tests" / "fixtures" / "h3_model_trainer_invalid.py"


def _request(tmp_path: Path) -> Tuple[Path, Path]:
    parent_path = tmp_path / "parent.safetensors"
    parent_path.write_bytes(b"immutable H3 parent")
    state = ModelState.from_dict({**ModelState.fake_baseline().to_dict(), "checkpoint_path": str(parent_path)})
    parent = ModelCandidate("M0000", None, 0, str(parent_path), state, None, "baseline")
    experiment = tmp_path / "experiment"
    artifacts = experiment / "artifacts"
    artifacts.mkdir(parents=True)
    request_path = experiment / "request.json"
    result_path = experiment / "result.json"
    request_path.write_text(
        json.dumps(
            {
                "operator": "create_student",
                "parent": parent.to_dict(),
                "child_model_id": "M0001",
                "operator_args": {"width_ratio": 0.5, "block_ratio": 0.5},
                "artifacts_dir": str(artifacts),
            }
        ),
        encoding="utf-8",
    )
    return request_path, result_path


def _config(path: Path, trainer: Path, deploy: Optional[Path] = None) -> None:
    path.write_text(
        json.dumps(
            {
                "trainer_command": [sys.executable, str(trainer)],
                "trainer_timeout_s": 5,
                "deploy_model_dir": str(deploy) if deploy else None,
            }
        ),
        encoding="utf-8",
    )


def test_worker_stages_child_deploys_it_and_preserves_parent(tmp_path):
    request_path, result_path = _request(tmp_path)
    config_path = tmp_path / "worker.json"
    deploy = tmp_path / "comfy-models"
    _config(config_path, TRAINER, deploy)
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(WORKER), "--config", str(config_path), "--request", request_path.name, "--result", result_path.name],
        cwd=request_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    child = Path(result["output_state"]["checkpoint_path"])
    assert result["status"] == "success"
    assert child.is_file()
    assert child.parent == (request_path.parent / "artifacts").resolve()
    assert Path(result["metrics"]["deployment"]["path"]).is_file()
    assert (tmp_path / "parent.safetensors").read_bytes() == b"immutable H3 parent"
    assert result["metrics"]["real_worker"] is True
    assert result["metrics"]["offline_simulation"] is False


def test_worker_rejects_invalid_trainer_result(tmp_path):
    request_path, result_path = _request(tmp_path)
    config_path = tmp_path / "worker.json"
    _config(config_path, INVALID_TRAINER)
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(WORKER), "--config", str(config_path), "--request", request_path.name, "--result", result_path.name],
        cwd=request_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["failure_type"] == "invalid_trainer_result"


def test_worker_runs_through_external_operator_contract(tmp_path):
    parent_path = tmp_path / "parent.safetensors"
    parent_path.write_bytes(b"immutable H3 parent")
    state = ModelState.from_dict({**ModelState.fake_baseline().to_dict(), "checkpoint_path": str(parent_path)})
    parent = ModelCandidate("M0000", None, 0, str(parent_path), state, None, "baseline")
    config_path = tmp_path / "worker.json"
    _config(config_path, TRAINER)
    operator = ExternalScriptOperator(
        "create_student",
        "fixture real worker",
        (sys.executable, str(WORKER), "--config", str(config_path)),
        {"width_ratio": (float,), "block_ratio": (float,)},
        LocalProcessExecutor(timeout_s=10),
        CostEstimate(wall_time_s=1.0, gpu_hours=0.01),
        timeout_s=10,
    )
    result = operator.execute(
        parent,
        {"width_ratio": 0.5, "block_ratio": 0.5},
        ExecutionContext(tmp_path / "operator-run", "M0001"),
    )
    assert result.ok
    assert result.output_state is not None
    assert result.output_state.model_id == "M0001"
    assert result.metrics["real_worker"] is True
    assert result.metrics["offline_simulation"] is False
