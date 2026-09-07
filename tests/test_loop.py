from __future__ import annotations

from pathlib import Path

from harness4h3.archive.store import Candidate, default_policy
from harness4h3.config import Target, WorkflowConfig
from harness4h3.evaluator.evaluator import EvaluationResult
from harness4h3.harness.loop import HarnessRunner
from harness4h3.harness.state import Task
from harness4h3.memory.trajectory import TrajectoryStore
from harness4h3.model.minimax_h3 import BackendError, BackendResult


class Backend:
    def __init__(self, artifact):
        self.artifact = artifact

    def run(self, workflow, output_dir):
        self.artifact.write_bytes(b"video")
        return BackendResult("p1", (self.artifact,), {}, 0.1)


class FailingBackend:
    def run(self, workflow, output_dir):
        raise BackendError("boom", "backend_execution")


class Evaluator:
    def evaluate(self, request):
        return EvaluationResult(0.8, {"quality": 0.8})


def make_runner(tmp_path, workflow, backend):
    config = WorkflowConfig(tmp_path / "w.json", Target("prompt", "text"), Target("seed", "seed"), {})
    return HarnessRunner(workflow, config, backend, Evaluator(), TrajectoryStore(tmp_path / "runs.jsonl"), tmp_path / "outputs")


def test_loop_records_success_and_external_score(tmp_path, workflow):
    artifact = tmp_path / "result.mp4"
    result = make_runner(tmp_path, workflow, Backend(artifact)).run_task(
        Task("t", "move", "dev"), Candidate("H0", None, 0, "baseline", {}, "", default_policy())
    )
    assert result.score == 0.8
    assert list(TrajectoryStore(tmp_path / "runs.jsonl").read())[0].task_id == "t"


def test_loop_records_backend_failure(tmp_path, workflow):
    result = make_runner(tmp_path, workflow, FailingBackend()).run_task(
        Task("t", "move", "dev"), Candidate("H0", None, 0, "baseline", {}, "", default_policy())
    )
    assert result.failure_type == "backend_execution"
    assert result.critical_regression is True

