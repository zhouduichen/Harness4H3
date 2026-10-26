import pytest

from harness4h3.rsi.bundle import HarnessBundle
from harness4h3.rsi.budget import BudgetLedger
from harness4h3.rsi.runner import HarnessComponentRunner, ResearchJobRunner, ResearchTaskFailure
from harness4h3.rsi.trace import TraceEvent, TraceStore
from harness4h3.sandbox.runner import AlgorithmSandbox


def _bundle(tmp_path, content):
    source = tmp_path / "source"
    source.mkdir()
    (source / "context.py").write_text(content, encoding="utf-8")
    return HarnessBundle.capture(source, tmp_path / "bundles", parent_id=None, hypothesis={}, allowed_paths=["context.py"])


def test_component_runner_executes_changed_source_in_isolated_sandbox(tmp_path):
    bundle = _bundle(tmp_path, "def build_context(task, evidence): return {'task': task, 'count': len(evidence)}\n")
    runner = HarnessComponentRunner(AlgorithmSandbox(tmp_path / "sandbox"))
    result = runner.run_component(bundle, "context.py", {"task": "x", "evidence": [1, 2, 3]})
    assert result == {"task": "x", "count": 3}


def test_component_runner_rejects_forbidden_filesystem_access(tmp_path):
    bundle = _bundle(tmp_path, "def build_context(task, evidence):\n    return open('/etc/passwd').read()\n")
    runner = HarnessComponentRunner(AlgorithmSandbox(tmp_path / "sandbox"))
    with pytest.raises(ValueError, match="component_failed"):
        runner.run_component(bundle, "context.py", {"task": "x", "evidence": []})


class _Service:
    def __init__(self):
        self.collect_calls = 0
        self.evaluate_calls = 0

    def collect(self, task_id, bundle_id, seed):
        self.collect_calls += 1
        if self.collect_calls == 1:
            raise RuntimeError("resource unavailable")
        return {"artifact_ref": "artifact:1"}

    def evaluate(self, task_id, bundle_id, artifact_ref):
        self.evaluate_calls += 1
        return {"score": 0.5}


def test_research_job_retry_is_charged_and_completed_job_replays(tmp_path):
    bundle = _bundle(tmp_path, "def build_context(task, evidence): return task\n")
    service = _Service()
    traces = TraceStore(tmp_path / "traces")
    budget = BudgetLedger(tmp_path / "budget.json", {"job_count": 2})
    runner = ResearchJobRunner(service=service, bundle_store=tmp_path / "bundles", traces=traces, budget=budget)
    args = dict(job_id="job-1", run_id="run-1", task_id="task-1", bundle_id=bundle.bundle_id, seed=3)
    with pytest.raises(RuntimeError, match="resource unavailable"):
        runner.run(**args)
    completed = runner.run(**args)
    assert runner.run(**args) == completed
    assert service.collect_calls == 2
    assert service.evaluate_calls == 1
    ledger = budget.snapshot()
    assert len(ledger["jobs"]) == 2
    assert sum(job["actual"]["job_count"] for job in ledger["jobs"].values()) == 2
    with pytest.raises(ValueError, match="identity_conflict"):
        runner.run(**{**args, "seed": 4})


def test_task_failure_class_is_preserved_in_trace(tmp_path):
    bundle = _bundle(tmp_path, "def build_context(task, evidence): return task\n")

    class ResourceFailureService(_Service):
        def collect(self, task_id, bundle_id, seed):
            raise ResearchTaskFailure("resource_limited", "fixture OOM")

    traces = TraceStore(tmp_path / "traces")
    runner = ResearchJobRunner(
        service=ResourceFailureService(),
        bundle_store=tmp_path / "bundles",
        traces=traces,
        budget=BudgetLedger(tmp_path / "budget.json", {"job_count": 1}),
    )
    with pytest.raises(ResearchTaskFailure):
        runner.run(job_id="oom", run_id="r", task_id="t", bundle_id=bundle.bundle_id, seed=1)
    assert traces.read("r")[0].failure_class == "resource_limited"


def test_resume_reconciles_completion_written_before_budget_settlement(tmp_path):
    bundle = _bundle(tmp_path, "def build_context(task, evidence): return task\n")
    traces = TraceStore(tmp_path / "traces")
    budget = BudgetLedger(tmp_path / "budget.json", {"job_count": 1})
    assert budget.reserve("job#attempt-1", {"job_count": 1})
    result = {"artifact_ref": "artifact:crash-window", "result": {"score": 1}, "seed": 2, "job_id": "job"}
    traces.append(TraceEvent.create(
        run_id="run", task_id="task", bundle_id=bundle.bundle_id, stage="research", event="job_complete", status="success",
        started_ns=1, ended_ns=2,
        payload={"job_id": "job", "attempt_id": "job#attempt-1", "seed": 2, "result": result,
                 "input_tokens": None, "output_tokens": None, "optimizer_updates": None, "gpu_lease_seconds": None},
    ))

    class NeverCallService(_Service):
        def collect(self, *args):
            raise AssertionError("completed job must not be re-executed")

    runner = ResearchJobRunner(service=NeverCallService(), bundle_store=tmp_path / "bundles", traces=traces, budget=budget)
    assert runner.run(job_id="job", run_id="run", task_id="task", bundle_id=bundle.bundle_id, seed=2) == result
    assert budget.snapshot()["jobs"]["job#attempt-1"]["actual"] == {"job_count": 1.0}
