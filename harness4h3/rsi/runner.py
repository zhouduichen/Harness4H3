from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..sandbox.runner import AlgorithmSandbox
from .budget import BudgetLedger
from .bundle import HarnessBundle
from .trace import TraceEvent, TraceStore


class ResearchTaskFailure(RuntimeError):
    """A task-service failure with an explicit, auditable experiment category."""

    ALLOWED = {"technical", "resource_limited", "method_negative", "protocol_violation"}

    def __init__(self, failure_class: str, message: str) -> None:
        if failure_class not in self.ALLOWED:
            raise ValueError("research_failure_class_invalid")
        super().__init__(message)
        self.failure_class = failure_class


class ResearchTaskService(Protocol):
    def collect(self, task_id: str, bundle_id: str, seed: int) -> Mapping[str, Any]: ...
    def evaluate(self, task_id: str, bundle_id: str, artifact_ref: str) -> Mapping[str, Any]: ...


class HarnessComponentRunner:
    """Run one declared component with the repository's audited source sandbox."""

    COMPONENTS = frozenset({"context.py", "retrieval.py", "strategy.py"})
    ENTRYPOINTS = {"context.py": "build_context", "retrieval.py": "retrieve", "strategy.py": "choose_experiment"}

    def __init__(self, sandbox: AlgorithmSandbox) -> None:
        self.sandbox = sandbox

    def run_component(self, bundle: HarnessBundle, component: str, input_payload: Mapping[str, Any]) -> Any:
        if component not in self.COMPONENTS or component not in bundle.files:
            raise ValueError("harness_component_not_allowlisted")
        result = self.sandbox.run_program(bundle.files[component], entrypoint=self.ENTRYPOINTS[component], payload=dict(input_payload))
        if result.status != "success":
            raise ValueError("harness_component_failed:%s:%s" % (result.failure_type, result.message))
        try:
            return json.loads(json.dumps(result.value, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("harness_component_output_not_json") from exc


class ResearchJobRunner:
    """Run a trusted task service and make completed jobs idempotent."""

    def __init__(self, *, service: ResearchTaskService, bundle_store: Path, traces: TraceStore, budget: BudgetLedger) -> None:
        self.service, self.bundle_store, self.traces, self.budget = service, Path(bundle_store), traces, budget

    def run(self, *, job_id: str, run_id: str, task_id: str, bundle_id: str, seed: int) -> Mapping[str, Any]:
        bundle = HarnessBundle.load(self.bundle_store, bundle_id)
        events = self.traces.read(run_id)
        attempts = 0
        for item in events:
            if item.event in {"job_complete", "job_failed"} and item.payload.get("job_id") == job_id:
                attempts += 1
            if item.event == "job_complete" and item.payload.get("job_id") == job_id:
                if (item.task_id, item.bundle_id, item.payload.get("seed")) != (task_id, bundle_id, seed):
                    raise ValueError("research_job_identity_conflict")
                attempt_id = item.payload.get("attempt_id")
                if attempt_id:
                    recorded = self.budget.snapshot()["jobs"].get(attempt_id)
                    if recorded is None:
                        raise ValueError("research_job_budget_evidence_missing")
                    if recorded.get("actual") is None:
                        self.budget.finalize(attempt_id, {"job_count": 1})
                return dict(item.payload["result"])
        attempt_id = "%s#attempt-%d" % (job_id, attempts + 1)
        if not self.budget.reserve(attempt_id, {"job_count": 1}):
            raise ValueError("research_budget_exhausted")
        started = time.monotonic_ns()
        try:
            result = dict(self.service.collect(task_id, bundle.bundle_id, int(seed)))
            if not result.get("artifact_ref"):
                raise ValueError("research_task_missing_artifact_ref")
            evaluated = dict(self.service.evaluate(task_id, bundle.bundle_id, str(result["artifact_ref"])))
            output = {"artifact_ref": str(result["artifact_ref"]), "result": evaluated, "seed": int(seed), "job_id": job_id}
            ended = time.monotonic_ns()
            self.traces.append(TraceEvent.create(run_id=run_id, task_id=task_id, bundle_id=bundle_id, stage="research", event="job_complete", status="success", started_ns=started, ended_ns=ended, payload={"job_id": job_id, "attempt_id": attempt_id, "seed": int(seed), "result": output, "input_tokens": None, "output_tokens": None, "optimizer_updates": None, "gpu_lease_seconds": None}, evidence_refs=(str(result["artifact_ref"]),)))
            self.budget.finalize(attempt_id, {"job_count": 1})
            return output
        except Exception as exc:
            ended = time.monotonic_ns()
            failure_class = exc.failure_class if isinstance(exc, ResearchTaskFailure) else "technical"
            self.traces.append(TraceEvent.create(run_id=run_id, task_id=task_id, bundle_id=bundle_id, stage="research", event="job_failed", status="failed", started_ns=started, ended_ns=ended, payload={"job_id": job_id, "attempt_id": attempt_id, "error": type(exc).__name__ + ": " + str(exc), "input_tokens": None, "output_tokens": None, "optimizer_updates": None, "gpu_lease_seconds": None}, failure_class=failure_class))
            self.budget.finalize(attempt_id, {"job_count": 1})
            raise
