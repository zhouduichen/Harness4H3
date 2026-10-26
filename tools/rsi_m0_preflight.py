#!/usr/bin/env python3
"""Offline engineering gate for the RSI harness foundation (no Wan/model access)."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness4h3.rsi.bundle import HarnessBundle
from harness4h3.rsi.budget import BudgetLedger
from harness4h3.rsi.runner import HarnessComponentRunner, ResearchJobRunner
from harness4h3.rsi.trace import CostLedger, TraceStore
from harness4h3.sandbox.runner import AlgorithmSandbox


REQUIRED_COMPONENTS = {
    "context": ("context.py", "build_context"),
    "retrieval": ("retrieval.py", "retrieve"),
    "strategy": ("strategy.py", "choose_experiment"),
}
FORBIDDEN_COMPONENTS = {"evaluator", "budget", "dataset_split", "task_service", "acceptance_policy"}


class _OfflineTaskService:
    """Deterministic fixture; deliberately has no network/model/GPU implementation."""

    def __init__(self) -> None:
        self.collect_calls = 0
        self.evaluate_calls = 0

    def collect(self, task_id: str, bundle_id: str, seed: int) -> Mapping[str, Any]:
        self.collect_calls += 1
        if self.collect_calls == 1:
            raise RuntimeError("fixture_interruption_before_artifact")
        return {"artifact_ref": "fixture://%s/%s/%d" % (task_id, bundle_id[:12], seed)}

    def evaluate(self, task_id: str, bundle_id: str, artifact_ref: str) -> Mapping[str, Any]:
        self.evaluate_calls += 1
        return {"score": 1.0, "status": "fixture_pass"}


def _load_config(config_path: Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    config_path = Path(config_path).resolve()
    repo_root = config_path.parents[2]
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("m0_config_schema_invalid")
    mapping = raw.get("editable_components")
    entrypoints = raw.get("component_entrypoints")
    if not isinstance(mapping, dict) or set(mapping) != set(REQUIRED_COMPONENTS):
        raise ValueError("m0_editable_component_set_invalid")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("m0_editable_component_duplicate")
    if not isinstance(entrypoints, dict) or set(entrypoints) != set(REQUIRED_COMPONENTS):
        raise ValueError("m0_component_entrypoint_set_invalid")
    if set(raw.get("forbidden_candidate_changes", ())) != FORBIDDEN_COMPONENTS:
        raise ValueError("m0_forbidden_boundary_invalid")
    paths: dict[str, Path] = {}
    for name, (filename, entrypoint) in REQUIRED_COMPONENTS.items():
        if Path(mapping[name]).name != filename or entrypoints[name] != entrypoint:
            raise ValueError("m0_component_contract_invalid:%s" % name)
        path = (repo_root / mapping[name]).resolve()
        if repo_root not in path.parents or not path.is_file():
            raise ValueError("m0_component_path_invalid:%s" % name)
        paths[name] = path
    limits = raw.get("limits")
    required_limits = {"component_timeout_seconds", "component_output_bytes", "research_jobs"}
    if not isinstance(limits, dict) or set(limits) != required_limits:
        raise ValueError("m0_limits_invalid")
    if limits["component_timeout_seconds"] <= 0 or limits["component_output_bytes"] <= 0 or limits["research_jobs"] < 2:
        raise ValueError("m0_limits_too_small_for_preflight")
    return repo_root, raw, paths


def preflight(config_path: Path, bundle_store: Path, trace_root: Path) -> dict[str, Any]:
    repo_root, config, source_paths = _load_config(config_path)
    relative_paths = {name: path.relative_to(repo_root).as_posix() for name, path in source_paths.items()}
    with tempfile.TemporaryDirectory(prefix="rsi-m0-source-") as temporary_source:
        source_root = Path(temporary_source)
        for name, path in source_paths.items():
            target_name = REQUIRED_COMPONENTS[name][0]
            (source_root / target_name).write_bytes(path.read_bytes())
        bundle = HarnessBundle.capture(
            source_root,
            Path(bundle_store),
            parent_id=None,
            hypothesis={"kind": "m0-frozen-seed", "source_paths": relative_paths},
            allowed_paths=[item[0] for item in REQUIRED_COMPONENTS.values()],
        )
    loaded = HarnessBundle.load(Path(bundle_store), bundle.bundle_id)
    if set(loaded.files) != {item[0] for item in REQUIRED_COMPONENTS.values()}:
        raise ValueError("m0_bundle_contains_undeclared_component")

    sandbox_root = (Path(trace_root).parent / "sandbox").resolve()
    sandbox = AlgorithmSandbox(sandbox_root, timeout_s=float(config["limits"]["component_timeout_seconds"]), max_output_bytes=int(config["limits"]["component_output_bytes"]))
    component_runner = HarnessComponentRunner(sandbox)
    fixtures = {
        "context.py": {"task": {"id": "fixture-task"}, "evidence": [{"n": i} for i in range(10)]},
        "retrieval.py": {"evidence": [{"id": "a"}, {"id": "b"}], "query": "fixture"},
        "strategy.py": {"context": {"task": "fixture"}, "candidates": ["candidate-1", "candidate-2"]},
    }
    component_results = {}
    for component, payload in fixtures.items():
        value = component_runner.run_component(loaded, component, payload)
        json.dumps(value, allow_nan=False)
        component_results[component] = {"passed": True, "output_sha256": hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}

    traces = TraceStore(Path(trace_root))
    budget = BudgetLedger(Path(trace_root).parent / "budget.json", {"job_count": float(config["limits"]["research_jobs"])})
    service = _OfflineTaskService()
    jobs = ResearchJobRunner(service=service, bundle_store=Path(bundle_store), traces=traces, budget=budget)
    run_id, job_id, task_id, seed = "m0-offline", "fixture-job", "fixture-task", 17
    existing = traces.read(run_id)
    existing_complete = next((item for item in existing if item.event == "job_complete" and item.payload.get("job_id") == job_id), None)
    failure_seen = any(item.event == "job_failed" and item.payload.get("job_id") == job_id for item in existing)
    if existing_complete is None:
        if not failure_seen:
            try:
                jobs.run(job_id=job_id, run_id=run_id, task_id=task_id, bundle_id=bundle.bundle_id, seed=seed)
            except RuntimeError as exc:
                failure_seen = "fixture_interruption" in str(exc)
            if not failure_seen:
                raise ValueError("m0_failure_fixture_did_not_fail")
        successful = jobs.run(job_id=job_id, run_id=run_id, task_id=task_id, bundle_id=bundle.bundle_id, seed=seed)
    else:
        successful = dict(existing_complete.payload["result"])
    replayed = jobs.run(job_id=job_id, run_id=run_id, task_id=task_id, bundle_id=bundle.bundle_id, seed=seed)
    complete_count = sum(item.event == "job_complete" and item.payload.get("job_id") == job_id for item in traces.read(run_id))
    if successful != replayed or complete_count != 1 or (existing_complete is None and (service.collect_calls != 2 or service.evaluate_calls != 1)):
        raise ValueError("m0_resume_or_idempotency_failed")
    try:
        jobs.run(job_id=job_id, run_id=run_id, task_id=task_id, bundle_id=bundle.bundle_id, seed=seed + 1)
    except ValueError as exc:
        identity_rejected = "identity_conflict" in str(exc)
    else:
        identity_rejected = False
    if not identity_rejected:
        raise ValueError("m0_identity_conflict_not_rejected")

    events = traces.read(run_id)
    budget_snapshot = budget.snapshot()
    actual_jobs = sum((record.get("actual") or {}).get("job_count", 0) for record in budget_snapshot["jobs"].values())
    cost = CostLedger.summary(events)
    report = {
        "schema_version": 1,
        "passed": True,
        "gate": "M0_offline_engineering_only",
        "blockers": [],
        "source_manifest": {relative_paths[name]: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()},
        "bundle": {"bundle_id": bundle.bundle_id, "file_paths": sorted(bundle.files), "file_count": len(bundle.files)},
        "component_fixtures": component_results,
        "research_job_fixture": {"failure_charged": failure_seen, "completed_once": complete_count == 1, "duplicate_replayed": successful == replayed, "identity_conflict_rejected": identity_rejected, "collect_calls_this_invocation": service.collect_calls, "evaluate_calls_this_invocation": service.evaluate_calls, "reused_existing_completion": existing_complete is not None},
        "budget": {"job_count_actual": actual_jobs, "job_count_limit": budget_snapshot["limits"]["job_count"], "job_attempts": len(budget_snapshot["jobs"]), "ledger_sha256": hashlib.sha256(Path(budget.path).read_bytes()).hexdigest()},
        "trace": {"event_count": len(events), "failure_classes": [event.failure_class for event in events], "cost_summary": cost},
        "execution_boundary": {"network": "not_used", "model_endpoint": "not_used", "wan_pipeline": "not_loaded", "gpu": "not_used", "frozen_split": "not_read"},
        "limitations": ["此预检只验证工程闭环，不证明 RSI 能力、研究效率或 Wan 质量收益。", "外部任务服务与真实 Wan worker 尚未接入；真实 worker 更新/重载证据不属于本地 CPU 预检。"],
        "generated_at_unix": time.time(),
    }
    if actual_jobs != 2 or budget_snapshot["jobs"].get("fixture-job#attempt-1", {}).get("actual") != {"job_count": 1.0}:
        raise ValueError("m0_budget_reconciliation_failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-store", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = preflight(args.config, args.bundle_store, args.trace_root)
    except Exception as exc:
        report = {"schema_version": 1, "passed": False, "blockers": [type(exc).__name__ + ": " + str(exc)]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
