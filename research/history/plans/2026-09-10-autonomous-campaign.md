# Fixed-Harness Autonomous Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the M6 runtime campaign produce a bounded, execution-grounded research record that distinguishes rejected candidates from failed experiments, prevents repeated configurations, and never chains rejected candidates as parents.

**Architecture:** Keep `Harness4H3-v1.0` unchanged. Add pure campaign bookkeeping helpers beside the experiment entry point, then update `m6_runtime_recipe.py` to use those helpers for fingerprints, failure classification, active-parent selection, budget accounting, complete trajectories, and the final research report. Runtime operators and the existing Controller protocol remain unchanged.

**Tech Stack:** Python 3.9+, dataclasses, `hashlib`, JSON, pytest, existing Ollama/ComfyUI M6 runner.

## Global Constraints

- Harness version is exactly `Harness4H3-v1.0`.
- Only correctness/evidence bookkeeping in `experiments/` may change; do not modify Controller protocol, ExperimentPlan schema, Model/Runtime state schema, Evaluator, archive/trajectory semantics, TargetProfile semantics, or Controller loop.
- `max_experiments=8` and `max_failed_experiments=5` for the first campaign.
- `rejected_candidate` consumes experiment budget but not failure budget.
- `failed_experiment` consumes both experiment and failure budget.
- An identical `(operator, normalized_operator_args, parent_state_digest, target_profile_id)` fingerprint is never retried.
- A changed configuration for a previously used operator must cite a prior experiment id in the Controller hypothesis.
- A rejected candidate never becomes the next experiment's parent or active baseline.
- `human_intervention_count` is `0` for the autonomous run.

---

### Task 1: Add pure campaign bookkeeping helpers

**Files:**
- Create: `experiments/m6_campaign.py`
- Create: `tests/unit/test_m6_campaign.py`

**Interfaces:**
- Produces `canonicalize(value: Any) -> Any`, `state_digest(state: ModelState) -> str`, `experiment_fingerprint(operator: str, operator_args: Mapping[str, Any], parent_state: ModelState, target_profile_id: str) -> str`, `validate_novelty(operator: str, hypothesis: str, fingerprint: str, seen_fingerprints: Set[str], recent: Sequence[Mapping[str, Any]]) -> Optional[str]`, `classify_outcome(operator_ok: bool, output_state_present: bool, split_results: Mapping[str, Any], split_errors: Mapping[str, Any], split_validated: bool) -> str`, and `build_research_report(payload: Mapping[str, Any], started_at: str, ended_at: str) -> Dict[str, Any]`.
- Consumes only existing `ModelState`, JSON-like mappings, and campaign payloads. It does not import or mutate any frozen Harness module beyond reading `ModelState.to_dict()`.

- [ ] **Step 1: Write failing tests for canonical fingerprints and outcome classification**

```python
from dataclasses import replace

from experiments.m6_campaign import (
    classify_outcome,
    experiment_fingerprint,
    state_digest,
    validate_novelty,
)
from harness4h3.h3.state import ModelState


def _state(policy=None):
    state = ModelState.fake_baseline("M0001")
    return replace(state, architecture_name="MiniMax-H3", runtime_state=policy or {})


def test_fingerprint_is_order_independent_but_changes_with_parent_or_args():
    first = experiment_fingerprint("runtime_offload", {"mode": "aggressive", "x": 1}, _state(), "rtx")
    second = experiment_fingerprint("runtime_offload", {"x": 1, "mode": "aggressive"}, _state(), "rtx")
    changed_parent = experiment_fingerprint("runtime_offload", {"mode": "aggressive", "x": 1}, _state({"runtime_policy": {"kind": "cache_release"}}), "rtx")
    changed_args = experiment_fingerprint("runtime_offload", {"mode": "balanced", "x": 1}, _state(), "rtx")
    assert first == second
    assert first != changed_parent
    assert first != changed_args
    assert len(state_digest(_state())) == 64


def test_same_operator_with_new_args_requires_prior_experiment_reference():
    recent = [{"experiment_id": "exp_0001", "operator": "vae_tiling", "operator_args": {"tile_size": 256}}]
    assert validate_novelty("vae_tiling", "try exp_0001 with lower temporary activations", "new", set(), recent) is None
    assert validate_novelty("vae_tiling", "try another tiling", "newer", set(), recent) == "same_operator_requires_prior_evidence"
    assert validate_novelty("runtime_offload", "new offload hypothesis", "new", set(), recent) is None
    assert validate_novelty("vae_tiling", "new", "old", {"old"}, recent) == "duplicate_experiment_fingerprint"


def test_rejection_does_not_consume_failure_budget_classification():
    assert classify_outcome(True, True, {"dev": {"validated": False}}, {}, False) == "rejected_candidate"
    assert classify_outcome(True, True, {"dev": {"validated": True}}, {"heldout": {"failure_type": "timeout"}}, False) == "failed_experiment"
    assert classify_outcome(False, False, {}, {}, False) == "failed_experiment"
    assert classify_outcome(True, True, {"dev": {"validated": True}, "heldout": {"validated": True}}, {}, True) == "accepted_candidate"
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/unit/test_m6_campaign.py`

Expected: collection/import failure because `experiments/m6_campaign.py` does not exist yet.

- [ ] **Step 3: Implement the minimal pure helper module**

Implement the following exact behavior:

```python
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Set

from harness4h3.h3.state import ModelState


def canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def state_digest(state: ModelState) -> str:
    return hashlib.sha256(_canonical_json(state.to_dict()).encode("utf-8")).hexdigest()


def experiment_fingerprint(operator: str, operator_args: Mapping[str, Any], parent_state: ModelState, target_profile_id: str) -> str:
    payload = {
        "operator": str(operator),
        "normalized_operator_args": canonicalize(dict(operator_args)),
        "parent_state_digest": state_digest(parent_state),
        "target_profile_id": str(target_profile_id),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def validate_novelty(operator: str, hypothesis: str, fingerprint: str, seen_fingerprints: Set[str], recent: Sequence[Mapping[str, Any]]) -> Optional[str]:
    if fingerprint in seen_fingerprints:
        return "duplicate_experiment_fingerprint"
    prior_ids = [str(item["experiment_id"]) for item in recent if str(item.get("operator", "")) == str(operator) and item.get("experiment_id")]
    if prior_ids and not any(prior_id in str(hypothesis) for prior_id in prior_ids):
        return "same_operator_requires_prior_evidence"
    return None


def classify_outcome(operator_ok: bool, output_state_present: bool, split_results: Mapping[str, Any], split_errors: Mapping[str, Any], split_validated: bool) -> str:
    if not operator_ok or not output_state_present or split_errors or not split_results:
        return "failed_experiment"
    return "accepted_candidate" if split_validated else "rejected_candidate"


def build_research_report(payload: Mapping[str, Any], started_at: str, ended_at: str) -> Dict[str, Any]:
    iterations = list(payload.get("iterations") or [])
    accepted = [item for item in iterations if item.get("outcome") == "accepted_candidate"]
    rejected = [item for item in iterations if item.get("outcome") == "rejected_candidate"]
    failed = [item for item in iterations if item.get("outcome") == "failed_experiment"]
    operators = [str(item.get("operator")) for item in iterations if item.get("operator")]
    return {
        "harness_version": payload["harness"]["version"],
        "target_profile_id": payload["target_profile_id"],
        "termination_reason": payload.get("termination_reason"),
        "target_satisfied": payload.get("status") == "accepted",
        "full_autonomous_experiment_sequence": iterations,
        "accepted_experiments": accepted,
        "rejected_experiments": rejected,
        "failed_experiments": failed,
        "final_pareto_candidates": list(payload.get("system_candidates") or []),
        "total_experiments": len(iterations),
        "failed_experiment_count": len(failed),
        "rejected_candidate_count": len(rejected),
        "wall_time_s": float(payload.get("wall_time_s", 0.0)),
        "gpu_hours": float(payload.get("gpu_hours", 0.0)),
        "gpu_hours_available": bool(payload.get("gpu_hours_available", False)),
        "human_intervention_count": 0,
        "repeated_failure_avoidance": {
            "duplicate_fingerprints_blocked": int(payload.get("duplicate_fingerprints_blocked", 0)),
            "same_operator_requires_prior_evidence": True,
            "rejected_parent_reuse": False,
        },
        "strategy_changed_after_negative_evidence": bool(operators) and (len(set(operators)) > 1 or operators[0] != "vae_tiling"),
        "final_optimization_recipe": payload.get("accepted_recipe"),
        "experience_influence_evidence": payload.get("experience_influence_evidence", []),
        "started_at": started_at,
        "ended_at": ended_at,
    }
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/unit/test_m6_campaign.py`

Expected: 3 tests pass.

- [ ] **Step 5: Commit the helper and tests**

```bash
git add experiments/m6_campaign.py tests/unit/test_m6_campaign.py
git commit -m "test: add autonomous campaign bookkeeping rules"
```

### Task 2: Update the runtime campaign loop and evidence payload

**Files:**
- Modify: `experiments/m6_runtime_recipe.py`
- Test: `tests/unit/test_m6_runtime_recipe.py`

**Interfaces:**
- Consumes the helpers from Task 1.
- Produces one iteration record per Controller call, including state before/after, diagnosis, hypothesis, operator arguments, expected effects, risks, execution result, quality/hardware metrics, outcome reason, next-decision context, fingerprint, and frozen Harness version.
- Keeps `system_store.active()` as the only active parent after a rejection and passes archive summaries through the existing `ControllerContext.pareto_front` field without changing that schema.

- [ ] **Step 1: Extend focused tests with budget, parent, and evidence assertions**

Add these report-level assertions to the existing unit test module. The
production loop tests below use the persisted iteration shape directly so the
tests do not need a network or GPU:

```python
from experiments.m6_campaign import build_research_report


def _report(iterations):
    return build_research_report(
        {
            "harness": {"version": "Harness4H3-v1.0"},
            "target_profile_id": "rtx5080_h3_v1",
            "status": "experiment_budget_exhausted",
            "iterations": iterations,
            "system_candidates": [{"id": "C0000", "status": "baseline"}],
            "duplicate_fingerprints_blocked": 1,
        },
        "2026-09-10T00:00:00+00:00",
        "2026-09-10T00:00:01+00:00",
    )


def test_rejected_candidate_does_not_increment_failure_count():
    report = _report([
        {"experiment_id": "exp_0001", "outcome": "rejected_candidate", "system_parent_id": "C0000"},
        {"experiment_id": "exp_0002", "outcome": "rejected_candidate", "system_parent_id": "C0000"},
    ])
    assert report["failed_experiment_count"] == 0
    assert report["rejected_candidate_count"] == 2
    assert [item["system_parent_id"] for item in report["rejected_experiments"]] == ["C0000", "C0000"]


def test_duplicate_fingerprint_is_recorded_as_failed_without_execution():
    report = _report([
        {"experiment_id": "exp_0001", "outcome": "rejected_candidate", "operator_executed": True},
        {"experiment_id": "exp_0002", "outcome": "failed_experiment", "failure_type": "duplicate_experiment_fingerprint", "operator_executed": False},
    ])
    duplicate = report["failed_experiments"][0]
    assert duplicate["failure_type"] == "duplicate_experiment_fingerprint"
    assert duplicate["operator_executed"] is False


def test_every_iteration_has_research_record_fields_and_frozen_version():
    record = {
        "experiment_id": "exp_0001", "outcome": "rejected_candidate", "state_before": {},
        "diagnosis": "memory", "hypothesis": "try offload", "operator": "runtime_offload",
        "operator_args": {"mode": "aggressive"}, "expected_effects": {}, "risks": [],
        "execution_result": {}, "quality_metrics": {}, "hardware_metrics": {},
        "accept_reject_reason": "peak_memory_gate", "state_after": {},
        "next_decision_context": {}, "harness_version": "Harness4H3-v1.0",
    }
    report = _report([record])
    record = report["full_autonomous_experiment_sequence"][0]
    for key in (
        "state_before", "diagnosis", "hypothesis", "operator", "operator_args",
        "expected_effects", "risks", "execution_result", "quality_metrics",
        "hardware_metrics", "accept_reject_reason", "state_after",
        "next_decision_context", "harness_version",
    ):
        assert key in record
    assert record["harness_version"] == "Harness4H3-v1.0"
```

The tests may use the existing fake classes; do not change the production Harness loop or schemas to make them pass.

- [ ] **Step 2: Run the focused tests and verify the new assertions fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/unit/test_m6_runtime_recipe.py`

Expected: the new assertions fail because the current loop counts rejection as failure, advances from rejected `system_child`, and omits explicit evidence fields.

- [ ] **Step 3: Change campaign defaults and initialize explicit state**

In `main()` change the CLI defaults to:

```python
parser.add_argument("--max-iterations", type=int, default=8)
parser.add_argument("--max-failed-experiments", type=int, default=5)
```

Initialize `campaign_started = time.monotonic()`, `started_at = datetime.now(timezone.utc).isoformat()`, `budget_state = BudgetState(max_iterations=args.max_iterations, max_failed_experiments=args.max_failed_experiments, max_controller_calls=args.max_iterations)`, `seen_fingerprints = set()`, `experiment_history = []`, `failed_history = []`, and `duplicate_fingerprints_blocked = 0` before the loop. Set `human_intervention_count` to `0` in the top-level payload.

- [ ] **Step 4: Pass archive summaries while retaining the frozen Controller context**

Add a local summary builder in the experiment entry point:

```python
def _archive_summary(store: SystemCandidateStore) -> List[Mapping[str, Any]]:
    return [
        {
            "candidate_id": candidate.id,
            "parent_id": candidate.parent_id,
            "generation": candidate.generation,
            "model_ref": candidate.model_ref,
            "status": candidate.status,
            "evaluation": dict(candidate.evaluation),
        }
        for candidate in store.lineage()
    ]
```

Add a `pareto_front` argument to the private `_controller_plan()` helper and pass `_archive_summary(system_store)` into the existing `ControllerContext` field. Do not add fields to `ControllerContext` or `ExperimentPlan`.

- [ ] **Step 5: Enforce novelty before operator execution**

After the plan is returned and effective operator args are computed, calculate:

```python
fingerprint = experiment_fingerprint(
    requested_operator,
    operator_args,
    current_state,
    target.id,
)
novelty_error = validate_novelty(
    requested_operator,
    plan.hypothesis,
    fingerprint,
    seen_fingerprints,
    experiment_history,
)
```

If `novelty_error` is non-null, do not call `registry.execute`. Record a complete failed iteration with `failure_type=novelty_error`, increment `duplicate_fingerprints_blocked` for the duplicate case, consume one experiment and one failure, append the record to history/trajectory, persist the payload, and continue until the configured failure budget stops the campaign.

- [ ] **Step 6: Separate rejected candidates from failed experiments**

Use the Task 1 classifier after operator execution and validation:

```python
outcome = classify_outcome(
    operator_result.ok,
    operator_result.output_state is not None,
    split_results,
    split_errors,
    split_validated,
)
```

For `rejected_candidate`, set `failure_type` to `None`, increment only experiment count, append the rejection to `experiment_history`, and leave `system` unchanged. For `failed_experiment`, append to `failed_history`, increment both experiment and failure counts, and leave `system` unchanged. For `accepted_candidate`, set the new system active and stop with `target_satisfied`.

The state transition must therefore be:

```python
parent_system = system_store.active()
current_state = parent_system.evaluation_state(base_model)
# rejected candidate: never assign system = system_child
# accepted candidate: system_store.set_active(system_child.id); stop
```

- [ ] **Step 7: Record all required fields and actual wall-time cost**

Build a record with this shape for every loop iteration, including Controller and backend failures:

```python
{
    "iteration": iteration,
    "experiment_id": plan.experiment_id if plan else "exp_%04d" % (iteration + 1),
    "system_parent_id": parent_system.id,
    "state_before": current_state.to_dict(),
    "diagnosis": plan.diagnosis if plan else None,
    "hypothesis": plan.hypothesis if plan else None,
    "operator": requested_operator if plan else None,
    "operator_args": operator_args,
    "expected_effects": dict(plan.expected_effects) if plan else {},
    "risks": list(plan.risks) if plan else [],
    "fingerprint": fingerprint if plan else None,
    "execution_result": operator_result.to_dict() if operator_result else {"status": "not_executed"},
    "quality_metrics": {split: _compact_split(value).get("branch_metrics", {}) for split, value in split_results.items()},
    "hardware_metrics": {split: _compact_split(value).get("gates", {}) for split, value in split_results.items()},
    "outcome": outcome,
    "accept_reject_reason": reason,
    "state_after": state_after.to_dict(),
    "next_decision_context": next_decision_context,
    "harness_version": HARNESS_VERSION,
    "failure_type": failure_type,
    "operator_executed": operator_result is not None,
}
```

Use `time.monotonic()` around the entire Controller/execute/validate iteration and consume `CostEstimate(wall_time_s=elapsed, gpu_hours=operator_result.cost.gpu_hours if operator_result else 0.0)` into `budget_state`. Keep `gpu_hours_available=False` unless a real non-zero GPU-hour measurement is present.

- [ ] **Step 8: Persist the final report and termination reason**

At loop exit set `payload["status"]` to `accepted` for target satisfaction, `failure_budget_exhausted` when `budget_state.used_failures >= args.max_failed_experiments`, otherwise `experiment_budget_exhausted`. Also write `budget`, `failure_count`, `rejected_candidate_count`, `total_experiments`, `wall_time_s`, `gpu_hours`, `gpu_hours_available`, `duplicate_fingerprints_blocked`, `termination_reason`, and `experience_influence_evidence` showing that the `H3-M6-VAE-Tiling-001` gene was supplied and which later operators/configurations were selected.

Call `build_research_report()` and store it under `payload["research_report"]`; add a `--report-output` argument and write the same report to that JSON path. The report must include the complete sequence, accepted/rejected/failed lists, final candidate archive, strict target result, intervention count, repeated-failure behavior, strategy-change assessment, final recipe, and negative-evidence influence.

- [ ] **Step 9: Run focused tests and verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/unit/test_m6_campaign.py tests/unit/test_m6_runtime_recipe.py tests/unit/test_harness_freeze.py`

Expected: all focused tests pass and no freeze metadata changes are detected.

- [ ] **Step 10: Commit the campaign loop and evidence changes**

```bash
git add experiments/m6_runtime_recipe.py tests/unit/test_m6_runtime_recipe.py
git commit -m "feat: enforce autonomous campaign evidence rules"
```

### Task 3: Full regression and campaign preflight

**Files:**
- Modify: none
- Test: existing `tests/`

- [ ] **Step 1: Run the full offline suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q`

Expected: all existing tests plus the new campaign tests pass without network or GPU access.

- [ ] **Step 2: Compile the package and inspect the frozen diff**

Run: `PYTHONPATH=. .venv/bin/python -m compileall -q harness4h3 experiments tests` and `git diff -- harness4h3/controller harness4h3/evaluator harness4h3/archive harness4h3/memory harness4h3/target`

Expected: compilation succeeds and the frozen Harness paths have no diff.

- [ ] **Step 3: Check Controller and backend availability without counting an experiment**

Run:

```bash
curl -fsS --max-time 8 http://100.88.143.10:11434/api/tags
curl -fsS --max-time 8 http://100.88.143.10:8188/system_stats
```

Expected: Ollama advertises `qwen3.5:9b-q8_0`; ComfyUI either responds or the campaign records backend failure only when the real campaign starts. Preflight requests are not experiment records.

### Task 4: Execute and audit the real autonomous campaign

**Files:**
- Create: `var/m6-runtime/campaign-20260910/m6-autonomous.json`
- Create: `var/m6-runtime/campaign-20260910/m6-autonomous-report.json`
- Create: `var/m6-runtime/campaign-20260910/trajectories.jsonl`
- Create: `var/m6-runtime/campaign-20260910/system-candidates/`

- [ ] **Step 1: Run the bounded real campaign**

Run:

```bash
PYTHONPATH=. .venv/bin/python experiments/m6_runtime_recipe.py \
  --controller ollama \
  --controller-model qwen3.5:9b-q8_0 \
  --controller-url http://100.88.143.10:11434 \
  --base-url http://100.88.143.10:8188 \
  --max-iterations 8 \
  --max-failed-experiments 5 \
  --splits dev,heldout \
  --request-timeout 120 \
  --output var/m6-runtime/campaign-20260910/m6-autonomous.json \
  --report-output var/m6-runtime/campaign-20260910/m6-autonomous-report.json \
  --benchmark-output var/m6-runtime/campaign-20260910/runs \
  --trajectory-output var/m6-runtime/campaign-20260910/trajectories.jsonl \
  --system-store-output var/m6-runtime/campaign-20260910/system-candidates
```

Allow the process to stop only on target satisfaction, eight experiments, or five true failed experiments. Do not supply an operator, operator arguments, parent, or acceptance override.

- [ ] **Step 2: Validate persisted evidence**

Run:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("var/m6-runtime/campaign-20260910")
payload = json.loads((root / "m6-autonomous.json").read_text())
report = json.loads((root / "m6-autonomous-report.json").read_text())
assert payload["harness"]["version"] == "Harness4H3-v1.0"
assert report["harness_version"] == "Harness4H3-v1.0"
assert report["total_experiments"] <= 8
assert report["failed_experiment_count"] <= 5
assert report["human_intervention_count"] == 0
required = {"state_before", "diagnosis", "hypothesis", "operator", "operator_args", "expected_effects", "risks", "execution_result", "quality_metrics", "hardware_metrics", "accept_reject_reason", "state_after", "next_decision_context", "harness_version"}
for item in report["full_autonomous_experiment_sequence"]:
    assert required <= set(item)
    assert item["harness_version"] == "Harness4H3-v1.0"
print(json.dumps({"status": payload["status"], "experiments": report["total_experiments"], "failed": report["failed_experiment_count"], "rejected": report["rejected_candidate_count"], "target_satisfied": report["target_satisfied"]}, indent=2))
PY
```

Expected: every record is complete, the frozen version is consistent, failure count excludes clean gate rejections, and target satisfaction is based on both dev and held-out strict gates.

- [ ] **Step 3: Report the research result**

Use the persisted report to state the full experiment sequence, Controller hypotheses, accepted/rejected/failed experiments, final Pareto/archive candidates, 16GB result, budgets, wall time/GPU-hours availability, zero-intervention status, repeated-failure avoidance, strategy changes after `vae_tiling` negative evidence, final recipe, and exact evidence paths. If the target is not met, classify the result as a bounded negative research result rather than a Harness failure.
