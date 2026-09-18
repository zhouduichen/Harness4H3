# A-Evolve 风格 H3 Round Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有真实 MiniMax-H3 远程闭环中加入有界 round policy、经验摘要和固定 gate，并把评价期的安全 GPU overlap 接入可恢复的远程 pipeline。

**Architecture:** 保留 Harness4H3 的真实 worker、ComfyUI、SSH scheduler、lease 和 checkpoint retention 作为不可变执行底座。新增的 `RoundPolicy` 只限制下一轮搜索空间，`DiscoveryDigest` 只把结构化经验摘要交给远程本地 LLM，`RoundGate` 固定决定接受、拒绝、重规划或等待。评价期间由现有 lane scheduler 维护互斥 lease，prefetch 和 speculative worker 只在 gate/资源约束允许时运行。

**Tech Stack:** Python 3.12、frozen dataclasses、JSON/JSONL、现有 SSH/nvidia-smi lease、ComfyUI capability probe、pytest。

## Global Constraints

- LLM 只能产生结构化 `RoundPolicy` 和 `ExperimentPlan`，不能产生命令、任意路径或 GPU 操作。
- substrate、评价 recipe、硬 gate、worker contract、lease 和 retention 由代码/configuration 固定。
- LPL/TDTM/CI-DL 只有 `safe_to_plan=true` 的真实 capability evidence 才能进入 plan；CI-DL baseline 不是 checkpoint operator。
- 外部 GPU 进程 fail-closed，绝不使用 `pkill`、`killall` 或模糊 PID 匹配。
- 不修改正在运行的 torchrun world size；功耗和利用率只记录/优化，不强行设置 300 W。
- 上下文只传摘要和 observation ID，不传 checkpoint bytes、视频或无限日志。
- 每个任务结束都运行对应的 focused pytest；不把无关 dirty worktree 加入提交。

---

### Task 1: Add the immutable RoundPolicy contract

**Files:**
- Create: `harness4h3/controller/round_policy.py`
- Modify: `harness4h3/controller/__init__.py`
- Test: `tests/unit/test_round_policy.py`

**Interfaces:**
- Produces `RoundPolicy`, `RoundPolicyValidationError` and `validate_round_policy` for campaign and provider code.
- `RoundPolicy.from_dict(raw: Mapping[str, Any]) -> RoundPolicy` rejects unknown fields and missing required fields.
- `RoundPolicy.to_dict() -> Mapping[str, Any]` emits JSON-safe tuples as lists.
- `validate_round_policy(policy, *, registered_operators, substrate_digest, evaluation_digest, gpu_count) -> None` rejects policy drift and unsafe resource bounds.

- [ ] **Step 1: Write failing schema tests.**

```python
def valid_policy_dict():
    return {
        "schema_version": 1,
        "round_id": "R0012",
        "substrate_digest": "sha256:substrate",
        "search_mode": "runtime_efficiency",
        "allowed_operators": ["step_distill", "lpl"],
        "axis_budget": {"max_trials": 2, "max_gpu_hours": 8.0},
        "objective": {"quality_floor": 0.82},
        "fixed_evaluation": {"split": "heldout", "recipe_digest": "sha256:eval"},
        "resource_policy": {"min_training_gpus": 2, "controller_overlap_gpus": 1},
        "stop_conditions": ["critical_regression", "budget_exhausted"],
        "source_observation_ids": ["obs-1"],
        "created_at": "2026-09-18T00:00:00+00:00",
    }


def test_round_policy_round_trips_and_rejects_unknown_fields():
    raw = {
        "schema_version": 1,
        "round_id": "R0012",
        "substrate_digest": "sha256:substrate",
        "search_mode": "runtime_efficiency",
        "allowed_operators": ["step_distill", "lpl"],
        "axis_budget": {"max_trials": 2, "max_gpu_hours": 8.0},
        "objective": {"quality_floor": 0.82},
        "fixed_evaluation": {"split": "heldout", "recipe_digest": "sha256:eval"},
        "resource_policy": {"min_training_gpus": 2, "controller_overlap_gpus": 1},
        "stop_conditions": ["critical_regression", "budget_exhausted"],
        "source_observation_ids": ["obs-1"],
        "created_at": "2026-09-18T00:00:00+00:00",
    }
    policy = RoundPolicy.from_dict(raw)
    assert policy.to_dict()["allowed_operators"] == ["step_distill", "lpl"]
    with pytest.raises(RoundPolicyValidationError, match="unknown"):
        RoundPolicy.from_dict({**raw, "unexpected": True})


def test_round_policy_cannot_drift_substrate_or_overclaim_gpu_overlap():
    policy = RoundPolicy.from_dict(valid_policy_dict())
    with pytest.raises(RoundPolicyValidationError, match="substrate"):
        validate_round_policy(
            policy,
            registered_operators={"step_distill", "lpl"},
            substrate_digest="sha256:other",
            evaluation_digest="sha256:eval",
            gpu_count=4,
        )
    unsafe = replace(policy, resource_policy={"min_training_gpus": 1, "controller_overlap_gpus": 4})
    with pytest.raises(RoundPolicyValidationError, match="GPU"):
        validate_round_policy(
            unsafe,
            registered_operators={"step_distill", "lpl"},
            substrate_digest="sha256:substrate",
            evaluation_digest="sha256:eval",
            gpu_count=4,
        )
```

- [ ] **Step 2: Run the focused tests and verify they fail.**

Run: `pytest -q tests/unit/test_round_policy.py`

Expected: FAIL because the new contract is not defined.

- [ ] **Step 3: Implement the minimal frozen contract.**

Use this public API:

- `RoundPolicy` is a `@dataclass(frozen=True)` with fields `schema_version: int`,
  `round_id: str`, `substrate_digest: str`, `search_mode: str`,
  `allowed_operators: Tuple[str, ...]`, `axis_budget: Mapping[str, Any]`,
  `objective: Mapping[str, Any]`, `fixed_evaluation: Mapping[str, Any]`,
  `resource_policy: Mapping[str, Any]`, `stop_conditions: Tuple[str, ...]`,
  `source_observation_ids: Tuple[str, ...]`, and `created_at: str`.
- `RoundPolicy.from_dict(raw: Mapping[str, Any]) -> RoundPolicy` performs strict parsing.
- `RoundPolicy.to_dict() -> Mapping[str, Any]` returns JSON-safe lists for tuple fields.
- `validate_round_policy(policy: RoundPolicy, *, registered_operators: Iterable[str],
  substrate_digest: str, evaluation_digest: str, gpu_count: int) -> None` performs
  the invariant checks listed below and raises `RoundPolicyValidationError`.

Require `schema_version == 1`, positive `axis_budget.max_trials`, non-negative
`max_gpu_hours`, `min_training_gpus >= 2`, `0 <= controller_overlap_gpus < gpu_count`,
and `min_training_gpus + controller_overlap_gpus <= gpu_count`. Require
`fixed_evaluation.recipe_digest == evaluation_digest` and every allowed operator to
be in `registered_operators`.

- [ ] **Step 4: Run the focused tests and the controller schema suite.**

Run: `pytest -q tests/unit/test_round_policy.py tests/unit/test_controller_providers.py`

Expected: PASS with existing provider behavior unchanged.

- [ ] **Step 5: Commit only the new contract and tests.**

```bash
git add harness4h3/controller/round_policy.py harness4h3/controller/__init__.py tests/unit/test_round_policy.py
git commit -m "feat: add immutable H3 round policy contract"
```

### Task 2: Build the bounded DiscoveryDigest

**Files:**
- Create: `harness4h3/memory/discovery_digest.py`
- Modify: `harness4h3/memory/__init__.py`
- Test: `tests/unit/test_discovery_digest.py`

**Interfaces:**
- `DigestLimits(max_recent_experiments=8, max_operator_findings=2, max_frontier=4, max_failures=16, max_item_chars=2048)` is immutable.
- `DiscoveryDigest.build(experiments, observations=(), pareto=(), telemetry=(), limits=DigestLimits()) -> DiscoveryDigest` accepts JSON-like mappings and never reads checkpoint payloads.
- `DiscoveryDigest.to_context() -> Mapping[str, Any]` is the only form passed to the Controller.

- [ ] **Step 1: Write failing size, provenance and failure-retention tests.**

```python
def test_digest_keeps_failure_recipe_and_bounds_long_history():
    records = [
        {"experiment_id": "exp-%04d" % index, "plan": {"operator": "lpl"},
         "execution": {"status": "failed" if index == 0 else "success"},
         "failure_type": "oom" if index == 0 else None,
         "evaluation": {"quality_score": 0.8 + index / 1000.0}}
        for index in range(40)
    ]
    digest = DiscoveryDigest.build(records)
    context = digest.to_context()
    assert len(context["recent_experiments"]) == 8
    assert any(item["experiment_id"] == "exp-0000" for item in context["operator_findings"]["lpl"])
    assert all(len(json.dumps(item, ensure_ascii=False)) <= 2048 for item in context["recent_experiments"])
    assert "source_digest" in context
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `pytest -q tests/unit/test_discovery_digest.py`

Expected: FAIL because `DiscoveryDigest` is not defined.

- [ ] **Step 3: Implement deterministic bounded selection.**

Partition records by operator and success/failure, retain the newest records first,
then add representative failure records if they fell outside the recent window.
Clip nested mappings recursively at `max_item_chars`; never include keys named
`checkpoint`, `checkpoint_path`, `stdout`, `stderr`, `video`, or `payload` unless they
are replaced by a short digest/reference. Compute `source_digest` as SHA-256 over the
canonical bounded JSON and expose `source_observation_ids`.

- [ ] **Step 4: Run memory and context tests.**

Run: `pytest -q tests/unit/test_discovery_digest.py tests/test_context.py`

Expected: PASS; the digest remains below the configured context budget for a 40-record fixture.

- [ ] **Step 5: Commit the digest module.**

```bash
git add harness4h3/memory/discovery_digest.py harness4h3/memory/__init__.py tests/unit/test_discovery_digest.py
git commit -m "feat: add bounded H3 discovery digest"
```

### Task 3: Feed policy and digest to the remote local LLM

**Files:**
- Modify: `harness4h3/controller/context.py`
- Modify: `harness4h3/controller/provider.py`
- Modify: `research/experiments/remote_h3_closed_loop.py`
- Test: `tests/unit/test_controller_providers.py`
- Test: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- Add backwards-compatible `round_policy: Mapping[str, Any] = {}` and `discovery_digest: Mapping[str, Any] = {}` fields to `ControllerContext`.
- Add `RemoteCampaign._build_discovery_digest() -> DiscoveryDigest` and `RemoteCampaign._active_round_policy() -> Optional[RoundPolicy]`.
- Provider prompt receives `round_policy` and `discovery_digest` through the existing bounded context serializer; raw `recent_experiments` remains bounded and is not duplicated.

- [ ] **Step 1: Add failing prompt-boundary tests.**

```python
from dataclasses import replace


def context_with(**fields):
    return replace(context(), **fields)


def test_controller_prompt_contains_digest_not_checkpoint_payload():
    context = context_with(
        round_policy={"round_id": "R1", "allowed_operators": ["step_distill"]},
        discovery_digest={"source_digest": "sha256:d", "recent_experiments": [{"experiment_id": "e1"}]},
    )
    prompt = _controller_prompt(context, experiment_plan_json_schema(context))
    assert "R1" in prompt
    assert "sha256:d" in prompt
    assert "checkpoint_payload" not in prompt
```

- [ ] **Step 2: Run the focused provider tests and verify the new assertion fails.**

Run: `pytest -q tests/unit/test_controller_providers.py -k digest`

Expected: FAIL until the context and prompt serializers include the new fields.

- [ ] **Step 3: Implement bounded context integration.**

At each primary Controller request, load the active policy from the campaign state,
build a digest from the existing `ExperimentStore`/observation store, and put only
`to_context()` output into `ControllerContext`. Add a prompt sentence stating that
the digest is evidence references, not authorization to change fixed gates. Keep
`optimization_capabilities` as the separate capability gate already used by the
provider.

- [ ] **Step 4: Add replan lineage tests.**

Assert that a plan with an operator outside `RoundPolicy.allowed_operators` is rejected
before worker launch, while a plan that consumes a listed observation ID is accepted.

- [ ] **Step 5: Run all controller/context tests.**

Run: `pytest -q tests/unit/test_controller_providers.py tests/test_context.py tests/integration/test_remote_h3_closed_loop.py`

Expected: PASS with legacy contexts that omit the new fields.

- [ ] **Step 6: Commit the context integration.**

```bash
git add harness4h3/controller/context.py harness4h3/controller/provider.py research/experiments/remote_h3_closed_loop.py tests/unit/test_controller_providers.py tests/integration/test_remote_h3_closed_loop.py
git commit -m "feat: expose bounded round policy experience to controller"
```

### Task 4: Add a fixed RoundGate and durable round state

**Files:**
- Create: `harness4h3/remote/round_gate.py`
- Modify: `research/experiments/remote_h3_closed_loop.py`
- Test: `tests/unit/test_round_gate.py`
- Test: `tests/integration/test_real_h3_closed_loop.py`

**Interfaces:**
- `RoundGateStatus = Literal["accepted", "rejected", "replan", "waiting"]`.
- `RoundGateResult(status, reasons, evidence, retention)` is immutable and JSON serializable.
- `evaluate_round_gate(*, worker_result, evaluation, target, capability_evidence, lane_evidence, parent_digest, child_digest, retention) -> RoundGateResult` never calls the LLM.

- [ ] **Step 1: Write failing gate tests.**

```python
def test_round_gate_rejects_missing_child_and_keeps_evidence():
    result = evaluate_round_gate(
        worker_result={"status": "success", "metrics": {"real_worker": True}},
        evaluation={"feasible": True, "quality_score": 0.9},
        target={"min_quality_score": 0.8},
        capability_evidence={},
        lane_evidence={"disjoint": True},
        parent_digest="sha256:p",
        child_digest=None,
        retention={"outcome": "rejected_candidate"},
    )
    assert result.status == "rejected"
    assert "child_digest_missing" in result.reasons
```

- [ ] **Step 2: Run the gate test and verify it fails.**

Run: `pytest -q tests/unit/test_round_gate.py`

Expected: FAIL because the fixed gate module is not defined.

- [ ] **Step 3: Implement fail-closed gate ordering.**

Check in order: real worker evidence, parent/child identity, capability evidence for
the requested operator, disjoint lane evidence, evaluation validity, critical
regression, target hard constraints, then continuation policy. Map resource or
dependency absence to `waiting`, recoverable worker/evaluation failure to `replan`,
and only a fully valid accepted candidate to `accepted`. The function must never
delete files; retention remains the existing exact-path service.

- [ ] **Step 4: Persist policy/gate cursors and events.**

Extend campaign state with `active_round_policy`, `discovery_digest`, `round_gate`,
and `next_plan_source`. Emit `round_policy_activated`, `round_gate_evaluated`, and
`round_policy_rejected` with bounded payloads. Update state atomically before model
promotion and before speculative plan publication.

- [ ] **Step 5: Run gate, retention and real-loop contract tests.**

Run: `pytest -q tests/unit/test_round_gate.py tests/unit/test_remote_checkpoint_retention.py tests/integration/test_real_h3_closed_loop.py`

Expected: PASS; rejected/failed artifacts are cleaned only through existing retention APIs.

- [ ] **Step 6: Commit the gate/state work.**

```bash
git add harness4h3/remote/round_gate.py research/experiments/remote_h3_closed_loop.py tests/unit/test_round_gate.py tests/integration/test_real_h3_closed_loop.py
git commit -m "feat: add fixed round gate and durable policy state"
```

### Task 5: Close the evaluation-time GPU gap

**Files:**
- Modify: `harness4h3/remote/pipeline.py`
- Modify: `harness4h3/remote/lane_packer.py`
- Modify: `research/experiments/remote_h3_closed_loop.py`
- Test: `tests/unit/test_remote_pipeline.py`
- Test: `tests/unit/test_remote_scheduler.py`
- Test: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- Add pure `EvaluationOverlap` and `pack_evaluation_overlap(total_gpu_count, evaluator_gpus, controller_gpus, minimum_training_gpus) -> EvaluationOverlap`.
- Extend `RemoteCampaign._record_lane_allocation` with `round_id: Optional[str] = None` without changing lease ownership.
- The existing `_start_speculative_worker` call with `preserve_controller_lane=True` remains the only path allowed to overlap a training candidate with the Controller.

- [ ] **Step 1: Write failing packing and event-order tests.**

```python
def test_evaluation_overlap_leaves_two_training_cards():
    result = pack_evaluation_overlap(4, evaluator_gpus=(0,), controller_gpus=(1,), minimum_training_gpus=2)
    assert result.training_gpus == (2, 3)
    assert result.disjoint is True


def test_overlap_waits_when_evaluator_and_controller_leave_too_few_cards():
    result = pack_evaluation_overlap(4, evaluator_gpus=(0, 1), controller_gpus=(2,), minimum_training_gpus=2)
    assert result.mode == "waiting"
    assert result.training_gpus == ()
```

- [ ] **Step 2: Run focused tests to verify failure.**

Run: `pytest -q tests/unit/test_remote_pipeline.py -k evaluation_overlap`

Expected: FAIL because the pure overlap contract is not defined.

- [ ] **Step 3: Implement pure disjoint packing and effective request shaping.**

Reject duplicate/out-of-range indices. Never reduce a worker below its declared
distributed minimum. Return `waiting` instead of a partial distributed allocation.
Keep the original plan request in the event and record the effective request after
the Controller/evaluator caps are applied.

- [ ] **Step 4: Wire the existing prefetch boundary to the overlap contract.**

During candidate evaluation:

1. reserve only the required ComfyUI workers;
2. start `controller_plan_prefetch_started` before waiting for the final benchmark result;
3. keep the Controller lease when at least two safe worker cards remain;
4. launch the speculative worker only after its plan is persisted and the scheduler returns a disjoint lease;
5. emit `lane_allocation` and `worker_started` with the same `round_id` and disjoint GPU sets;
6. if the plan requires all four cards, persist it before acquiring the worker lease and wait for the next boundary.

Do not add a second vLLM process and do not kill a foreign ComfyUI process.

- [ ] **Step 5: Add fake-scheduler integration assertions.**

Assert the event order `controller_plan_prefetch_started -> lane_allocation -> worker_started`,
that worker/controller/evaluator GPU sets are pairwise disjoint, and that a failed
prefetch or unavailable GPU returns `waiting` without consuming an optimization iteration.

- [ ] **Step 6: Run the remote orchestration suite.**

Run: `pytest -q tests/unit/test_remote_pipeline.py tests/unit/test_remote_scheduler.py tests/integration/test_remote_h3_closed_loop.py tests/unit/test_overnight_controller.py`

Expected: PASS with existing elastic 2/3/4-GPU behavior and no broad process termination.

- [ ] **Step 7: Commit the evaluation overlap work.**

```bash
git add harness4h3/remote/pipeline.py harness4h3/remote/lane_packer.py research/experiments/remote_h3_closed_loop.py tests/unit/test_remote_pipeline.py tests/unit/test_remote_scheduler.py tests/integration/test_remote_h3_closed_loop.py
git commit -m "feat: pack evaluation and speculative training lanes"
```

### Task 6: Deploy the autonomous watcher and verify the remote boundary

**Files:**
- Modify: `tools/remote-campaign-service.sh`
- Modify: `tools/remote-campaign-supervisor.sh`
- Modify: `tools/remote-idle-autostart.sh`
- Test: `tests/unit/test_remote_campaign_service.py`
- Test: `tests/unit/test_remote_campaign_supervisor.py`
- Test: `tests/unit/test_remote_idle_autostart.py`
- Documentation: `docs/quickstart.md`

**Interfaces:**
- `remote-idle-autostart.sh` remains CPU-only and starts only when all four GPUs pass the external-process and free-memory gate.
- The supervisor owns the exact campaign process tree and runs `cleanup-controller` after terminal result or graceful stop.
- Status output reports watcher PID, campaign state, active round, pipeline stage, and last terminal result without requiring Codex.

- [ ] **Step 1: Run local shell/static tests before deployment.**

Run:

```bash
bash -n tools/remote-campaign-service.sh tools/remote-campaign-supervisor.sh tools/remote-idle-autostart.sh
pytest -q tests/unit/test_remote_campaign_service.py tests/unit/test_remote_campaign_supervisor.py tests/unit/test_remote_idle_autostart.py
```

Expected: PASS and no `pkill`/`killall` occurrence in the project-owned scripts.

- [ ] **Step 2: Synchronize exact files to the remote project.**

Use the configured SSH endpoint `intern@222.29.98.132` on port `30902`; copy only
the changed files to `/home/intern/huangjiahao/Harness4H3-rsi`, then run remote
`bash -n` and the focused tests. Do not copy the local `.venv`, checkpoints, or
unrelated dirty files.

- [ ] **Step 3: Read remote status before any start/stop action.**

Run:

```bash
ssh -p 30902 intern@222.29.98.132 \
  'cd /home/intern/huangjiahao/Harness4H3-rsi && \
   tools/remote-idle-autostart.sh status && \
   tools/remote-campaign-service.sh status'
```

If SSH is actively closed, report the transport blocker and do not infer that the
watcher is dead or restart it solely because observation failed.

- [ ] **Step 4: Perform one bounded remote smoke window when the external GPU gate is idle.**

Verify the event stream contains policy activation, digest source IDs, controller
prefetch, disjoint lane allocation, ComfyUI unload/release, worker completion, gate
result, and retention outcome. Record actual per-GPU power/utilization/memory and
idle reasons; do not claim 300 W/100% unless telemetry proves it.

- [ ] **Step 5: Update operator documentation and commit deployment changes.**

Document the direct SSH command (not the unavailable local alias), `status`, `watch`,
graceful stop, and the fact that Codex need not stay attached. Commit only scripts,
tests, and documentation changed by this task.

## Verification matrix

After all tasks, run:

```bash
pytest -q tests/unit/test_round_policy.py tests/unit/test_discovery_digest.py tests/unit/test_round_gate.py
pytest -q tests/unit/test_controller_providers.py tests/unit/test_remote_pipeline.py tests/unit/test_remote_scheduler.py
pytest -q tests/integration/test_remote_h3_closed_loop.py tests/integration/test_real_h3_closed_loop.py
pytest -q tests/unit/test_remote_campaign_service.py tests/unit/test_remote_campaign_supervisor.py tests/unit/test_remote_idle_autostart.py
git diff --check
```

The acceptance report must separate: (1) local contract evidence, (2) remote
runtime evidence, and (3) unavailable evidence caused by external GPU jobs or SSH
transport. A green local suite alone does not prove the four-card remote objective.
