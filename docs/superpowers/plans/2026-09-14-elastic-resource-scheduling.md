# Controller-Owned Elastic Resource Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent queueing and Controller-authorized 2/3/4-GPU training allocation without preempting unrelated remote jobs.

**Architecture:** Extend the validated `ExperimentPlan.resource_request` with an explicit preferred GPU count, elastic range, and policy. `RemoteResourceScheduler` maps only currently safe GPUs and returns a concrete allocation; `RemoteCampaign` persists pending plans, retries them at loop boundaries, and uses the allocation to construct the trusted launcher. Elasticity is evaluated only between worker invocations, never by changing a running distributed job.

**Tech Stack:** Python 3, dataclasses, JSONL/campaign-state persistence, `nvidia-smi` over SSH, `torchrun`, pytest.

## Global Constraints

- Controller remains the only source of optimization operator, operator arguments, and whether elasticity is allowed.
- Scheduler may only inspect resources, queue/retry, map GPUs, and execute a previously validated command.
- Scheduler must never terminate, suspend, or modify unrelated remote processes.
- `elastic=false` means the exact preferred GPU count is required.
- `elastic=true` permits any actual count in `[min_gpu_count, max_gpu_count]`; distributed execution still requires at least 2 GPUs.
- Dynamic resizing happens only at a new worker start; no mid-run GPU hot-add or hot-remove.
- Every queue, retry, allocation, command, and terminal resource decision is appended to `controller-events.jsonl`.
- Existing dirty changes in `research/experiments/m6_campaign.py` and `research/experiments/m6_runtime_recipe.py` must remain untouched.

### Task 1: Extend and validate the Controller resource contract

**Files:**
- Modify: `harness4h3/controller/provider.py:100-110,151-155,360-400`
- Modify: `research/experiments/remote_h3_closed_loop.py:801-824`
- Modify: `tests/unit/test_controller_providers.py`
- Modify: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- Produce a resource request with `gpu_count`, `min_gpu_count`, `max_gpu_count`, `elastic`, `distributed`, `exclusive`, `evaluation_workers`, and `on_unavailable`.
- Preserve legacy persisted plans by normalizing absent elasticity fields to exact-count behavior during remote validation.

- [ ] **Step 1: Add failing schema tests** for accepting a complete elastic request, rejecting `min_gpu_count > max_gpu_count`, rejecting a distributed minimum below 2, and rejecting a request whose preferred count is outside its range.
- [ ] **Step 2: Run the focused tests** with `pytest tests/unit/test_controller_providers.py tests/integration/test_remote_h3_closed_loop.py -q`; verify the new tests fail against the current five-field contract.
- [ ] **Step 3: Extend the provider JSON schema and prompt** so Controller output contains the four new fields and is told that the scheduler cannot invent elasticity.
- [ ] **Step 4: Update `RuleBasedMockController`** to emit `elastic=true`, `gpu_count=4`, `min_gpu_count=2`, `max_gpu_count=4` for distributed training and exact `gpu_count=0`/elastic false for CPU-only operators.
- [ ] **Step 5: Add a normalization/validation helper** in `RemoteCampaign` that fills absent legacy fields as `min=max=gpu_count` and `elastic=false`, then enforces the range and distributed constraints before any queue or worker action.
- [ ] **Step 6: Run the focused tests** and confirm the resource contract tests pass.

### Task 2: Make the remote scheduler elastic and non-destructive

**Files:**
- Modify: `harness4h3/remote/scheduler.py`
- Modify: `tests/unit/test_remote_scheduler.py`

**Interfaces:**
- `RemoteResourceScheduler.acquire(request) -> ResourceDecision` returns the largest safe available allocation not exceeding `max_gpu_count` and not below `min_gpu_count` when elasticity is enabled.
- `ResourceDecision` includes the requested range and a reason suitable for event logging.

- [ ] **Step 1: Add failing scheduler tests** for 3 free GPUs satisfying a 2–4 elastic request, 1 free GPU returning `wait`, exact requests remaining exact, and `exclusive=true` waiting when any compute process exists.
- [ ] **Step 2: Run `pytest tests/unit/test_remote_scheduler.py -q`** and verify the elastic tests fail.
- [ ] **Step 3: Implement request normalization inside the scheduler** with strict type/range checks and exact-count defaults for legacy callers.
- [ ] **Step 4: Change allocation** to count memory-safe GPUs, choose `min(max_free, max_gpu_count)` when it meets the minimum, and preserve strict exclusive semantics.
- [ ] **Step 5: Include `min_gpu_count`, `max_gpu_count`, and `elastic` in `ResourceDecision.to_dict()` and run the focused tests.

### Task 3: Persist pending plans and retry at loop boundaries

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py:843-948,1074-1170`
- Modify: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- Campaign state stores `pending_plan`, `pending_parent_model_id`, `pending_training_calls`, `pending_attempts`, and `pending_last_resource_decision`.
- `_train_one()` first retries a persisted plan; it asks Controller for a new plan only when no pending plan exists or the Controller explicitly requested replan.
- `run_loop()` continues after `wait` when the plan says `on_unavailable=wait`, bounded by the caller's iteration count and an injectable retry interval.

- [ ] **Step 1: Add failing integration tests** that simulate a first resource wait, assert the plan is persisted, make GPUs available on the next snapshot, and assert the same plan executes without a second Controller call.
- [ ] **Step 2: Add a test** that `on_unavailable=replan` clears the pending plan and invokes Controller on the next loop cycle.
- [ ] **Step 3: Run the new integration tests** and verify they fail because `run_loop()` currently exits on `waiting_for_resources`.
- [ ] **Step 4: Implement plan serialization/recovery** using `ExperimentPlan.from_dict()` and model-store lookup by the persisted parent ID; keep the consumed Observation IDs unchanged while a plan waits.
- [ ] **Step 5: Emit `resource_queued` and `resource_retry` events** containing the experiment ID, attempt count, request, decision, and a bounded snapshot summary.
- [ ] **Step 6: Update `run_loop()`** to continue waiting plans, sleep only for a configurable/non-negative interval between retries, and stop with `resources_unavailable` only when the loop bound or budget ends.
- [ ] **Step 7: Run the focused integration tests** and confirm pending plans survive a newly constructed campaign instance.

### Task 4: Use the actual allocation in trusted training launch

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py:656-678,821-824,881-943`
- Modify: `tools/h3_real_train_worker.py:107-120,214-230`
- Modify: `tests/integration/test_remote_h3_closed_loop.py`
- Modify: `tests/training/test_real_h3_train_worker_contract.py`

**Interfaces:**
- `_worker_command(operator, config_path, request_path, result_path, allocated_gpu_count)` emits `--nproc_per_node=<allocated_gpu_count>` for torchrun.
- `worker_command_selected` and `worker_started` events include the actual allocation and process count.

- [ ] **Step 1: Add a failing command-construction test** asserting a 2-GPU allocation produces `--nproc_per_node=2` and `CUDA_VISIBLE_DEVICES=1,3`.
- [ ] **Step 2: Run that focused test** and verify the current fixed `=4` command fails it.
- [ ] **Step 3: Pass the scheduler allocation into `_worker_command()`** and replace the hard-coded process count only after validating the allocated count against the Controller request.
- [ ] **Step 4: Change `_validate_worker_resource_match()`** to require a valid elastic range for torchrun rather than requiring exactly 4 in all cases.
- [ ] **Step 5: Preserve `python` CPU-only launch behavior** and ensure no `CUDA_VISIBLE_DEVICES` prefix is added for an empty allocation.
- [ ] **Step 6: Relax the real worker's distributed preflight** to accept `WORLD_SIZE` 2, 3, or 4 and require the local visible CUDA device count to meet that world size; retain the existing minimum of 2 for distributed H3 execution.
- [ ] **Step 7: Run integration, scheduler, and worker-contract tests** and verify actual allocation is reflected in command, worker evidence, and events.

### Task 5: Documentation, regression tests, and static verification

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/optimization-flow.md`
- Modify: `docs/operator-contract.md`
- Modify: `docs/quickstart.md`
- Modify: `tests/unit/test_remote_scheduler.py`
- Modify: `tests/integration/test_remote_h3_closed_loop.py`

- [ ] **Step 1: Document** the distinction between Controller-authorized elasticity and Scheduler-only mapping, including the no-preemption rule.
- [ ] **Step 2: Add regression coverage** for legacy exact requests, pending-plan replay, event redaction, and loop stop reasons.
- [ ] **Step 3: Run all focused tests** with `pytest tests/unit/test_remote_scheduler.py tests/unit/test_controller_providers.py tests/integration/test_remote_h3_closed_loop.py -q`.
- [ ] **Step 4: Run the full suite** with `.venv/bin/python -m pytest -q` and run `git diff --check`.
- [ ] **Step 5: Run a remote dry/sanity campaign** only with the configured iteration bound and no process termination; verify events show queue/retry or actual allocation and that the final report distinguishes target satisfaction from iteration completion.
