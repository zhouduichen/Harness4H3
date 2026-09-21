# Evaluation-Phase GPU Fill Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-generate a validated GPU successor during CPU-only training so evaluation can immediately use the two non-ComfyUI cards instead of waiting for a second LLM request.

**Architecture:** Reuse the existing `parallel_prefetched_plan` durable cursor and cloned Controller provider. Add an eager parallel prefetch job keyed by `(source_child_model_id, training_calls)`; the evaluation callback consumes that job or its persisted plan through the existing scheduler and speculative-worker gates.

**Tech Stack:** Python 3, existing `RemoteCampaign`, JSON campaign state, SSH GPU scheduler, pytest.

## Global Constraints

- The remote campaign remains operator-paused during implementation and synchronization.
- Only campaign-owned leases may be released; foreign processes are never stopped.
- A plan must pass the existing schema, operator registry, worker contract, and resource scheduler before launch.
- A single benchmark task is not duplicated merely to inflate utilization.
- 300 W and 100% utilization are telemetry targets and require live per-GPU evidence.

---

### Task 1: Add an eager GPU-fill prefetch job

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py` near `_arm_parallel_prefetch_after_primary` and `_train_one`
- Test: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- Add `_arm_eager_parallel_gpu_prefetch(parent, current_plan, current_child_id, training_calls) -> Optional[Mapping[str, Any]]`.
- It returns a handle with `parallel_handle` and `done`, or `None`; it stores the handle in the existing `_early_parallel_prefetch_jobs` map and `parallel_prefetched_plan` cursor.

- [x] **Step 1: Add a failing test**

  Build a CPU-only `prune_blocks`/`quantize` worker boundary with
  `pipeline_max_inflight=2`, replace `_start_controller_prefetch` with a
  recorder returning a fake parallel handle, and assert the call uses:

  ```python
  {
      "planning_intent": "parallel_gpu_fill",
      "prefetch_state_key": "parallel",
      "operator_filter": ("recovery_finetune", "distill", "step_distill", "dmd2"),
  }
  ```

- [x] **Step 2: Run the focused test**

  Run `./.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k eager_parallel_gpu_prefetch`.
  It must fail before the helper exists.

- [x] **Step 3: Implement the helper**

  Use the existing `key = (str(current_child_id), int(training_calls) + 1)`;
  return an existing job when present; otherwise call
  `_start_controller_prefetch(parent, training_calls, source_plan=current_plan,
  source_experiment_id=current_plan.experiment_id,
  source_child_model_id=current_child_id, planning_intent="parallel_gpu_fill",
  operator_filter=("recovery_finetune", "distill", "step_distill", "dmd2"),
  prefetch_state_key="parallel")`, append the handle to
  `_deferred_prefetch_handles`, and set `done` when its thread exits.

- [x] **Step 4: Run the focused test**

  The eager helper test and existing parallel-prefetch tests must pass.

### Task 2: Start the queue during CPU-only training and consume it at evaluation

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py` in `_train_one` and the local `start_parallel_gpu_fill`
- Test: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- CPU-only trusted workers call the eager helper alongside the primary
  successor prefetch.
- Evaluation first consumes the matching early job/persisted plan, then falls
  back to the existing batched candidate or one filtered parallel request.

- [x] **Step 1: Add a failing consumption assertion**

  Seed a matching `parallel_prefetched_plan` for a one-task evaluation and
  assert `controller_plan_parallel_reused` precedes
  `speculative_worker_started`, whose allocation contains exactly the two
  non-ComfyUI GPUs.

- [x] **Step 2: Run the focused evaluation test**

  Run `./.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k parallel_gpu_fill`.

- [x] **Step 3: Wire eager and persisted paths without duplicate LLM calls**

  In `_train_one`, for a CPU-only plan call the eager helper after the primary
  prefetch is started; do not also call the sequential helper for the same key.
  In `start_parallel_gpu_fill`, prefer the early job, then the persisted plan,
  then an eligible n-way candidate, and only then issue one filtered fallback
  request. Record `parallel_gpu_fill_waiting`, `parallel_gpu_fill_ready`, or
  `parallel_gpu_fill_skipped` with available/blocked GPU indices and reason.

- [x] **Step 4: Run focused integration tests**

  Run `./.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k 'parallel_gpu_fill or speculative_worker or pipeline_evaluation'`.

### Task 3: Document lane accounting and verify remotely while paused

**Files:**
- Modify: `docs/superpowers/specs/2026-09-18-a-evolve-round-policy-design.md`
- Modify: `research/evidence/remote-pipeline-utilization-2026-09-16.md`
- Test: `tests/unit/test_remote_pipeline.py`

- [x] **Step 1: Document the eager queue and no-claim rule**

  State that CPU-only work does not count as GPU fill, and that a safe
  `ComfyUI 1 + Controller 1 + trial 2` boundary is the target only when the
  scheduler reports two free cards.

- [x] **Step 2: Run static and full regression checks**

  Run `python3 -m py_compile research/experiments/remote_h3_closed_loop.py`,
  `bash -n tools/*.sh`, `git diff --check`, and `./.venv/bin/python -m pytest -q`.

- [x] **Step 3: Sync without starting services**

  Run `REMOTE_START_CONTROLLER=0 tools/sync-remote-pipeline.sh`, activate the
  staged files over `ssh Jiayu-intern`, and verify `state=paused`, campaign
  `pid=none`, idle watcher `pid=none`, and `.operator-paused` present.

### Acceptance

- The current worker can finish while the next primary and GPU-fill plans are
  being generated concurrently.
- A matching GPU-fill plan is durable and launchable at evaluation start;
  there is no second same-boundary LLM call caused by a missing cursor.
- All actual GPU claims remain disjoint and fail closed on foreign activity.
