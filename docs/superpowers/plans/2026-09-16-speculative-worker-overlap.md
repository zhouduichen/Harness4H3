# Speculative Worker Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a validated next experiment on GPUs left free by ComfyUI while the current candidate is being benchmarked, without promoting or retaining an invalid speculative child.

**Architecture:** Add a campaign-local speculative worker handle around the existing trusted worker command. The handle owns its lease, exact result paths, background process/thread, and state record; the main loop starts it only after the current candidate's benchmark has reserved ComfyUI, then joins, imports, or discards it after the benchmark decision. The active model and Pareto archive remain changed only by the existing evaluation decision path.

**Tech Stack:** Python `threading`, existing `RemoteResourceScheduler`, `SSHClient`, `RemoteCheckpointRetention`, JSON campaign state, pytest.

## Global Constraints

- Do not stop or signal unrelated remote processes.
- GPU0 is unavailable to speculative training while the ComfyUI lease is active.
- Distributed H3 worker allocations must remain within 2–4 GPUs and obey the Controller resource request.
- Speculative results are not active or Pareto-promoted before the current benchmark decision.
- Checkpoint deletion is restricted to the existing campaign retention policy and exact child paths.
- Preserve all pre-existing user changes in the dirty worktree.

---

### Task 1: Add explicit speculative state and plan rebasing

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py`
- Test: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- Produces `_speculative_plan_for_candidate(plan, candidate)` and state serialization fields consumed by the worker lifecycle.
- Preserves `ExperimentPlan` validation and existing prefetch reuse behavior.

- [ ] **Step 1: Write the failing tests** for rebasing a prefetched plan to the candidate being benchmarked and for refusing a plan with an unrelated source child.
- [ ] **Step 2: Run the focused tests** and verify the new behavior is absent.
- [ ] **Step 3: Implement deterministic parent/source-child validation and `replace()`-based rebasing without changing operator arguments or evidence ids.
- [ ] **Step 4: Run the focused tests** and verify both cases pass.

### Task 2: Launch an isolated worker during evaluation

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py`
- Test: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- Adds a campaign-local speculative handle with `start`, `join`, `cancel`, and `finalize` behavior.
- Reuses `_worker_command`, `RemoteResourceScheduler.acquire`, lease release, result import, and retention helpers.

- [ ] **Step 1: Write a test** that starts evaluation with GPU0 reserved and observes `speculative_worker_started` with an allocation drawn from the remaining GPUs.
- [ ] **Step 2: Run the focused test** and verify it fails because evaluation is currently synchronous.
- [ ] **Step 3: Implement the background worker lifecycle. Keep remote result paths unique, persist state before launch, and prevent speculative execution from modifying active/Pareto state.
- [ ] **Step 4: Join the worker after evaluation and import its result exactly once as an unevaluated experience.
- [ ] **Step 5: Run the focused tests** and verify the current candidate remains active until its evaluation decision.

### Task 3: Implement discard, cancellation, and recovery

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py`
- Modify: `harness4h3/remote/checkpoint_retention.py` only if exact speculative cleanup needs a narrowly scoped helper.
- Test: `tests/integration/test_remote_h3_closed_loop.py`
- Test: `tests/unit/test_remote_checkpoint_retention.py`

**Interfaces:**
- Emits `speculative_worker_completed`, `speculative_worker_promoted`, `speculative_worker_discarded`, and `speculative_worker_recovered`.
- Uses exact child paths and existing retention policy for discarded output.

- [ ] **Step 1: Write tests** for rejection/replan discard, worker cancellation, and completed-state recovery.
- [ ] **Step 2: Run the focused tests** and verify failure.
- [ ] **Step 3: Implement exact cleanup and state transitions; never infer promotion from a checkpoint file.
- [ ] **Step 4: Run the focused tests** and verify no unrelated file or process is touched.

### Task 4: Verify the full controller loop and remote boundary

**Files:**
- Modify: `docs/architecture.md` and `docs/optimization-flow.md` if the final state transition needs documentation.
- Test: `tests/integration/test_remote_h3_closed_loop.py`

- [ ] **Step 1: Run the complete local test suite:** `.venv/bin/pytest -q`.
- [ ] **Step 2: Run `git diff --check` and inspect only the intended changes.
- [ ] **Step 3: Sync the implementation files to `/home/intern/huangjiahao/Harness4H3-rsi` without touching the original remote repository.
- [ ] **Step 4: Start or resume the remote supervisor only after checking for an existing worker and lease.
- [ ] **Step 5: Verify events show `evaluation_started` followed by `speculative_worker_started` on non-ComfyUI GPUs, then verify the current candidate remains unpromoted until evaluation completes.
