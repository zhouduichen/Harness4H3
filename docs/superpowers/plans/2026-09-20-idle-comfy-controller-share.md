# Idle ComfyUI Controller Sharing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the local Controller warm on a safely idle ComfyUI card so evaluation and next-plan generation do not leave two eligible GPUs idle during the 35B model cold-start window.

**Architecture:** Extend the existing Bash launcher’s conservative GPU selection with one narrowly scoped exception: the configured primary ComfyUI process may coexist with TP1 Controller only when `/queue` is idle and that exact ComfyUI PID uses no more than `COMFY_MAX_IDLE_USED_MIB` (default 2048 MiB). Any other compute PID, active queue, benchmark lease, or rising ComfyUI memory blocks or terminates the Controller. Worker/evaluator leases remain authoritative, and no foreign process is stopped.

**Tech Stack:** Bash, `nvidia-smi`, ComfyUI HTTP API, existing remote campaign launcher, pytest static contract tests, SSH staging/activation.

## Global Constraints

- Never kill or reassign an unrelated process.
- Share only the configured primary ComfyUI PID and only while its queue is idle and its measured memory is at or below `COMFY_MAX_IDLE_USED_MIB=2048` MiB.
- A campaign-owned ComfyUI benchmark lease always blocks the card from Controller placement.
- A Controller already sharing the primary card must be terminated if the queue becomes active or the ComfyUI memory waterline is exceeded.
- Preserve TP1 fallback and the existing 300-second startup watchdog.
- Report measured power/utilization only; do not claim 300 W or 100% without samples.

---

### Task 1: Make idle primary ComfyUI sharing explicit in the launcher

**Files:**
- Modify: `tools/controller-wait-launch.sh:612-633,376-422`
- Test: `tests/unit/test_sync_remote_pipeline.py`

**Interfaces:**
- Add a shell predicate `comfy_idle_compute_pid_allowed(gpu, compute_pids)` that returns success only when the selected GPU is `COMFY_GPU_INDEX`, the configured ComfyUI process is alive, `comfy_cache_released` is true, and every compute PID on that card is the exact ComfyUI PID.
- Use the predicate in `candidate_is_quiet` so a TP1 group containing GPU0 can be selected without ignoring unrelated compute.
- Add a monitor-time guard that calls `terminate_owned_child` when a shared primary ComfyUI queue becomes active or its memory exceeds the idle threshold.

- [ ] **Step 1: Add contract assertions**

Extend `test_on_demand_comfyui_launcher_is_lease_bound` with assertions for `comfy_cache_released`, exact PID filtering, and the monitor-time guard.

- [ ] **Step 2: Run the focused test and confirm the contract is missing**

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_sync_remote_pipeline.py::test_on_demand_comfyui_launcher_is_lease_bound
```

Expected: FAIL because the launcher does not yet allow the exact idle ComfyUI PID through `candidate_is_quiet`.

- [ ] **Step 3: Implement the exact-PID exception**

Keep the existing fail-closed check for all other PIDs and add:

```bash
comfy_idle_compute_pid_allowed() {
    local gpu="$1" compute_pids="$2" comfy_pid
    [ "$gpu" -eq "$COMFY_GPU_INDEX" ] || return 1
    comfy_process_active || return 1
    comfy_cache_released || return 1
    comfy_pid=$(comfy_process_pid)
    [ -n "$comfy_pid" ] || return 1
    for pid in $compute_pids; do
        [ "$pid" = "$comfy_pid" ] || return 1
    done
    return 0
}
```

In `candidate_is_quiet`, clear `compute_pids` only after this predicate succeeds; then retain the utilization and memory checks. In `monitor_child`, when `child_gpus` includes `COMFY_GPU_INDEX`, terminate the owned vLLM child if `comfy_cache_released` becomes false.

- [ ] **Step 4: Run launcher syntax and focused tests**

Run:

```bash
bash -n tools/controller-wait-launch.sh
.venv/bin/python -m pytest -q tests/unit/test_sync_remote_pipeline.py
```

Expected: syntax passes and all sync/launcher contract tests pass.

### Task 2: Verify local regressions and document the measured layout

**Files:**
- Modify: `docs/optimization-flow.md`
- Test: `tests/unit/test_sync_remote_pipeline.py`, full test suite

**Interfaces:**
- Document the valid four-card overlap as `Controller+idle ComfyUI context on GPU0 + evaluator on GPU2 + worker on GPU1/3`; state that an active ComfyUI lease removes GPU0 from this layout.
- Keep the existing policy that worker leases and unknown processes fail closed.

- [ ] **Step 1: Add the lane-accounting note**
- [ ] **Step 2: Run `git diff --check`, Python compilation, and `pytest -q`**

Expected: `518 passed, 2 skipped` or a larger passing count if new tests are added.

### Task 3: Stage, activate, and observe the real remote boundary

**Files:**
- Remote staged copies of the files listed by `tools/sync-remote-pipeline.sh`.

**Interfaces:**
- Stage with `REMOTE_START_CONTROLLER=0`; activate only after the campaign reaches a safe restart boundary; restart only the campaign-owned Controller launcher if needed.
- Do not stop external ComfyUI 8188.

- [ ] **Step 1: Sync and activate the launcher change**
- [ ] **Step 2: Confirm the launcher log records selection of GPU0 with TP1 while GPU0 ComfyUI is idle**
- [ ] **Step 3: Observe one evaluation-to-training transition**

The acceptance evidence must show `controller_plan_prefetch_ready` before the next evaluation boundary, no `controller_plan_overlap_waiting` caused solely by Controller cold start, disjoint worker/evaluator GPU sets, and a monitor-time release if the ComfyUI queue becomes active.
