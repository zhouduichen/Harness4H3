# Dynamic Four-GPU Worker Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let a distributed H3 worker use all four GPUs when its validated plan requests a full-card round, while preserving the normal `3 worker + 1 Controller` overlap and generating the successor plan before releasing the Controller.

**Architecture:** Treat a full-card request as an explicit execution mode derived from the validated worker resource request, not as an LLM command to kill or preempt anything. Before acquiring a full-card lease, the campaign finishes and durably records the next Controller plan, places a handoff hold, releases only the campaign-owned Controller lease, then acquires all available GPUs through the existing fail-closed scheduler. Requests that do not explicitly allow four GPUs continue through the existing lane packer and retain the Controller overlap lane.

**Tech Stack:** Python 3, dataclass-based campaign configuration, `RemoteResourceScheduler`, SSH lease markers, pytest, shell/Python static checks.

## Global Constraints

- Do not start, stop, or inspectively modify foreign GPU processes.
- Do not force `nvidia-smi -pl`; 300 W and 100% utilization remain measured targets.
- A full-card worker may start only after its successor Controller plan is validated and durably persisted.
- A worker lease is acquired only through `RemoteResourceScheduler`; an external process or unknown GPU mapping causes a wait.
- Preserve the existing bounded context, checkpoint retention, ComfyUI unload, and append-only experience records.
- The remote campaign remains operator-paused during implementation and synchronization.

---

### Task 1: Encode the full-card execution mode in request shaping

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py:3737-3764`
- Test: `tests/integration/test_remote_h3_closed_loop.py:1354-1422`

**Interfaces:**
- Consumes: validated `ExperimentPlan.resource_request` and `RemoteCampaignConfig.controller_overlap_gpus`.
- Produces: `_effective_worker_request(plan) -> (planned_request, effective_request)` where an explicit distributed `gpu_count` or `min_gpu_count` at the scheduler GPU count retains the full-card request; smaller/ordinary elastic requests still use `pack_worker_request()`.

- [x] **Step 1: Add the failing assertion for a full-card request**

  Extend the existing speculative resource test with a separate full-card `ExperimentPlan` and assert:

  ```python
  planned_full, effective_full = campaign._effective_worker_request(full_card_plan)
  assert planned_full["max_gpu_count"] == 4
  assert effective_full["max_gpu_count"] == 4
  assert effective_full["gpu_count"] == 4
  ```

- [x] **Step 2: Run the focused test to verify the old cap is observed**

  Run:

  ```bash
  ./.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k 'effective_worker_request or speculative_worker'
  ```

  Expected before implementation: the new full-card assertion fails because `pack_worker_request()` changes `max_gpu_count` from 4 to 3.

- [x] **Step 3: Preserve full-card requests and retain the existing packer for overlap requests**

  In `_effective_worker_request`, use the scheduler GPU count and the normalized request:

  ```python
  total_gpu_count = int(getattr(self.scheduler, "gpu_count", 4))
  distributed = bool(planned.get("distributed"))
  requests_full_card = distributed and (
      int(planned.get("gpu_count", 0)) >= total_gpu_count
      or int(planned.get("min_gpu_count", 0)) >= total_gpu_count
  )
  if self.config.pipeline_enabled and int(self.config.controller_overlap_gpus) > 0 and not requests_full_card:
      effective = dict(
          pack_worker_request(
              planned,
              total_gpu_count=total_gpu_count,
              controller_overlap_gpus=int(self.config.controller_overlap_gpus),
          )
      )
  ```

  Emit `worker_resource_request_full_card` when `requests_full_card` is true, including both request mappings and the reason `plan_requests_full_card_before_controller_handoff`. Keep `pack_worker_request()` unchanged so its standalone safety contract remains explicit.

- [x] **Step 4: Run the focused tests**

  Run:

  ```bash
  ./.venv/bin/python -m pytest -q tests/unit/test_remote_pipeline.py tests/integration/test_remote_h3_closed_loop.py -k 'effective_worker_request or speculative_worker'
  ```

  Expected: existing `3-card + Controller` assertions and the new full-card assertion pass.

### Task 2: Release the Controller only for a preplanned full-card worker

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py:5890-6045`
- Test: `tests/integration/test_remote_h3_closed_loop.py:1760-1810`

**Interfaces:**
- Consumes: `effective_request` from Task 1, the existing Controller handoff marker, and `_finish_controller_prefetch()`.
- Produces: a full-card boundary that has a durable successor plan before Controller release and a `controller_release_skipped` event only for the normal overlap mode.

- [x] **Step 1: Add a test that a full-card request forces a Controller handoff**

  Configure a fake campaign with `controller_reserved_gpu_indices = (3,)`, a four-GPU elastic plan, and a ready scheduler. Assert that `_request_controller_release()` is called with `training_gpu_allocation`, that the event contains `full_card_training=True`, and that the worker request remains `max_gpu_count == 4`.

- [x] **Step 2: Run the new focused test and observe the current behavior**

  Run:

  ```bash
  ./.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k 'full_card or controller_release'
  ```

  Expected before implementation: the current code keeps the TP1 Controller because the unconditional packer makes the request look like a three-card worker.

- [x] **Step 3: Make full-card mode override `can_keep_controller_lane`**

  After computing `full_gpu_training`, change the predicate to require `not full_gpu_training`:

  ```python
  can_keep_controller_lane = bool(
      self.config.pipeline_enabled
      and distributed_training
      and not full_gpu_training
      and int(self.config.controller_overlap_gpus) > 0
      and not bool(request_for_pipeline.get("exclusive"))
      and controller_reserved is not None
      and len(controller_reserved) <= total_gpu_count - int(request_for_pipeline.get("min_gpu_count", 0))
  )
  ```

  Set `full_gpu_training` from `request_for_pipeline["max_gpu_count"] >= total_gpu_count`, not from the original plan. Keep `handoff_hold=True` for this path. The existing release result gate remains authoritative: a failed release persists the plan as pending and returns without acquiring any worker GPUs.

- [x] **Step 4: Ensure plan prefetch happens before the full-card lease**

  Keep the existing `prefetch_before_full_training` branch before Controller release. Add `full_card_training` to `controller_release_requested`, `controller_release_blocked`, `resource_scheduled`, and `worker_started` event payloads so later telemetry can distinguish a deliberate four-card phase from an ordinary 3+1 overlap.

- [x] **Step 5: Run the focused integration suite**

  Run:

  ```bash
  ./.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k 'full_card or controller_release or speculative_worker'
  ```

  Expected: full-card tests pass; existing Controller handoff and worker lease tests remain green.

### Task 3: Document and verify the two safe lane modes

**Files:**
- Modify: `configs/remote-l40-h3-rsi-overnight.yaml: pipeline comments`
- Modify: `docs/superpowers/specs/2026-09-17-four-gpu-lane-packer-design.md`
- Modify: `research/evidence/four-gpu-lane-packer-2026-09-17.md`
- Test: `tests/unit/test_remote_pipeline.py`

**Interfaces:**
- Consumes: the events and request shaping from Tasks 1–2.
- Produces: operator-facing documentation that distinguishes measured `3+1` overlap from preplanned full-card training and does not claim 300 W/100% without telemetry.

- [x] **Step 1: Add pure lane-mode assertions**

  Keep the existing `pack_worker_request()` test for normal overlap and add a pure assertion that `pack_overlap_resources(4, reserved=(), minimum_training_gpus=4, maximum_training_gpus=4)` returns all four cards with `plan_must_be_ready_before_training=True`.

- [x] **Step 2: Update configuration comments**

  State that `controller_overlap_gpus: 1` is the default for ordinary elastic rounds, while a validated `gpu_count` or `min_gpu_count` of 4 round prefetches the successor and temporarily releases the Controller to claim all four cards.

- [x] **Step 3: Update the evidence/spec language**

  Replace the unconditional “default max is 3” wording with the two-mode policy, and explicitly state that actual power/utilization evidence is required before reporting success.

- [x] **Step 4: Run documentation/static checks**

  Run:

  ```bash
  python3 -m py_compile research/experiments/remote_h3_closed_loop.py
  bash -n tools/*.sh
  git diff --check
  ```

### Task 4: Full regression and paused remote synchronization

**Files:**
- No new source files.

- [x] **Step 1: Run the full local regression**

  Run:

  ```bash
  ./.venv/bin/python -m pytest -q
  ```

  Expected: all existing tests pass; CUDA-only tests may remain skipped when CUDA is unavailable locally.

- [x] **Step 2: Stage and activate remotely without starting services**

  Run:

  ```bash
  REMOTE_START_CONTROLLER=0 tools/sync-remote-pipeline.sh
  ssh Jiayu-intern "REMOTE_START_CONTROLLER=0 REMOTE_PIPELINE_STAGE_ROOT=/home/intern/huangjiahao/Harness4H3-rsi/work/remote-pipeline-v2-20260918 bash /home/intern/huangjiahao/Harness4H3-rsi/work/remote-pipeline-v2-20260918/tools/activate-remote-pipeline.sh"
  ```

- [x] **Step 3: Verify the remote remains paused**

  Confirm `remote-campaign-service.sh status` reports `state=paused`, the idle watcher is absent, and no campaign-owned Controller/worker process is running. Do not start an experiment in this task. (The status check reports campaign `pid=none` and idle watcher `pid=none`.)

- [ ] **Step 4: Leave live validation as the next gate**

  After the operator removes the pause, collect per-GPU power/utilization and lane events over a real boundary. A successful implementation must show either `3 worker + 1 Controller` or `4 worker` with disjoint leases; it must not infer 300 W or 100% from configuration alone.
