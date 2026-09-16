# Remote Pipeline GPU Utilization Implementation Plan

> **For agentic workers:** Execute the checked steps in order; each task ends with an independently testable deliverable.

**Goal:** Make the SSH-hosted H3 campaign work-conserving across LLM planning, trusted training, and ComfyUI evaluation while preserving evidence, lineage, and bounded checkpoint storage.

**Architecture:** Add a pure pipeline policy layer for safe 3+1 packing and full-4-GPU pre-plan fallback, then integrate it into `RemoteCampaign` at the existing worker/evaluation boundaries. Campaign-owned ComfyUI processes become durable, tracked, and stoppable after verified model unload; all leases and pipeline cursors remain atomically persisted. Existing Controller/provider, worker, evaluator, archive, and retention contracts remain authoritative.

**Tech Stack:** Python 3.9+, dataclasses, JSON/YAML, existing SSH/nvidia-smi transport, ComfyUI HTTP API, pytest; no new runtime dependency.

## Global Constraints

- All experiment, worker, ComfyUI, and local LLM execution stays on the remote SSH host.
- Controller output is structured data only; trusted worker commands remain configuration-owned.
- Scheduler changes only happen at worker boundaries and never stop or reassign foreign processes.
- The current model, accepted lineage, evaluator, TargetProfile, and quality gates remain authoritative.
- Raw media, full logs, and checkpoint payloads never enter the LLM prompt; only bounded summaries and references do.
- Rejected/failed checkpoint payloads are removable only inside the campaign result root; evidence and recipe records remain append-only.
- 300 W and 100% GPU utilization are telemetry/optimization targets, never overrides for thermal, memory, power, or isolation gates.

---

### Task 1: Add a pure work-conserving pipeline policy

**Files:**
- Create: `harness4h3/remote/pipeline.py`
- Create: `tests/unit/test_remote_pipeline.py`

**Interfaces:**
- Produces `PipelineStage`, `PipelineState`, `PipelineAllocation`, `pack_overlap_resources()`, and `evaluation_gpu_count()` for the campaign integration.
- Consumes a total GPU count, live reserved indices, and a validated elastic resource request; it does not call SSH or choose an operator.

- [ ] **Step 1: Write failing tests**

```python
from harness4h3.remote.pipeline import evaluation_gpu_count, pack_overlap_resources


def test_three_training_gpus_leave_one_for_controller_or_evaluation():
    result = pack_overlap_resources(4, reserved=(0,), minimum_training_gpus=2, maximum_training_gpus=4)
    assert result.training_gpus == (1, 2, 3)
    assert result.free_gpus == ()
    assert result.mode == "evaluation_plus_training"


def test_full_training_requires_plan_before_worker_lease():
    result = pack_overlap_resources(4, reserved=(), minimum_training_gpus=4, maximum_training_gpus=4)
    assert result.training_gpus == (0, 1, 2, 3)
    assert result.plan_must_be_ready_before_training is True


def test_evaluation_worker_count_never_exceeds_tasks_or_cards():
    assert evaluation_gpu_count(task_count=1, configured_workers=4, free_gpu_count=4) == 1
    assert evaluation_gpu_count(task_count=4, configured_workers=4, free_gpu_count=3) == 3
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python3 -m pytest -q tests/unit/test_remote_pipeline.py`

Expected: FAIL with `ModuleNotFoundError` for `harness4h3.remote.pipeline`.

- [ ] **Step 3: Implement the minimal policy**

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class PipelineStage(str, Enum):
    IDLE = "idle"
    EVALUATING = "evaluating"
    TRAINING = "training"
    PREFETCHING = "prefetching"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class PipelineAllocation:
    training_gpus: Tuple[int, ...]
    free_gpus: Tuple[int, ...]
    mode: str
    plan_must_be_ready_before_training: bool


@dataclass(frozen=True)
class PipelineState:
    stage: PipelineStage = PipelineStage.IDLE
    iteration: int = 0
    evaluation_model_id: Optional[str] = None
    training_model_id: Optional[str] = None
    updated_at: float = 0.0


def pack_overlap_resources(total_gpu_count, reserved=(), minimum_training_gpus=2, maximum_training_gpus=4):
    available = tuple(index for index in range(total_gpu_count) if index not in set(reserved))
    if len(available) < minimum_training_gpus:
        return PipelineAllocation((), available, "waiting", False)
    count = min(len(available), maximum_training_gpus)
    training = available[-count:]
    free = tuple(index for index in available if index not in training)
    return PipelineAllocation(training, free, "evaluation_plus_training" if reserved else "training", count == total_gpu_count)


def evaluation_gpu_count(task_count, configured_workers, free_gpu_count):
    return max(0, min(int(task_count), int(configured_workers), int(free_gpu_count)))
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python3 -m pytest -q tests/unit/test_remote_pipeline.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/remote/pipeline.py tests/unit/test_remote_pipeline.py
git commit -m "feat: add work-conserving pipeline policy"
```

### Task 2: Add remote pipeline and ComfyUI lifecycle configuration

**Files:**
- Modify: `harness4h3/remote/config.py:52-112,113-298`
- Modify: `configs/remote-l40-h3-rsi-overnight.yaml`
- Modify: `tests/unit/test_remote_config.py`

**Interfaces:**
- `RemoteCampaignConfig.pipeline_enabled`, `pipeline_max_inflight`, `prefetch_before_full_training`, `comfyui_process_policy`, `comfyui_idle_shutdown_s`, and `power_target_w` are typed configuration fields.
- Valid `comfyui_process_policy` values are `on_demand` and `persistent_api`; the overnight profile selects `on_demand`.

- [ ] **Step 1: Add failing configuration tests**

```python
def test_overnight_config_enables_pipeline_and_on_demand_comfyui():
    config = load_remote_campaign_config("configs/remote-l40-h3-rsi-overnight.yaml")
    assert config.pipeline_enabled is True
    assert config.prefetch_before_full_training is True
    assert config.comfyui_process_policy == "on_demand"
    assert config.comfyui_idle_shutdown_s > 0
    assert config.power_target_w == 300.0
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python3 -m pytest -q tests/unit/test_remote_config.py::test_overnight_config_enables_pipeline_and_on_demand_comfyui`

Expected: FAIL because the fields do not exist.

- [ ] **Step 3: Implement validation and defaults**

Parse an optional `pipeline` mapping after `benchmark` and validate:

```python
pipeline_enabled = bool(pipeline_raw.get("enabled", True))
pipeline_max_inflight = int(pipeline_raw.get("max_inflight", 1))
if pipeline_max_inflight != 1:
    raise RemoteConfigError("pipeline.max_inflight must be 1")
prefetch_before_full_training = bool(pipeline_raw.get("prefetch_before_full_training", True))
comfyui_process_policy = str(pipeline_raw.get("comfyui_process_policy", "on_demand")).strip()
if comfyui_process_policy not in {"on_demand", "persistent_api"}:
    raise RemoteConfigError("pipeline.comfyui_process_policy must be on_demand or persistent_api")
comfyui_idle_shutdown_s = _positive(pipeline_raw.get("comfyui_idle_shutdown_s", 30.0), "pipeline.comfyui_idle_shutdown_s")
power_target_w = _positive(pipeline_raw.get("power_target_w", 300.0), "pipeline.power_target_w")
```

Return those values in `RemoteCampaignConfig` and configure the overnight YAML with `enabled: true`, `max_inflight: 1`, `prefetch_before_full_training: true`, `comfyui_process_policy: on_demand`, `comfyui_idle_shutdown_s: 30`, and `power_target_w: 300`.

- [ ] **Step 4: Run configuration tests**

Run: `python3 -m pytest -q tests/unit/test_remote_config.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/remote/config.py configs/remote-l40-h3-rsi-overnight.yaml tests/unit/test_remote_config.py
git commit -m "feat: configure remote pipeline lifecycle"
```

### Task 3: Track and stop campaign-owned ComfyUI workers

**Files:**
- Modify: `harness4h3/remote/comfyui_lease.py:28-228`
- Modify: `research/experiments/remote_h3_closed_loop.py:1211-1247,4117-4145`
- Create: `tests/unit/test_comfyui_process_lifecycle.py`

**Interfaces:**
- `ComfyUILeaseManager.release_if_idle(stop_process=False)` retains existing release semantics and optionally stops only the PID recorded by the campaign.
- `ComfyUILeaseManager.stop_owned_process()` returns `{"status": "stopped"|"not_owned"|"already_stopped", "pid": ...}` and never issues a broad process kill.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_idle_release_stops_only_owned_comfyui_pid(fake_ssh):
    lease = make_lease(fake_ssh, process_policy="on_demand")
    lease.set_owned_process(4321)
    result = lease.stop_owned_process()
    assert result["status"] == "stopped"
    assert ("kill", "-TERM", "4321") in fake_ssh.commands


def test_process_without_owned_pid_is_not_killed(fake_ssh):
    lease = make_lease(fake_ssh, process_policy="on_demand")
    assert lease.stop_owned_process()["status"] == "not_owned"
    assert not any(command[0] == "kill" for command in fake_ssh.commands)
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest -q tests/unit/test_comfyui_process_lifecycle.py`

Expected: FAIL because the ownership API does not exist.

- [ ] **Step 3: Implement PID ownership and graceful shutdown**

Add `_owned_pid: Optional[int]`, `set_owned_process(pid)`, and `stop_owned_process()`; first probe `kill -0 PID`, send `TERM`, wait for `kill -0` to fail, and send `KILL` only to the same recorded PID after the bounded wait. Clear the PID only after the process is gone. Call it after successful `/free` and waterline verification when `comfyui_process_policy == "on_demand"`; preserve the lease marker until release metadata is written.

- [ ] **Step 4: Run focused lifecycle and existing lease tests**

Run: `python3 -m pytest -q tests/unit/test_comfyui_process_lifecycle.py tests/unit/test_comfyui_lease.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/remote/comfyui_lease.py research/experiments/remote_h3_closed_loop.py tests/unit/test_comfyui_process_lifecycle.py
git commit -m "feat: stop campaign-owned ComfyUI workers after idle release"
```

### Task 4: Integrate 3+1 overlap and full-GPU plan fallback

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py:3293-3719,3917-4105`
- Modify: `harness4h3/remote/scheduler.py:281-355`
- Create: `tests/integration/test_remote_pipeline_overlap.py`

**Interfaces:**
- `RemoteCampaign._plan_before_full_training(parent, plan, training_calls)` returns a validated `ExperimentPlan` or `None` and emits `controller_plan_prefetch_ready_before_training`.
- `RemoteCampaign._start_speculative_worker(...)` receives the current evaluation reservation and uses the largest safe elastic allocation from the remaining GPUs.
- `RemoteResourceScheduler.acquire()` continues returning `ResourceDecision`; no caller may assume a fixed GPU index.

- [ ] **Step 1: Write failing integration tests**

```python
def test_full_training_uses_plan_prefetched_before_worker_lease(campaign):
    campaign.scheduler = FakeScheduler(free=(0, 1, 2, 3))
    plan = make_four_gpu_plan()
    prepared = campaign._plan_before_full_training(campaign.models.active(), plan, 0)
    assert prepared is not None
    assert any(event["event_type"] == "controller_plan_prefetch_ready_before_training" for event in campaign.events.read())


def test_evaluation_reservation_starts_successor_on_all_remaining_gpus(campaign):
    campaign.scheduler = FakeScheduler(free=(1, 2, 3))
    handle = campaign._start_speculative_worker(campaign.models.active(), make_elastic_plan())
    assert handle["resource_decision"]["allocated_gpus"] == [1, 2, 3]
    assert "--nproc_per_node=3" in handle["command"]
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest -q tests/integration/test_remote_pipeline_overlap.py`

Expected: FAIL because full-training prefetch and allocation assertions are not present.

- [ ] **Step 3: Integrate the policy at the worker boundary**

Before acquiring a plan whose normalized request has `min_gpu_count == 4` and `prefetch_before_full_training` is enabled, call `_controller_plan` against the deterministic predicted child state and persist the plan with a `prefetched_before_training` marker. Do not acquire a worker lease until that call has either produced a validated plan or recorded a bounded unavailable event. During evaluation, pass the reserved ComfyUI GPU indices into the scheduler and let an elastic successor request use the largest safe remaining set. Record `pipeline_stage`, `reserved_gpu_indices`, `allocated_gpus`, `nproc_per_node`, and `overlap_with_model_id` in both the event stream and `campaign_state.json`.

When no safe successor exists, continue with the existing serialized path; do not manufacture a duplicate metric workload.

- [ ] **Step 4: Run the full remote test subset**

Run: `python3 -m pytest -q tests/unit/test_remote_pipeline.py tests/unit/test_remote_scheduler.py tests/unit/test_comfyui_lease.py tests/integration/test_remote_pipeline_overlap.py tests/integration/test_remote_h3_closed_loop.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/remote/scheduler.py research/experiments/remote_h3_closed_loop.py tests/integration/test_remote_pipeline_overlap.py
git commit -m "feat: overlap remote training with evaluation"
```

### Task 5: Add remote autonomous service entrypoint and utilization evidence

**Files:**
- Modify: `tools/run_overnight_controller.py:106-168`
- Create: `tools/remote-campaign-service.sh`
- Create: `tests/unit/test_remote_campaign_service.py`
- Modify: `docs/quickstart.md`
- Modify: `README.md`

**Interfaces:**
- `tools/remote-campaign-service.sh start|status|stop` manages only the configured campaign PID file and log; `start` launches `run_overnight_controller.py --on-server` with `nohup`, `status` reads the PID and durable campaign state, and `stop` sends TERM to that recorded PID.
- The final campaign report contains `pipeline` with `overlap_cycles`, `gpu_seconds_by_index`, `power_target_w`, `power_peak_w`, `utilization_avg_pct_by_index`, and `idle_seconds_by_index`.

- [ ] **Step 1: Write failing service/report tests**

```python
def test_service_script_has_scoped_start_status_stop_commands():
    text = Path("tools/remote-campaign-service.sh").read_text()
    assert "start|status|stop" in text
    assert "run_overnight_controller.py" in text
    assert "CAMPAIGN_PID_FILE" in text


def test_report_exposes_per_gpu_utilization_and_power_target(campaign):
    report = campaign._pipeline_report()
    assert report["power_target_w"] == 300.0
    assert set(report["utilization_avg_pct_by_index"]) == {"0", "1", "2", "3"}
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest -q tests/unit/test_remote_campaign_service.py`

Expected: FAIL because the service script and report method do not exist.

- [ ] **Step 3: Implement scoped remote service and report**

Use explicit `CAMPAIGN_ROOT`, `CAMPAIGN_CONFIG`, `CAMPAIGN_OUTPUT`, and `CAMPAIGN_PID_FILE` variables. Refuse to start when the PID file names a live process, write the PID only after launch, and remove it on normal exit. Extend `RemotePowerSampler` to retain per-GPU samples and compute average utilization/power from fixed `nvidia-smi` queries; include missing data as `null` rather than zero.

- [ ] **Step 4: Run tests and compile checks**

Run: `python3 -m pytest -q tests/unit/test_remote_campaign_service.py tests/unit/test_overnight_controller.py tests/integration/test_remote_h3_closed_loop.py && python3 -m compileall -q harness4h3 tools research`

Expected: PASS and no compile errors.

- [ ] **Step 5: Commit**

```bash
git add tools/remote-campaign-service.sh tools/run_overnight_controller.py tests/unit/test_remote_campaign_service.py docs/quickstart.md README.md
git commit -m "feat: run remote campaign as an autonomous service"
```

### Task 6: Remote smoke verification and final audit

**Files:**
- Modify: `docs/validation-plan.md`
- Create: `research/evidence/remote-pipeline-utilization-2026-09-16.md`

- [ ] **Step 1: Run offline verification**

Run: `python3 -m pytest -q` using the server training environment that has PyTorch installed; on the Mac, record the existing PyTorch collection limitation rather than treating skipped/failed collection as GPU evidence.

- [ ] **Step 2: Run a bounded remote smoke campaign**

On the SSH server, run:

```bash
bash tools/remote-campaign-service.sh start
bash tools/remote-campaign-service.sh status
```

Wait for one complete worker/evaluation boundary, then inspect `controller-events.jsonl`, `campaign_state.json`, `evaluations.json`, and the remote GPU telemetry. Verify no foreign process was killed, ComfyUI model memory was released, and the next plan was persisted.

- [ ] **Step 3: Record evidence**

Record the exact config hash, campaign command, active model/checkpoint hashes, resource decisions, per-GPU telemetry, plan-prefetch event, ComfyUI release result, retention results, and any failure classification. Do not claim 300 W or 100% unless measured.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/validation-plan.md research/evidence/remote-pipeline-utilization-2026-09-16.md
git commit -m "docs: record remote pipeline validation"
```

## Self-Review Checklist

- [ ] Every requirement in the design maps to a task: remote autonomy (Task 5), overlap (Task 4), bounded context and experience reuse (existing provider plus Task 4), checkpoint retention (existing retention plus Task 6 audit), dynamic GPU allocation (Tasks 1 and 4), ComfyUI unload (Task 3), and utilization evidence (Tasks 5 and 6).
- [ ] No task deletes outside the campaign root or kills an unowned process.
- [ ] No task treats a synthetic benchmark repeat as real quality evidence.
- [ ] All later interfaces match earlier names and types.
- [ ] No unresolved placeholders or unbounded context requirements remain.
