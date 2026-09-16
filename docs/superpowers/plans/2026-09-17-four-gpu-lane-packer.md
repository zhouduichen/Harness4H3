# Four-GPU Lane Packer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在远程评价、Controller 规划和训练之间实现安全的动态 GPU lane packing，消除 plan 生成期间的无主空卡。

**Architecture:** 新增纯函数 lane packer 计算有效 worker 请求；scheduler 通过 Controller lease marker 排除正在加载/运行的 vLLM GPU；campaign 在 full-GPU 训练请求上保留一张 Controller lane，并在评价期间让 elastic speculative worker 使用剩余水位。第一阶段不抢占运行中的分布式 rank；第二阶段再增加 checkpoint/resume 扩缩容。

**Tech Stack:** Python 3、dataclasses、pytest、Bash、nvidia-smi、SSH/local command transport、vLLM launcher。

## Global Constraints

- 不停止未被当前 campaign 精确记录的进程。
- 分布式训练只在完整 worker lease 内运行；不在 rank 中途改变 GPU 可见性。
- Controller、ComfyUI 和 worker 使用过期可回收的精确 lease marker。
- checkpoint 只保留 active、Pareto rollback 和 in-flight 分支；experience/evaluation/telemetry append-only。
- 300W/100% 只作为目标与实测指标，不使用功耗上限强制命令。

---

### Task 1: Add pure lane packing and effective request rules

**Files:**
- Create: `harness4h3/remote/lane_packer.py`
- Modify: `harness4h3/remote/config.py:108-125,295-356`
- Modify: `configs/remote-l40-h3-rsi-overnight.yaml:50-61`
- Test: `tests/unit/test_remote_pipeline.py`
- Test: `tests/unit/test_remote_config.py`

**Interfaces:**
- `pack_worker_request(request: Mapping[str, Any], total_gpu_count: int, controller_overlap_gpus: int) -> Mapping[str, Any]` returns a validated effective request and never lowers `min_gpu_count`.
- `RemoteCampaignConfig.controller_overlap_gpus: int` defaults to `1` and is loaded from `pipeline.controller_overlap_gpus`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pack_worker_request_caps_four_gpu_plan_to_three_for_controller():
    effective = pack_worker_request(
        {"gpu_count": 4, "min_gpu_count": 2, "max_gpu_count": 4,
         "elastic": True, "distributed": True, "exclusive": True,
         "evaluation_workers": 0, "on_unavailable": "wait"},
        total_gpu_count=4,
        controller_overlap_gpus=1,
    )
    assert effective["gpu_count"] == 3
    assert effective["min_gpu_count"] == 2
    assert effective["max_gpu_count"] == 3
    assert effective["elastic"] is True

def test_pack_worker_request_rejects_controller_cap_below_distributed_minimum():
    with pytest.raises(ValueError, match="controller overlap leaves fewer"):
        pack_worker_request(
            {"gpu_count": 3, "min_gpu_count": 3, "max_gpu_count": 3,
             "elastic": False, "distributed": True, "exclusive": True,
             "evaluation_workers": 0, "on_unavailable": "wait"},
            total_gpu_count=4,
            controller_overlap_gpus=2,
        )
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest -q tests/unit/test_remote_pipeline.py tests/unit/test_remote_config.py -k 'pack_worker_request or controller_overlap'`

Expected: FAIL because `pack_worker_request` and `controller_overlap_gpus` do not exist.

- [ ] **Step 3: Implement the pure function and config field**

Implement `pack_worker_request` by copying the mapping, validating integer GPU bounds, returning CPU-only requests unchanged, and for distributed requests setting:

```python
cap = total_gpu_count - controller_overlap_gpus
if cap < int(request["min_gpu_count"]):
    raise ValueError("controller overlap leaves fewer GPUs than the distributed minimum")
effective["max_gpu_count"] = min(int(request["max_gpu_count"]), cap)
effective["gpu_count"] = min(int(request["gpu_count"]), effective["max_gpu_count"])
effective["elastic"] = True
```

Add strict config validation for `0 <= controller_overlap_gpus < 4`, and set the overnight YAML value to `1`.

- [ ] **Step 4: Run focused tests and config validation**

Run: `pytest -q tests/unit/test_remote_pipeline.py tests/unit/test_remote_config.py`

Expected: PASS.

- [ ] **Step 5: Commit the isolated task**

```bash
git add harness4h3/remote/lane_packer.py harness4h3/remote/config.py configs/remote-l40-h3-rsi-overnight.yaml tests/unit/test_remote_pipeline.py tests/unit/test_remote_config.py
git commit -m "feat: add controller-aware GPU lane packing"
```

### Task 2: Make the scheduler honor a live Controller GPU lease

**Files:**
- Modify: `harness4h3/remote/scheduler.py:35-240`
- Modify: `tools/controller-wait-launch.sh:20-80,500-650`
- Test: `tests/unit/test_remote_scheduler.py`
- Test: `tests/unit/test_overnight_controller.py`

**Interfaces:**
- `RemoteResourceScheduler(..., controller_lease_path: Optional[str] = None)` reads the campaign Controller lease.
- `_read_controller_lease()` returns a mapping only for a live lease whose owner PID still exists; `_controller_reserved_gpu_indices()` returns sorted GPU indices.

- [ ] **Step 1: Add failing scheduler tests**

```python
def test_scheduler_excludes_live_controller_lease_gpu(fake_client, tmp_path):
    client = fake_client(gpu_output="0, 1000, 46068\n1, 1000, 46068\n2, 1000, 46068\n3, 1000, 46068\n")
    client.files[str(tmp_path / "controller.json")] = {
        "state": "allocated_for_controller", "owner_pid": 123,
        "created_at": time.time(), "expires_at": time.time() + 60,
        "allocated_gpus": [1],
    }
    client.ps_pids.add(123)
    scheduler = RemoteResourceScheduler(client, controller_lease_path=str(tmp_path / "controller.json"))
    decision = scheduler.acquire({"gpu_count": 2, "min_gpu_count": 2, "max_gpu_count": 2,
        "elastic": False, "distributed": True, "exclusive": True})
    assert decision.allocated_gpus == (0, 2)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest -q tests/unit/test_remote_scheduler.py -k controller_lease`

Expected: FAIL because scheduler currently has no Controller lease input.

- [ ] **Step 3: Implement fail-closed lease parsing**

Add the optional path, parse `created_at`, `expires_at`, `owner_pid`, and `allocated_gpus`, reject malformed values, and add those indices to the `busy` set before computing `free`. Do not reclaim a lease whose owner process is alive. Ignore only an expired lease or a dead exact owner PID.

In the launcher add `CONTROLLER_LEASE_FILE`, `CONTROLLER_HOLD_FILE`, `write_controller_lease`, and `clear_controller_lease`. Publish the selected GPU list before starting vLLM, update the PID after `$!` is available, and clear the exact file in the launcher cleanup path. While the hold file exists, stop/avoid launching the owned child and poll without calling `wait` on a D-state process.

- [ ] **Step 4: Run scheduler and launcher contract tests**

Run: `pytest -q tests/unit/test_remote_scheduler.py tests/unit/test_overnight_controller.py`

Expected: PASS.

- [ ] **Step 5: Commit the isolated task**

```bash
git add harness4h3/remote/scheduler.py tools/controller-wait-launch.sh tests/unit/test_remote_scheduler.py tests/unit/test_overnight_controller.py
git commit -m "feat: reserve controller GPUs before vLLM startup"
```

### Task 3: Integrate three-lane execution into the campaign

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py:1900-2065,4300-4530,4887-5160`
- Modify: `harness4h3/remote/config.py:108-125`
- Test: `tests/integration/test_remote_h3_closed_loop.py`
- Test: `tests/unit/test_remote_pipeline.py`

**Interfaces:**
- `RemoteCampaign._effective_worker_request(plan) -> Tuple[Mapping[str, Any], Mapping[str, Any]]` returns `(planned, effective)` and emits an execution adjustment when they differ.
- `_start_speculative_worker(..., preserve_controller_lane: bool = False)` keeps Controller alive when the evaluation lane is active and acquires only the remaining elastic GPU capacity.

- [ ] **Step 1: Write failing campaign tests**

```python
def test_full_gpu_training_keeps_one_controller_lane(fake_campaign):
    plan = make_distributed_plan(gpu_count=4, min_gpu_count=2, max_gpu_count=4, elastic=True)
    planned, effective = fake_campaign._effective_worker_request(plan)
    assert planned["max_gpu_count"] == 4
    assert effective["max_gpu_count"] == 3
    assert effective["elastic"] is True

def test_speculative_worker_does_not_release_controller_during_evaluation(fake_campaign, monkeypatch):
    releases = []
    monkeypatch.setattr(fake_campaign, "_request_controller_release", lambda reason, wait_s=30.0: releases.append(reason))
    fake_campaign._start_speculative_worker(make_candidate(), make_distributed_plan(gpu_count=4), preserve_controller_lane=True)
    assert releases == []
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest -q tests/integration/test_remote_h3_closed_loop.py tests/unit/test_remote_pipeline.py -k 'controller_lane or effective_worker_request or speculative_worker'`

Expected: FAIL because execution requests are not capped and speculative launch always releases Controller.

- [ ] **Step 3: Implement the campaign integration**

Import `pack_worker_request`. In `_train_one`, compute `effective_request` before `full_gpu_training`, use it for scheduler acquire, worker world size, persisted `last_resource_decision`, and `worker_started`. Only the original plan is sent to plan validation; emit both mappings. With a distributed plan and `controller_overlap_gpus > 0`, skip the “prefetch before full training” join, release the pre-existing Controller under the handoff hold, acquire the effective 3-GPU request, then start `_start_controller_prefetch` after `worker_started`.

In `_start_speculative_worker`, validate and cap the request through the same helper. Add `preserve_controller_lane`; when true, do not call `_request_controller_release`. Pass `preserve_controller_lane=True` from the evaluation callback. Keep current branch state/lineage logic unchanged.

- [ ] **Step 4: Run focused, full, and diff checks**

Run: `pytest -q tests/integration/test_remote_h3_closed_loop.py tests/unit/test_remote_pipeline.py`

Expected: PASS.

Run: `pytest -q`

Expected: PASS, with only the existing CUDA-dependent skips.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 5: Commit the isolated task**

```bash
git add research/experiments/remote_h3_closed_loop.py tests/integration/test_remote_h3_closed_loop.py tests/unit/test_remote_pipeline.py
git commit -m "feat: overlap controller planning with elastic workers"
```

### Task 4: Add lane telemetry and remote handoff

**Files:**
- Modify: `harness4h3/remote/power.py:180-290`
- Modify: `research/experiments/remote_h3_closed_loop.py:580-640,4500-4645`
- Modify: `configs/remote-l40-h3-rsi-overnight.yaml:50-80`
- Create: `research/evidence/four-gpu-lane-packer-2026-09-17.md`
- Test: `tests/unit/test_remote_power.py`

**Interfaces:**
- `RemotePowerSampler.summary()` includes `per_gpu[*].utilization_mean`, `power_mean_w`, `sample_count`, and `lane_unknown` when no exact lane marker exists.
- Campaign events include `lane_allocation` with planned/effective requests and exact lease files.

- [ ] **Step 1: Write failing telemetry assertions**

```python
def test_power_summary_preserves_per_gpu_utilization_and_lane_state():
    summary = summarize_samples([(0, 100.0, 300.0), (1, 0.0, 80.0)])
    assert summary["per_gpu"]["0"]["utilization_mean"] == 100.0
    assert summary["per_gpu"]["1"]["power_mean_w"] == 80.0
```

- [ ] **Step 2: Run focused test and verify failure**

Run: `pytest -q tests/unit/test_remote_power.py -k utilization_mean`

Expected: FAIL because the summary does not expose all lane-level fields.

- [ ] **Step 3: Implement bounded telemetry and evidence recording**

Aggregate existing bounded `gpu_samples`, attach the current lane manifest when readable, and write the exact idle reason (`barrier_wait`, `controller_generation`, `nfs_staging`, `comfyui_release`, or `unowned`). Never replace missing data with 300W or 100%.

- [ ] **Step 4: Run tests and write the evidence template**

Run: `pytest -q tests/unit/test_remote_power.py tests/unit/test_remote_pipeline.py`

Expected: PASS.

Create the evidence file with commands for `ps`, `nvidia-smi`, lease JSON, event ordering, and checkpoint retention. Record measured values only after the remote short window completes.

- [ ] **Step 5: Commit the isolated task**

```bash
git add harness4h3/remote/power.py research/experiments/remote_h3_closed_loop.py configs/remote-l40-h3-rsi-overnight.yaml research/evidence/four-gpu-lane-packer-2026-09-17.md tests/unit/test_remote_power.py
git commit -m "feat: record GPU lane utilization evidence"
```

### Task 5: Sync and validate at a safe remote boundary

**Files:**
- Remote: `/home/intern/huangjiahao/Harness4H3-rsi/harness4h3/remote/*`
- Remote: `/home/intern/huangjiahao/Harness4H3-rsi/research/experiments/remote_h3_closed_loop.py`
- Remote: `/home/intern/huangjiahao/Harness4H3-rsi/tools/controller-wait-launch.sh`
- Remote: `/home/intern/huangjiahao/Harness4H3-rsi/configs/remote-l40-h3-rsi-overnight.yaml`
- Evidence: `/home/intern/huangjiahao/Harness4H3-rsi/var/remote-h3-controller-20260914/controller-events.jsonl`

**Interfaces:**
- Deploy only the files listed above; do not sync broad dirty-worktree changes.
- Do not terminate the current `torchrun` or legacy ComfyUI PID. Apply the new supervisor only after the exact current worker PID exits or at a naturally persisted campaign boundary.

- [ ] **Step 1: Compile and run local full suite**

Run: `python -m py_compile harness4h3/remote/lane_packer.py harness4h3/remote/scheduler.py research/experiments/remote_h3_closed_loop.py` and `pytest -q`.

Expected: compile succeeds and tests pass.

- [ ] **Step 2: Sync exact implementation files and compile remotely**

Run `rsync` for the exact files, then remote `python -m py_compile` and inspect `git diff --` only for those paths.

Expected: no syntax errors and the remote config contains `controller_overlap_gpus: 1`.

- [ ] **Step 3: Check the current exact worker PID before handoff**

Run: `ssh Jiayu-intern 'ps -p 635999 -o pid=,stat=,args=; nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv'`.

Expected: if PID `635999` is still alive, leave it untouched; if it is gone, verify `trainer_result_m0040.json` and the worker lease are released before supervisor handoff.

- [ ] **Step 4: Start the updated supervisor only after the boundary**

Use the existing remote service/handoff entrypoint with the same campaign root and stop file. Do not start a duplicate supervisor while PID `618053` is alive. Verify exact process ownership and watcher logs.

- [ ] **Step 5: Collect bounded remote evidence**

Run a short window covering one plan/evaluation boundary and inspect event order, lease JSON, per-GPU telemetry, result JSON and retained checkpoints. Leave the supervisor running autonomously and document any unavoidable low-utilization interval as an evidence-backed idle reason.
