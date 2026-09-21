# ComfyUI GPU Lease Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release GPU0's resident ComfyUI model at safe campaign boundaries so the remote scheduler can reuse the card, while preserving benchmark isolation and fail-closed behavior.

**Architecture:** Add a small `ComfyUILeaseManager` around the existing SSH transport and scheduler. The campaign reserves GPU0 before every benchmark, and after benchmark evidence is written it checks the ComfyUI queue, calls `/free`, verifies the scheduler memory waterline, and only then removes GPU0 from the scheduler reservation. Configuration selects `idle_release`, `warm_cache`, or `cold_cache`; all transitions are observable and do not touch the dynamically placed LLM process.

**Tech Stack:** Python 3, dataclasses, existing `SSHClient`, `RemoteResourceScheduler`, ComfyUI HTTP API, PyYAML, pytest, JSONL event stores.

## Global Constraints

- Never call `/free` while `queue_running` or `queue_pending` is non-empty.
- Queue/API timeout, malformed response, `/free` error, or insufficient post-release memory evidence keeps GPU0 reserved.
- Remove GPU0 from scheduler reservations only after the `/free` response and the post-release memory check succeed.
- Training allocation remains task-boundary-only; no GPU is hot-added to a running worker.
- The LLM service is unaffected and is not moved, reloaded, or co-located by this feature; its launcher dynamically selects 1/2/4 cards.
- Do not terminate or alter unrelated remote processes.
- Preserve the existing dirty worktree; do not commit unrelated user changes.

---

## File map

- Create: `harness4h3/remote/comfyui_lease.py` — queue inspection, `/free` request, post-release verification, and immutable result type.
- Modify: `harness4h3/remote/scheduler.py` — public reservation and snapshot/waterline primitives used by the lease manager.
- Modify: `harness4h3/remote/config.py` — validated `benchmark.comfyui_cache_policy` field.
- Modify: `research/experiments/remote_h3_closed_loop.py` — integrate lease boundaries, events, benchmark provenance, and training handoff.
- Modify: `configs/remote-l40-h3.yaml`, `configs/remote-l40-h3-rsi-controller.yaml`, `configs/remote-l40-h3-rsi-overnight.yaml` — make the policy explicit.
- Modify: `tests/unit/test_remote_scheduler.py` — test public reservation and memory-waterline behavior.
- Create: `tests/unit/test_comfyui_lease.py` — test queue/API safety and idempotence with an SSH double.
- Modify: `tests/unit/test_remote_config.py` — test policy defaults and invalid values.
- Modify: `tests/integration/test_remote_h3_closed_loop.py` — test benchmark reservation and release event behavior without real SSH.
- Modify: `docs/quickstart.md` — document warm/cold/idle cache semantics and the safe release event.

## Interfaces

The implementation exposes these exact interfaces:

```python
class ComfyUILeaseManager:
    def __init__(self, client: SSHClient, scheduler: RemoteResourceScheduler, port: int, gpu_index: int = 0, api_timeout_s: float = 10.0): ...
    def prepare_for_benchmark(self) -> Mapping[str, Any]: ...
    def release_if_idle(self) -> ComfyUILeaseResult: ...

@dataclass(frozen=True)
class ComfyUILeaseResult:
    state: str
    success: bool
    reason: str
    queue: Mapping[str, Any]
    response: Mapping[str, Any]
    memory_snapshot: Mapping[int, tuple[int, Optional[int]]]
    elapsed_s: float
    gpu_index: int

class RemoteResourceScheduler:
    def reserve_gpu(self, index: int) -> None: ...
    def release_gpu(self, index: int) -> None: ...
    def snapshot(self) -> Tuple[Mapping[int, Tuple[int, Optional[int]]], str]: ...
    def meets_memory_waterline(self, index: int, snapshot: Optional[Mapping[int, Tuple[int, Optional[int]]]] = None) -> bool: ...
```

`ComfyUILeaseResult.state` is one of `reserved_for_benchmark`, `released_for_other_work`, or `release_failed`. `release_if_idle()` never raises expected transport/API failures; it returns `release_failed` and keeps the reservation. Invalid constructor arguments still raise `ValueError`.

### Task 1: Add scheduler primitives and cache-policy validation

**Files:**
- Modify: `harness4h3/remote/scheduler.py:36-176`
- Modify: `harness4h3/remote/config.py:65-219`
- Modify: `tests/unit/test_remote_scheduler.py`
- Modify: `tests/unit/test_remote_config.py`

**Interfaces:**
- Consumes: existing `_snapshot()`, `min_free_memory_mb`, and `RemoteCampaignConfig` parsing.
- Produces: `reserve_gpu`, `release_gpu`, `snapshot`, `meets_memory_waterline`, and `RemoteCampaignConfig.comfyui_cache_policy` for later tasks.

- [ ] **Step 1: Write failing scheduler tests**

Append these tests to `tests/unit/test_remote_scheduler.py`:

```python
def test_scheduler_can_reserve_and_release_one_gpu():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n"))
    scheduler.reserve_gpu(0)
    assert 0 in scheduler.reserved_gpu_indices
    scheduler.release_gpu(0)
    assert 0 not in scheduler.reserved_gpu_indices


def test_scheduler_waterline_uses_total_and_used_memory():
    scheduler = RemoteResourceScheduler(
        FakeSSH("0, 20000, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n"),
        min_free_memory_mb=26000,
    )
    memory, _ = scheduler.snapshot()
    assert scheduler.meets_memory_waterline(0, memory) is True
    assert scheduler.meets_memory_waterline(9, memory) is False


def test_scheduler_reservation_rejects_out_of_range_gpu():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n"))
    with pytest.raises(ValueError, match="within the GPU range"):
        scheduler.reserve_gpu(4)
```

Run: `pytest -q tests/unit/test_remote_scheduler.py::test_scheduler_can_reserve_and_release_one_gpu tests/unit/test_remote_scheduler.py::test_scheduler_waterline_uses_total_and_used_memory tests/unit/test_remote_scheduler.py::test_scheduler_reservation_rejects_out_of_range_gpu`

Expected: FAIL because the public methods do not exist.

- [ ] **Step 2: Implement scheduler primitives**

Add the following methods to `RemoteResourceScheduler` without changing `acquire()` semantics:

```python
    def reserve_gpu(self, index: int) -> None:
        index = int(index)
        if index < 0 or index >= self.gpu_count:
            raise ValueError("reserved GPU index must be within the GPU range")
        self.reserved_gpu_indices = tuple(sorted(set(self.reserved_gpu_indices) | {index}))

    def release_gpu(self, index: int) -> None:
        index = int(index)
        if index < 0 or index >= self.gpu_count:
            raise ValueError("reserved GPU index must be within the GPU range")
        self.reserved_gpu_indices = tuple(item for item in self.reserved_gpu_indices if item != index)

    def snapshot(self) -> Tuple[Mapping[int, Tuple[int, Optional[int]]], str]:
        return self._snapshot()

    def meets_memory_waterline(
        self,
        index: int,
        snapshot: Optional[Mapping[int, Tuple[int, Optional[int]]]] = None,
    ) -> bool:
        values = self.snapshot()[0] if snapshot is None else snapshot
        item = values.get(int(index))
        if item is None or item[1] is None:
            return False
        used_mb, total_mb = item
        return int(total_mb) - int(used_mb) >= self.min_free_memory_mb
```

Run the three tests again. Expected: PASS.

- [ ] **Step 3: Write failing configuration tests**

Append to `tests/unit/test_remote_config.py`:

```python
def test_remote_config_defaults_to_idle_comfyui_cache_release():
    config = load_remote_campaign_config(Path("configs/remote-l40-h3.yaml"))
    assert config.comfyui_cache_policy == "idle_release"


def test_remote_config_rejects_unknown_comfyui_cache_policy(tmp_path):
    source = Path("configs/remote-l40-h3.yaml").read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(source.replace("  reset_backend_before_run: true\n", "  reset_backend_before_run: true\n  comfyui_cache_policy: invalid\n"))
    with pytest.raises(Exception, match="comfyui_cache_policy"):
        load_remote_campaign_config(path)
```

Add `import pytest` to that test module. Run: `pytest -q tests/unit/test_remote_config.py::test_remote_config_defaults_to_idle_comfyui_cache_release tests/unit/test_remote_config.py::test_remote_config_rejects_unknown_comfyui_cache_policy`. Expected: FAIL because the field and validation do not exist.

- [ ] **Step 4: Implement policy parsing**

Add a frozen dataclass field with default:

```python
    comfyui_cache_policy: str = "idle_release"
```

Parse and validate before constructing `RemoteCampaignConfig`:

```python
    comfyui_cache_policy = str(benchmark_raw.get("comfyui_cache_policy", "idle_release")).strip()
    if comfyui_cache_policy not in {"idle_release", "warm_cache", "cold_cache"}:
        raise RemoteConfigError(
            "benchmark.comfyui_cache_policy must be idle_release, warm_cache, or cold_cache"
        )
```

Pass `comfyui_cache_policy=comfyui_cache_policy` to the return value. Run both configuration tests and then `pytest -q tests/unit/test_remote_scheduler.py tests/unit/test_remote_config.py`; expected: PASS.

### Task 2: Implement the fail-closed ComfyUI lease manager

**Files:**
- Create: `harness4h3/remote/comfyui_lease.py`
- Create: `tests/unit/test_comfyui_lease.py`

**Interfaces:**
- Consumes: `SSHClient.run`, `RemoteResourceScheduler.reserve_gpu/release_gpu/snapshot/meets_memory_waterline`.
- Produces: `ComfyUILeaseManager.prepare_for_benchmark()` and `release_if_idle()` for campaign integration.

- [ ] **Step 1: Write failing lease tests**

Create `tests/unit/test_comfyui_lease.py` with this complete test double and tests:

```python
from types import SimpleNamespace

from harness4h3.remote.comfyui_lease import ComfyUILeaseManager
from harness4h3.remote.scheduler import RemoteResourceScheduler


class LeaseSSH:
    def __init__(self, queue, free_response='{"ok": true}', memory="0, 0, 46080\\n1, 0, 46080\\n2, 0, 46080\\n3, 0, 46080\\n"):
        self.queue = queue
        self.free_response = free_response
        self.memory = memory
        self.commands = []

    def run(self, command, **kwargs):
        command = tuple(command)
        self.commands.append(command)
        if "/queue" in command:
            return SimpleNamespace(stdout=self.queue, stderr="", returncode=0)
        if "/free" in command:
            return SimpleNamespace(stdout=self.free_response, stderr="", returncode=0)
        if "query-compute-apps" in command:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        return SimpleNamespace(stdout=self.memory, stderr="", returncode=0)


def manager(remote):
    scheduler = RemoteResourceScheduler(remote, min_free_memory_mb=26000, reserved_gpu_indices=(0,))
    return ComfyUILeaseManager(remote, scheduler, port=8188), scheduler


def test_release_if_idle_calls_free_and_releases_gpu0():
    remote = LeaseSSH('{"queue_running": [], "queue_pending": []}')
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.success is True
    assert result.state == "released_for_other_work"
    assert 0 not in scheduler.reserved_gpu_indices
    assert any("/free" in command for command in remote.commands)


def test_release_refuses_active_queue_without_calling_free():
    remote = LeaseSSH('{"queue_running": [{"prompt": "busy"}], "queue_pending": []}')
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.success is False
    assert result.state == "reserved_for_benchmark"
    assert "queue_active" in result.reason
    assert 0 in scheduler.reserved_gpu_indices
    assert not any("/free" in command for command in remote.commands)


def test_release_api_failure_is_fail_closed():
    remote = LeaseSSH('{"queue_running": [], "queue_pending": []}', free_response="not-json")
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.success is False
    assert result.state == "release_failed"
    assert 0 in scheduler.reserved_gpu_indices


def test_prepare_is_idempotent_and_reserves_gpu0():
    remote = LeaseSSH('{"queue_running": [], "queue_pending": []}')
    lease, scheduler = manager(remote)
    scheduler.release_gpu(0)
    payload = lease.prepare_for_benchmark()
    assert payload["state"] == "reserved_for_benchmark"
    assert 0 in scheduler.reserved_gpu_indices
```

Run: `pytest -q tests/unit/test_comfyui_lease.py`. Expected: FAIL because the module does not exist.

- [ ] **Step 2: Implement queue/API helpers and result type**

Create `harness4h3/remote/comfyui_lease.py` with this implementation shape:

```python
from dataclasses import dataclass
import json
import time
from typing import Any, Mapping, Optional, Tuple

from .scheduler import RemoteResourceScheduler
from .ssh import SSHClient


@dataclass(frozen=True)
class ComfyUILeaseResult:
    state: str
    success: bool
    reason: str
    queue: Mapping[str, Any]
    response: Mapping[str, Any]
    memory_snapshot: Mapping[int, Tuple[int, Optional[int]]]
    elapsed_s: float
    gpu_index: int


class ComfyUILeaseManager:
    def __init__(self, client, scheduler, port, gpu_index=0, api_timeout_s=10.0):
        self.client = client
        self.scheduler = scheduler
        self.port = int(port)
        self.gpu_index = int(gpu_index)
        self.api_timeout_s = float(api_timeout_s)
        if self.gpu_index < 0 or self.gpu_index >= self.scheduler.gpu_count:
            raise ValueError("gpu_index must be within the scheduler GPU range")

    @property
    def _base_url(self):
        return "http://127.0.0.1:%d" % self.port

    def _request(self, path, payload=None):
        command = ["curl", "-fsS", "--max-time", str(int(self.api_timeout_s))]
        if payload is not None:
            command.extend(["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(payload, sort_keys=True)])
        command.append(self._base_url + path)
        result = self.client.run(tuple(command), check=False, timeout_s=self.api_timeout_s + 2.0)
        if int(getattr(result, "returncode", 1)) != 0:
            raise RuntimeError(str(getattr(result, "stderr", "")).strip() or "ComfyUI request failed")
        try:
            value = json.loads(str(getattr(result, "stdout", "")))
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid JSON from ComfyUI %s: %s" % (path, exc))
        if not isinstance(value, Mapping):
            raise RuntimeError("ComfyUI %s response must be an object" % path)
        return dict(value)

    def prepare_for_benchmark(self):
        self.scheduler.reserve_gpu(self.gpu_index)
        return {"state": "reserved_for_benchmark", "gpu_index": self.gpu_index, "reserved_gpu_indices": list(self.scheduler.reserved_gpu_indices)}

    def release_if_idle(self):
        started = time.monotonic()
        self.scheduler.reserve_gpu(self.gpu_index)
        try:
            queue = self._request("/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                return ComfyUILeaseResult("reserved_for_benchmark", False, "queue_active", queue, {}, {}, time.monotonic() - started, self.gpu_index)
            response = self._request("/free", {"unload_models": True, "free_memory": True})
            snapshot, _ = self.scheduler.snapshot()
            if not self.scheduler.meets_memory_waterline(self.gpu_index, snapshot):
                return ComfyUILeaseResult("release_failed", False, "post_release_memory_below_waterline", queue, response, snapshot, time.monotonic() - started, self.gpu_index)
            self.scheduler.release_gpu(self.gpu_index)
            return ComfyUILeaseResult("released_for_other_work", True, "released_and_verified", queue, response, snapshot, time.monotonic() - started, self.gpu_index)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return ComfyUILeaseResult("release_failed", False, str(exc), locals().get("queue", {}), locals().get("response", {}), {}, time.monotonic() - started, self.gpu_index)
```

The implementation must never remove the reservation in an exception path. Run the four lease tests; expected: PASS.

- [ ] **Step 3: Add malformed queue and low-memory tests**

Add these tests to the same file:

```python
def test_release_malformed_queue_is_fail_closed():
    remote = LeaseSSH("not-json")
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.state == "release_failed"
    assert 0 in scheduler.reserved_gpu_indices


def test_release_keeps_gpu_reserved_when_memory_waterline_fails():
    remote = LeaseSSH('{"queue_running": [], "queue_pending": []}', memory="0, 30000, 46080\\n1, 0, 46080\\n2, 0, 46080\\n3, 0, 46080\\n")
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.reason == "post_release_memory_below_waterline"
    assert 0 in scheduler.reserved_gpu_indices
```

Run: `pytest -q tests/unit/test_comfyui_lease.py`; expected: PASS.

### Task 3: Integrate lease transitions into the remote campaign

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py:200-220, 830-960, 1420-1450, 1610-1625, 1970-1995`
- Modify: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- Consumes: `ComfyUILeaseManager`, `RemoteCampaignConfig.comfyui_cache_policy`.
- Produces: lease events, cache policy in benchmark recipes, GPU0 release before training, and GPU0 reservation before preflight/benchmark.

- [ ] **Step 1: Write failing campaign integration assertions**

Update `_FakeSSH.run()` in `tests/integration/test_remote_h3_closed_loop.py` to return an idle queue JSON for commands containing `/queue`, an acknowledged JSON for `/free`, and a four-GPU memory CSV for scheduler queries. Add this test:

```python
def test_campaign_records_comfyui_lease_release_after_evaluation(tmp_path):
    campaign = build_campaign(tmp_path, {}, {"M0000": 0.86}, {"M0000": {"latency_s": 30, "peak_memory_gb": 42, "energy_j": 120}})
    result = campaign.run(resume=False, max_experiments=1)
    events = [json.loads(line) for line in (tmp_path / "controller-events.jsonl").read_text().splitlines()]
    names = [event["event"] for event in events]
    assert "comfyui_lease_reserved" in names
    assert "comfyui_cache_released" in names
    assert campaign.scheduler.reserved_gpu_indices == ()
    assert result.report["benchmark_cache_policy"] == "idle_release"
```

Add `import json` to the test module. Run the test. Expected: FAIL because the campaign has no lease manager or release report field.

- [ ] **Step 2: Construct the lease manager and centralize reservation**

Import `ComfyUILeaseManager`, construct it immediately after `self.scheduler`:

```python
        self.comfyui_lease = ComfyUILeaseManager(
            self.ssh,
            self.scheduler,
            port=self.config.remote.comfyui_port,
            gpu_index=0,
        )
```

Replace `_refresh_comfyui_reservation()` with two campaign helpers:

```python
    def _prepare_comfyui_lease(self) -> Mapping[str, Any]:
        payload = self.comfyui_lease.prepare_for_benchmark()
        self.events.append("comfyui_lease_reserved", {"cache_policy": self.config.comfyui_cache_policy, **payload})
        return payload

    def _release_comfyui_if_idle(self, reason: str) -> ComfyUILeaseResult:
        self.events.append(
            "comfyui_cache_release_requested",
            {"reason": reason, "cache_policy": self.config.comfyui_cache_policy, "gpu_index": 0},
        )
        result = self.comfyui_lease.release_if_idle()
        payload = {
            "reason": reason,
            "cache_policy": self.config.comfyui_cache_policy,
            "state": result.state,
            "success": result.success,
            "release_reason": result.reason,
            "gpu_index": result.gpu_index,
            "reserved_gpu_indices": list(self.scheduler.reserved_gpu_indices),
            "elapsed_s": result.elapsed_s,
            "queue": {"running": len(result.queue.get("queue_running", [])), "pending": len(result.queue.get("queue_pending", []))},
            "memory_snapshot": {str(key): list(value) for key, value in result.memory_snapshot.items()},
        }
        self.events.append("comfyui_cache_released" if result.success else "comfyui_cache_release_failed", payload)
        return result
```

Call `_prepare_comfyui_lease()` at the start of `_remote_comfyui_preflight()` and at the start of `_evaluate()` so direct `run()` calls cannot bypass reservation. Do not inspect queue state to remove reservations anymore.

- [ ] **Step 3: Add the policy to benchmark provenance and release at the campaign boundary**

Include these fields in both benchmark recipe dictionaries in `_evaluate()`:

```python
"comfyui_cache_policy": self.config.comfyui_cache_policy,
"comfyui_lease_state": "reserved_for_benchmark",
```

After the evaluation loop has persisted the candidate `EvaluationRecord`, decision, experience, experiment record, and campaign state, invoke release exactly once for the completed phase:

```python
        if evaluated_count:
            if self.config.comfyui_cache_policy in {"idle_release", "cold_cache"}:
                lease_result = self._release_comfyui_if_idle("evaluation_phase_completed")
                report_lease = {
                    "state": lease_result.state,
                    "success": lease_result.success,
                    "reason": lease_result.reason,
                    "elapsed_s": lease_result.elapsed_s,
                }
            else:
                report_lease = {"state": "reserved_for_benchmark", "success": False, "reason": "warm_cache_policy", "elapsed_s": 0.0}
        else:
            report_lease = {"state": "reserved_for_benchmark", "success": False, "reason": "no_evaluation_phase", "elapsed_s": 0.0}
```

Store `report_lease` in the returned report as `benchmark_cache_lease`. Add `benchmark_cache_policy` to the report. Update the saved evaluation summary's `benchmark_recipe` with `comfyui_lease_release` before the final `_save_evaluations()` call. The release result is diagnostic metadata only and must not alter evaluator hard gates or acceptance decisions. The release helper must emit `comfyui_cache_release_requested` immediately before calling the manager, followed by exactly one of `comfyui_cache_released` or `comfyui_cache_release_failed`.

For `cold_cache`, call `_release_comfyui_if_idle("cold_cache_before_benchmark")` immediately after a previous phase is complete and before the next benchmark's `_prepare_comfyui_lease()`. Keep the scheduler reservation while the current benchmark is running.

- [ ] **Step 4: Release idle ComfyUI before a training allocation**

At the beginning of `_train_one()`, before `scheduler.acquire()` and only for `idle_release`/`cold_cache`, call:

```python
        if self.config.comfyui_cache_policy in {"idle_release", "cold_cache"}:
            self._release_comfyui_if_idle("training_boundary")
```

Do not proceed based on the return value alone; `scheduler.acquire()` remains authoritative and will wait if GPU0 is still busy. Remove the old queue-dependent assignment that set `reserved_gpu_indices` to `()` merely because the queue was empty.

- [ ] **Step 5: Run integration tests and inspect event ordering**

Run:

```bash
pytest -q tests/integration/test_remote_h3_closed_loop.py
```

Expected: PASS. Then inspect the generated JSONL in the test temporary directory and verify the order is `evaluation_completed`/decision persistence, followed by `comfyui_cache_released` or `comfyui_cache_release_failed`; no release event occurs while benchmark execution is active.

### Task 4: Make remote configurations and operator documentation explicit

**Files:**
- Modify: `configs/remote-l40-h3.yaml`
- Modify: `configs/remote-l40-h3-rsi-controller.yaml`
- Modify: `configs/remote-l40-h3-rsi-overnight.yaml`
- Modify: `docs/quickstart.md`

**Interfaces:**
- Consumes: validated `benchmark.comfyui_cache_policy`.
- Produces: explicit remote runtime behavior and operator-facing instructions.

- [ ] **Step 1: Add the explicit policy to all remote campaign configs**

Under each `benchmark:` section, add:

```yaml
  comfyui_cache_policy: idle_release
```

Keep the overnight run on `idle_release` so GPU0 is returned between benchmark and training phases. Do not change the controller endpoint, dynamic worker GPU bounds, or benchmark split selection.

- [ ] **Step 2: Document operational semantics**

In `docs/quickstart.md`, add a short remote subsection stating:

```text
benchmark.comfyui_cache_policy controls only the campaign-level ComfyUI model cache:
idle_release unloads ComfyUI after a completed evaluation phase;
warm_cache keeps GPU0 resident for warm-cache latency comparisons;
cold_cache releases at each controlled phase boundary.
The scheduler removes GPU0 from its reservation only after /free and a fresh nvidia-smi memory-waterline check succeed. A failed release is fail-closed and is visible as comfyui_cache_release_failed.
```

- [ ] **Step 3: Validate configs and docs**

Run:

```bash
pytest -q tests/unit/test_remote_config.py
python -m py_compile harness4h3/remote/comfyui_lease.py harness4h3/remote/scheduler.py research/experiments/remote_h3_closed_loop.py
git diff --check
```

Expected: PASS with no whitespace errors.

### Task 5: Full local verification and remote deployment

**Files:**
- Verify all files from Tasks 1–4; no new source file is expected.

**Interfaces:**
- Consumes: passing unit/integration tests and the existing remote staging deployment mechanism.
- Produces: deployed remote campaign code and real evidence of the lease transition.

- [ ] **Step 1: Run focused regression tests**

Run:

```bash
pytest -q tests/unit/test_remote_scheduler.py tests/unit/test_comfyui_lease.py tests/unit/test_remote_config.py tests/integration/test_remote_h3_closed_loop.py
```

Expected: PASS.

- [ ] **Step 2: Run the full local suite**

Run: `pytest -q`

Expected: all existing tests pass; CUDA-only tests may remain skipped for the local laptop environment.

- [ ] **Step 3: Deploy only the changed runtime/config files to isolated remote staging**

Use the existing `Jiayu-intern` SSH deployment convention and copy these exact paths to `/home/intern/huangjiahao/Harness4H3-rsi`:

```text
harness4h3/remote/comfyui_lease.py
harness4h3/remote/scheduler.py
harness4h3/remote/config.py
research/experiments/remote_h3_closed_loop.py
configs/remote-l40-h3-rsi-overnight.yaml
```

Do not overwrite `/home/intern/huangjiahao/Harness4H3` or stop the existing overnight process. Run remote syntax validation with the configured Python interpreter before restarting anything.

- [ ] **Step 4: Observe the current run at a safe boundary**

The current M0003 ComfyUI evaluation must be allowed to finish. After its queue is empty, run a read-only check of `/queue`, then inspect the campaign event tail for `comfyui_cache_released` or `comfyui_cache_release_failed`. Confirm with `nvidia-smi` that GPU0 memory drops only after the release event and that GPUs2/3 training allocation is unchanged.

- [ ] **Step 5: Verify next benchmark re-reservation**

On the next campaign iteration, confirm the event sequence contains `comfyui_lease_reserved` before `benchmark_preflight` and that ComfyUI reloads the model as part of normal workflow submission. Confirm the vLLM endpoint remains healthy and no LLM process is moved.

Because the worktree already contains unrelated user changes, leave commits to the user; report the exact changed-file list, test results, remote event names, and any release failure reason instead of creating a broad commit.
