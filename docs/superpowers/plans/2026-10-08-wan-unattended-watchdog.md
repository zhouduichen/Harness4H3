# Wan Campaign Unattended Watchdog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution in this side conversation; sub-agents are unavailable. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a bounded remote watchdog so the real Wan vLLM campaign survives Codex/SSH disconnects and resumes after a child-process crash without creating a second experiment state system.

**Architecture:** Extend the existing `tools/remote-campaign-service.sh` with a `REMOTE_CAMPAIGN_KIND=wan_vllm` dispatch while preserving its overnight default. The Wan branch launches a focused `tools/wan-vllm-campaign-watchdog.sh`; the watchdog owns only the Wan campaign child, uses the existing `wan_vllm_campaign.py` state/checkpoint/memory, restarts only when no final result exists, and stops on an operator marker or durable final result.

**Tech Stack:** Bash, existing SSH remote host, Python `wan_vllm_campaign.py`, existing campaign artifacts and checkpoint resume, pytest.

## Global Constraints

- Do not modify Wan model architecture, algorithm search space, evaluator, or ExperienceStore schema.
- Preserve the existing overnight service behavior when `REMOTE_CAMPAIGN_KIND` is unset.
- Never kill the existing `w8-dtype-smoke` process or another campaign root.
- Restart only the trusted Wan campaign command and only when `campaign-result.json` is absent.
- Stop retrying on a durable final result, operator stop/pause, or exhausted retry budget.
- Keep `offline_simulation=false` and parent/round lineage enforcement inside `wan_vllm_campaign.py`.

---

### Task 1: Add the bounded Wan watchdog

**Files:**
- Create: `tools/wan-vllm-campaign-watchdog.sh`
- Test: `tests/unit/test_wan_campaign_watchdog.py`

**Interfaces:**
- Consumes environment variables `WAN_CAMPAIGN_PYTHON`, `WAN_CAMPAIGN_SCRIPT`, `WAN_CAMPAIGN_CONFIG`, `WAN_CAMPAIGN_OUTPUT`, `WAN_CAMPAIGN_MAX_ROUNDS`, `WAN_CAMPAIGN_RESTART_LIMIT`, `WAN_CAMPAIGN_BACKOFF_SECONDS`, `WAN_CAMPAIGN_STOP_FILE`, `WAN_CAMPAIGN_LOCK_FILE`, and `WAN_CAMPAIGN_CHILD_PID_FILE`.
- Produces a detached watchdog process that exits `0` after a final result or operator stop, and exits nonzero after the restart budget is exhausted.

- [ ] **Step 1: Write the failing bounded-restart test.**

```python
def test_watchdog_restarts_only_when_no_final_result(tmp_path):
    runner = tmp_path / "runner.py"
    counter = tmp_path / "count"
    result = tmp_path / "campaign-result.json"
    runner.write_text(
        "from pathlib import Path\n"
        "p=Path(%r)\n"
        "n=int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n+1))\n"
        "if n == 0: raise SystemExit(17)\n"
        "Path(%r).write_text('{\\\"status\\\":\\\"success\\\"}')\n"
        % (str(counter), str(result)),
        encoding="utf-8",
    )
    completed = subprocess.run(["bash", "tools/wan-vllm-campaign-watchdog.sh"], env={...}, check=False)
    assert completed.returncode == 0
    assert counter.read_text() == "2"
    assert result.is_file()
```

- [ ] **Step 2: Run the focused test and verify the script is missing.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_wan_campaign_watchdog.py`

Expected: FAIL because `tools/wan-vllm-campaign-watchdog.sh` does not exist.

- [ ] **Step 3: Implement the watchdog loop.**

The script will acquire an exclusive `flock` lock, reject missing trusted paths, remove stale child PID files, and run:

```bash
attempt=0
while :; do
    [ -f "$WAN_CAMPAIGN_STOP_FILE" ] && exit 0
    [ -f "$WAN_CAMPAIGN_OUTPUT/campaign-result.json" ] && exit 0
    "$WAN_CAMPAIGN_PYTHON" "$WAN_CAMPAIGN_SCRIPT" --config "$WAN_CAMPAIGN_CONFIG" --max-rounds "$WAN_CAMPAIGN_MAX_ROUNDS"
    status=$?
    [ -f "$WAN_CAMPAIGN_OUTPUT/campaign-result.json" ] && exit "$status"
    [ -f "$WAN_CAMPAIGN_STOP_FILE" ] && exit 0
    attempt=$((attempt + 1))
    [ "$attempt" -ge "$WAN_CAMPAIGN_RESTART_LIMIT" ] && exit "$status"
    sleep "$((WAN_CAMPAIGN_BACKOFF_SECONDS * attempt))"
done
```

Write the child PID around the Python launch, log each restart, and trap TERM/INT to create the stop marker and terminate only the current child.

- [ ] **Step 4: Add stop-marker and shell-syntax tests.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_wan_campaign_watchdog.py && bash -n tools/wan-vllm-campaign-watchdog.sh`

Expected: PASS and zero shell syntax errors.

- [ ] **Step 5: Commit the watchdog.**

```bash
git add tools/wan-vllm-campaign-watchdog.sh tests/unit/test_wan_campaign_watchdog.py
git commit -m "feat: add bounded Wan campaign watchdog"
```

### Task 2: Extend the existing campaign service with Wan dispatch

**Files:**
- Modify: `tools/remote-campaign-service.sh`
- Modify: `tests/unit/test_remote_campaign_service.py`

**Interfaces:**
- Consumes `REMOTE_CAMPAIGN_KIND=wan_vllm` and the existing service action `start|status|stop|pause|resume`.
- Produces `REMOTE_WAN_CAMPAIGN_*` bindings for the watchdog while leaving the old overnight variables and behavior unchanged.

- [ ] **Step 1: Add failing source-contract tests.**

```python
def test_campaign_service_dispatches_wan_watchdog():
    source = Path("tools/remote-campaign-service.sh").read_text(encoding="utf-8")
    assert "REMOTE_CAMPAIGN_KIND" in source
    assert "wan-vllm-campaign-watchdog.sh" in source
    assert "WAN_CAMPAIGN_RESTART_LIMIT" in source
```

- [ ] **Step 2: Run the test and confirm the dispatch is absent.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_remote_campaign_service.py -k wan`

Expected: FAIL on the missing Wan dispatch strings.

- [ ] **Step 3: Add the additive Wan branch.**

At service startup, preserve `REMOTE_CAMPAIGN_KIND=overnight` as the default. For `wan_vllm`, bind `REPO_ROOT`, `REMOTE_CAMPAIGN_CONFIG`, `REMOTE_CAMPAIGN_OUTPUT`, `REMOTE_CAMPAIGN_PYTHON`, and the retry/stop/lock paths into the watchdog environment. `start` must refuse an already-live matching watchdog, `status` must report the watchdog and child PID, `stop` must create the stop marker and signal only the matching watchdog, and `pause` must use the existing pause marker.

- [ ] **Step 4: Test service syntax and preserve overnight contracts.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_remote_campaign_service.py && bash -n tools/remote-campaign-service.sh`

Expected: all existing service tests plus the new Wan dispatch test pass.

- [ ] **Step 5: Commit the service integration.**

```bash
git add tools/remote-campaign-service.sh tests/unit/test_remote_campaign_service.py
git commit -m "feat: dispatch Wan campaign through remote service"
```

### Task 3: Add the remote Wan service configuration and recovery evidence

**Files:**
- Modify: `configs/student-campaign.wan-vllm-loop.yaml`
- Modify: `docs/student-campaign.md`
- Test: `tests/unit/test_wan_vllm_campaign.py`

**Interfaces:**
- Consumes the existing w8 Wan campaign config.
- Produces explicit watchdog settings: retry limit, backoff, stop marker, lock, child PID, and service log paths under the w8 root.

- [ ] **Step 1: Add configuration assertions.**

```python
def test_wan_loop_config_exposes_unattended_service_defaults(tmp_path):
    config = load_loop_config(Path("configs/student-campaign.wan-vllm-loop.yaml"))
    assert int(config["watchdog_restart_limit"]) >= 1
    assert int(config["watchdog_backoff_seconds"]) >= 0
    assert str(config["watchdog_stop_file"]).startswith(str(config["output_root"]))
```

- [ ] **Step 2: Run the test and confirm the fields are absent.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_wan_vllm_campaign.py -k unattended`

Expected: FAIL before configuration defaults are added.

- [ ] **Step 3: Add trusted defaults and operator documentation.**

Set `watchdog_restart_limit: 3`, `watchdog_backoff_seconds: 30`, and derive `watchdog_stop_file`, `watchdog_lock_file`, `watchdog_child_pid_file`, and `watchdog_log` from `output_root` in `load_loop_config`. Document the exact remote start/status/stop commands and the rule that a final `campaign-result.json` stops retries.

- [ ] **Step 4: Run configuration and campaign tests.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_wan_vllm_campaign.py tests/integration/test_wan_vllm_algorithm_loop.py`

Expected: PASS without starting a GPU job.

- [ ] **Step 5: Commit the configuration and docs.**

```bash
git add configs/student-campaign.wan-vllm-loop.yaml docs/student-campaign.md tests/unit/test_wan_vllm_campaign.py
git commit -m "docs: configure unattended Wan campaign recovery"
```

### Task 4: Deploy and verify detached long-running execution

**Files:**
- Remote: `/home/intern/huangjiahao/Harness4H3-long-video-30s/tools/wan-vllm-campaign-watchdog.sh`
- Remote: `/home/intern/huangjiahao/Harness4H3-long-video-30s/tools/remote-campaign-service.sh`
- Remote: `/home/intern/huangjiahao/Harness4H3-long-video-30s/configs/student-campaign.wan-vllm-loop.yaml`

- [ ] **Step 1: Run the full local watchdog/service checks.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_wan_campaign_watchdog.py tests/unit/test_remote_campaign_service.py tests/unit/test_wan_vllm_campaign.py`

Expected: all tests pass; unrelated INT8 quantization failure remains outside this feature.

- [ ] **Step 2: Sync only the watchdog, service, config, and current Wan supervisor.**

Use `rsync -avR` so directory paths are preserved. Verify remote SHA-256 values before launch.

- [ ] **Step 3: Start the Wan service without stopping the old smoke.**

```bash
REMOTE_CAMPAIGN_KIND=wan_vllm \
REMOTE_CAMPAIGN_REPO_ROOT=/home/intern/huangjiahao/Harness4H3-long-video-30s \
REMOTE_CAMPAIGN_CONFIG=/home/intern/huangjiahao/Harness4H3-long-video-30s/configs/student-campaign.wan-vllm-loop.yaml \
REMOTE_CAMPAIGN_OUTPUT=/data/models/harness4h3-v27-code-long-30s/wan-vllm-loop-w8 \
bash /home/intern/huangjiahao/Harness4H3-long-video-30s/tools/remote-campaign-service.sh start
```

- [ ] **Step 4: Verify detached service and child ownership.**

Check `status`, `ps -o pid,ppid,sid,stat,args`, watchdog log, `campaign-state.json`, round artifacts, and that the process survives SSH disconnect. Do not claim quality improvement until the existing evaluator emits evidence.

- [ ] **Step 5: Commit final operator documentation/status.**

```bash
git add docs/student-campaign.md docs/superpowers/specs/2026-10-08-wan-unattended-watchdog-design.md
git commit -m "docs: record unattended Wan campaign operation"
```
