# Remote Validation Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, bounded remote validation window that records real four-GPU evidence and exits without Codex long monitoring.

**Architecture:** Extend the existing campaign supervisor with an optional iteration bound. Add a remote shell wrapper that owns only its sampler and the configured campaign service, and a Python summarizer that emits bounded JSON/Markdown evidence from telemetry and event-log offsets.

**Tech Stack:** Bash, Python 3 standard library, `nvidia-smi`, existing campaign service/supervisor, pytest.

## Global Constraints

- The validation window requires explicit `--start`; default invocation is read-only.
- An operator pause or any pre-existing compute process blocks launch.
- Only the configured campaign and its Controller launcher may be stopped; foreign processes are never killed.
- A timeout uses the existing graceful stop marker; hard process termination is not part of the window.
- Missing telemetry is reported as unverified; it cannot be converted into a 300 W/100% claim.
- Model/checkpoint bytes are never copied into the evidence directory.

### Task 1: Pass an optional iteration bound through the supervisor

**Files:**
- Modify: `tools/remote-campaign-service.sh`
- Modify: `tools/remote-campaign-supervisor.sh`
- Test: `tests/unit/test_remote_campaign_supervisor.py`

- [x] Add source assertions for the optional `REMOTE_CAMPAIGN_MAX_ITERATIONS` environment variable and runner flag.
- [x] Export the variable from service `start`; preserve the existing default when unset.
- [x] Pass `--max-iterations` in the supervisor only when the value is a positive integer.
- [x] Run focused supervisor/service tests.

### Task 2: Add the remote validation window and telemetry sampler

**Files:**
- Create: `tools/remote-validation-window.sh`
- Test: `tests/unit/test_remote_validation_window.py`

- [x] Add read-only default and explicit `--start` behavior tests.
- [x] Implement operator-pause, active-campaign, and foreign-compute-process refusal checks.
- [x] Implement bounded sampler lifecycle, status polling, graceful timeout pause, and trap cleanup.
- [x] Record pre/post snapshots, status TSV, telemetry CSV, event-log byte offset, and a manifest without model bytes.
- [x] Run focused script contract tests and `bash -n`.

### Task 3: Summarize the window into bounded evidence

**Files:**
- Create: `tools/summarize_remote_validation.py`
- Test: `tests/unit/test_remote_validation_summary.py`
- Modify: `docs/validation-plan.md`

- [x] Add parser tests for per-GPU aggregation, event filtering, missing telemetry, and explicit unverified status.
- [x] Implement standard-library-only JSON/Markdown output with bounded event names and references.
- [x] Include lane/power/utilization/memory summaries plus prefetch, policy, ComfyUI, and terminal-state event names.
- [x] Document the invocation and evidence limits.

### Task 4: Regression and paused remote synchronization

**Files:**
- Modify: `docs/superpowers/specs/2026-09-18-a-evolve-round-policy-design.md`
- Modify: `docs/superpowers/plans/2026-09-19-remote-validation-window.md`

- [ ] Run focused tests, static checks, and the full local regression.
- [ ] Sync allowlisted runtime files to the remote staging checkout without starting services.
- [ ] Verify campaign paused, campaign PID absent, idle watcher absent, and operator pause marker present.
