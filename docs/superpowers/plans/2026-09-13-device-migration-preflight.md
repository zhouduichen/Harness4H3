# Device Migration Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Turn the approved device-migration design into a small, repeatable, evidence-producing readiness workflow for the four-L40 target without implementing H3 training or changing Harness4H3-v1.0 core behavior.

**Architecture:** Extend the existing read-only device preflight in place with strict verification-state and path-kind checks plus an atomic `--result` output. Keep the L40 profile and one-step FSDP recipe declarative and disabled. Make `docs/device-porting.md` the operator runbook and use existing `harness4h3 benchmark` as the fixed baseline command.

**Tech Stack:** Python 3.9+, argparse, pathlib, JSON/YAML, subprocess, pytest, existing Harness4H3 CLI.

## Global Constraints

- Do not modify Harness4H3-v1.0 core acceptance rules or orchestration semantics.
- Do not implement or enable `recovery_finetune`, pruning, or distillation.
- Do not start a real training or benchmark process from this workstation.
- Do not change the user’s pre-existing edits in `research/experiments/m6_campaign.py` or `research/experiments/m6_runtime_recipe.py`.
- M6 remains an optional runtime study and is not part of the migration critical path.
- A pending or skipped required check must result in `blocked` and exit code 2.
- Existing non-GPU device profiles must retain their current preflight behavior.

---

### Task 1: Persist readiness evidence and enforce verification state

**Files:**
- Modify: `tools/device_preflight.py`
- Test: `tests/unit/test_device_preflight.py`

**Interfaces:**
- Preserve `preflight(profile_path, operator, check_services=True, timeout_s=3.0, check_hardware=True) -> dict`.
- Add `write_result(path: Path, result: Mapping[str, Any]) -> Path`, which writes one JSON object plus a trailing newline using a temporary sibling and atomic replacement.
- Add CLI option `--result PATH`; the CLI still prints the normal result and returns 0 for `ready`, otherwise 2.

- [x] **Step 1: Write failing tests for pending verification and result persistence**

Add tests that construct a distributed profile with `verification.status: pending`, assert `profile.verification` is failed and overall status is `blocked`, then set the status to `verified` and assert the profile can pass with mocked hardware. Add a test that calls `write_result`, reloads the JSON, and asserts the exact result is preserved.

- [x] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_device_preflight.py -k 'verification or write_result'
```

Expected: failure because the verification check and `write_result` do not yet exist.

- [x] **Step 3: Implement the smallest compatible preflight changes**

In `tools/device_preflight.py`, add:

```python
def write_result(path: Path, result: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
```

After schema validation, if the profile contains `verification.status`, add a `profile.verification` check that passes only for `verified` or `measured` and otherwise reports the observed status as blocked. Add `--result` to the parser, call `write_result` after `preflight`, and include the resolved result path in the emitted JSON payload under `result` without changing the returned preflight decision.

- [x] **Step 4: Run the focused tests and the full offline suite**

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_device_preflight.py
.venv/bin/python -m pytest -q
```

Expected: both commands pass; existing profiles without `verification` remain compatible.

- [x] **Step 5: Commit the preflight evidence change**

```bash
git add tools/device_preflight.py tests/unit/test_device_preflight.py
git commit -m "feat: persist device preflight evidence"
```

### Task 2: Make declared path expectations executable

**Files:**
- Modify: `tools/device_preflight.py`
- Modify: `configs/devices/l40x4-server.yaml`
- Test: `tests/unit/test_device_preflight.py`
- Test: `tests/unit/test_l40x4_configs.py`

**Interfaces:**
- Preserve existing path check names such as `path.repository` and `path.trainer`.
- Support optional path entry key `kind` with values `file` or `directory`; an entry without `kind` keeps existence-only behavior.

- [x] **Step 1: Write failing tests for file/directory mismatches**

Add one test with a required file entry pointing to a directory and assert the named path check fails with a detail containing `expected file`. Add a second test with a required directory entry pointing to a file and assert the detail contains `expected directory`. Extend the L40 config test to assert the trainer and smoke recipe are files while repository and deployment paths are directories.

- [x] **Step 2: Run the focused tests to verify they fail**

```bash
.venv/bin/python -m pytest -q tests/unit/test_device_preflight.py tests/unit/test_l40x4_configs.py
```

Expected: the new assertions fail because path checks currently test only existence.

- [x] **Step 3: Implement kind-aware path checking and declare L40 kinds**

Update the path loop in `tools/device_preflight.py` so it computes `exists`, then validates `entry.get("kind")` when present:

```python
kind = str(entry.get("kind", "")).strip()
kind_ok = kind in {"", "file", "directory"} and (
    not exists
    or kind == ""
    or (kind == "file" and resolved.is_file())
    or (kind == "directory" and resolved.is_dir())
)
passed = exists and kind_ok
detail = str(resolved) if exists else "path is required"
if exists and kind and not kind_ok:
    detail = "%s; expected %s" % (resolved, kind)
checks.append(_check("path.%s" % name, passed, detail))
```

Add `kind: directory` to repository, official model, ComfyUI, deployment, and artifact entries in `configs/devices/l40x4-server.yaml`; add `kind: file` to trainer, smoke recipe, and training-data entries.

- [x] **Step 4: Run path/config tests and full verification**

```bash
.venv/bin/python -m pytest -q tests/unit/test_device_preflight.py tests/unit/test_l40x4_configs.py
.venv/bin/python -m compileall -q tools harness4h3 research
```

Expected: all selected tests pass and compilation exits 0.

- [x] **Step 5: Commit the path contract change**

```bash
git add tools/device_preflight.py configs/devices/l40x4-server.yaml tests/unit/test_device_preflight.py tests/unit/test_l40x4_configs.py
git commit -m "feat: validate device path kinds"
```

### Task 3: Publish the migration and baseline runbook

**Files:**
- Modify: `docs/device-porting.md`
- Modify: `docs/quickstart.md`
- Modify: `README.md`
- Modify: `research/README.md`

**Interfaces:**
- Use only existing commands: virtual-environment setup, `validate-config`, `tools/device_preflight.py`, and `harness4h3 benchmark`.
- Use `--result` for persisted preflight JSON and existing benchmark `--result` for EvaluationResult JSON.

- [x] **Step 1: Add the single operator sequence to the porting guide**

Document the exact four-L40 order:

```text
install repository environment
→ verify Harness offline
→ fill measured profile facts
→ run benchmark preflight with --result
→ start ComfyUI and verify /system_stats
→ run sanity baseline
→ only after sanity, run dev and heldout baseline
→ retain JSON, workflow, checkpoint hash, and environment evidence
```

Include both POSIX and PowerShell evidence-directory commands, the expected exit meanings, and a note that `configs/targets/rtx5080_example.yaml` must not be reused for an L40 claim unless its target constraints are intentionally adopted and recorded.

- [x] **Step 2: Clarify command tiers and evidence locations**

Update `docs/quickstart.md` with the `--result` example and label the offline fake optimization command as protocol-only. Update `README.md` and `research/README.md` so the current critical path is migration preflight and real baseline, while M6 is optional and A1 model-changing training remains blocked.

- [x] **Step 3: Check documentation links and wording**

Run:

```bash
.venv/bin/python -m pytest -q
rg -n -- "--result|M6|recovery_finetune|blocked|sanity" README.md docs/device-porting.md docs/quickstart.md research/README.md
```

Expected: tests pass; the docs contain an executable sequence and no statement that real training or autonomous model evolution is complete.

- [x] **Step 4: Commit the runbook update**

```bash
git add README.md docs/device-porting.md docs/quickstart.md research/README.md
git commit -m "docs: publish device migration runbook"
```

### Task 4: Final repository gate and handoff evidence

**Files:**
- Verify: `tools/device_preflight.py`
- Verify: `configs/devices/l40x4-server.yaml`
- Verify: `configs/experiments/a1-t0-l40x4.yaml`
- Verify: `configs/a1-worker.l40x4.example.json`
- Verify: all active documentation links

- [x] **Step 1: Run the full offline verification suite**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q tools harness4h3 research
```

Expected: all tests pass and compileall exits 0.

- [x] **Step 2: Run the L40 preflight locally as a safe blocked check**

```bash
.venv/bin/python tools/device_preflight.py \
  --profile configs/devices/l40x4-server.yaml \
  --operator benchmark \
  --skip-services \
  --result var/preflight/l40x4-local.json \
  --json
```

Expected on this Mac: exit code 2 and a persisted JSON result with status `blocked`; no trainer, ComfyUI, or remote command is started.

- [x] **Step 3: Verify only intended files changed**

```bash
git status --short
git diff --check
```

Expected: only the two pre-existing M6 working-tree edits remain uncommitted; all migration changes are committed.

- [x] **Step 4: Report the handoff boundary**

Report the committed files, validation results, and the exact next action on the remote host: run the persisted preflight, then establish a real H3 ComfyUI baseline. Do not report A1-T0, M0001, or autonomous real model evolution as complete.

