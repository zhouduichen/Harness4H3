# Checkpoint Retention V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add storage-safe checkpoint retention to the A0 campaign lifecycle so A1 rejects release staged child weights only after final evaluation while preserving all experiment evidence.

**Architecture:** Keep worker staging and all Harness4H3-v1.0 semantics unchanged. Add small path-safe retention functions beside `A0Campaign` in `research/experiments/a0_model_evolution.py`; the campaign calls them after final outcome classification and stores the returned JSON record inside each iteration and trajectory. Tests exercise the helper directly and one campaign integration path.

**Tech Stack:** Python 3.9+, `pathlib`, existing campaign dataclasses, pytest, JSONL trajectory store.

## Global Constraints

- Do not modify the Evaluator, TargetProfile, reward, Controller decision logic, acceptance rules, or Harness4H3-v1.0 semantics.
- Do not modify `h3_training/engine/checkpoint.py` or resume/full-state checkpoint semantics.
- Keep `tools/h3_model_worker.py` staging child checkpoints under experiment `artifacts_dir`.
- Delete only resolved regular files strictly inside the current campaign `output_root`.
- Refuse outside-root, URI, parent, M0000, directory, and symlink deletion targets.
- Missing deletion targets are non-fatal no-ops; actual deletion failures are recorded as errors.
- Retain candidate metadata, ExperimentPlan, EvaluationResult, metrics, hashes, trajectories, and failure/reject evidence.
- Do not implement delta checkpoints, keep-last-N accepted retention, or Pareto candidate deletion.

---

### Task 1: Add failing retention safety tests

**Files:**
- Modify: `tests/unit/test_a0_model_evolution.py`
- Test: `tests/unit/test_a0_model_evolution.py`

**Interfaces:**
- Tests consume the planned `apply_checkpoint_retention` helper and the existing `run_campaign` result format.
- Tests produce the expected retention record contract for the campaign integration.

- [x] **Step 1: Add direct helper tests for rejected, accepted, M0000, outside-root, and missing paths**

Use local files under `tmp_path / "campaign"`, pass the campaign root explicitly, and assert only the rejected in-root child is removed. Assert outside-root and M0000 paths remain byte-for-byte unchanged, accepted paths remain present, and a missing path returns without raising.

- [x] **Step 2: Add a failed-experiment cleanup test**

Create `runs/exp_0001/artifacts/trainer-child.pt`, `optimizer_state.pt`, and `reject.evidence.json`; call the failed-experiment retention path with the parent outside the run directory; assert only checkpoint/optimizer files are removed and the evidence file remains.

- [x] **Step 3: Add a campaign-level preservation test**

Run a one-iteration rejected campaign with an operator backend that returns a local child checkpoint under the campaign run artifacts directory. Assert the child is gone, `models/candidates/M0001.json`, `campaign.json`, `report.json`, and `trajectories.jsonl` remain, and the iteration's `checkpoint_retention` is present with `deleted is True`.

- [x] **Step 4: Run the new focused tests and verify they fail before implementation**

Run:

```bash
python -m pytest -q tests/unit/test_a0_model_evolution.py -k 'retention or rejected_local_child'
```

Expected: collection or assertion failures because the helper and campaign retention record do not yet exist.

### Task 2: Implement path-safe retention helpers

**Files:**
- Modify: `research/experiments/a0_model_evolution.py`

**Interfaces:**
- Produces `apply_checkpoint_retention(output_root, outcome, checkpoint_path, parent_checkpoint_path, model_id, experiment_dir=None, cleanup_paths=()) -> Dict[str, Any]`.
- The returned dictionary always contains `policy`, `checkpoint_path`, `retained`, `reason`, `deleted`, and `delete_error`; failed cleanup may additionally contain `deleted_paths`.

- [x] **Step 1: Add local-path resolution and containment checks**

Resolve `output_root` and each target with `Path.resolve(strict=False)`. Reject URI-like paths, root itself, paths not relative to the resolved output root, symlinks, and protected parent/M0000 paths before any unlink. Use `Path.relative_to` rather than string-prefix checks.

- [x] **Step 2: Add non-recursive unlink with no-op and error reporting**

Unlink only regular files. Return `(True, None)` when a target is already absent, `(False, error)` for refusal or `OSError`, and never raise retention exceptions into the campaign.

- [x] **Step 3: Implement outcome-specific policy**

For `accepted_candidate`, return retained metadata without unlinking. For `rejected_candidate`, attempt only the candidate checkpoint path. For `failed_experiment`, inspect explicit cleanup paths and the experiment artifact subtree for checkpoint/optimizer filename patterns, excluding JSON/log/evidence and protected paths.

- [x] **Step 4: Run the focused helper tests**

Run:

```bash
python -m pytest -q tests/unit/test_a0_model_evolution.py -k 'retention or rejected_local_child'
```

Expected: all retention safety tests pass.

### Task 3: Integrate retention after final campaign outcome

**Files:**
- Modify: `research/experiments/a0_model_evolution.py`
- Modify: `tests/unit/test_a0_model_evolution.py`

**Interfaces:**
- Campaign calls `apply_checkpoint_retention` only after evaluator completion/final outcome classification and before iteration persistence.
- Existing `execution_result`, `evaluation`, plan fields, metrics, and trajectory evidence remain unchanged; `checkpoint_retention` is additive.

- [x] **Step 1: Derive the candidate checkpoint and experiment directory**

Use the created `child.checkpoint_path` when available, otherwise the successful operator output state path; use `output_root / "runs" / experiment_id` for failed experiment cleanup.

- [x] **Step 2: Call retention and add the returned record to each iteration**

Insert `checkpoint_retention` into `record` before appending `iterations`, persisting `campaign.json`, or constructing the trajectory. Do not change `ExperimentPlan` or `EvaluationResult` dictionaries.

- [x] **Step 3: Include retention in the trajectory evidence**

Add an additive trajectory step containing the same retention record so storage decisions remain inspectable after a checkpoint is removed.

- [x] **Step 4: Run all campaign and A1 tests**

Run:

```bash
python -m pytest -q tests/unit/test_a0_model_evolution.py tests/unit/test_a1_real_evolution.py
```

Expected: PASS, including the existing A1 reuse of `run_campaign`.

### Task 4: Full verification and requirement audit

**Files:**
- Verify: `research/experiments/a0_model_evolution.py`
- Verify: `tests/unit/test_a0_model_evolution.py`
- Verify unchanged: `tools/h3_model_worker.py`, `h3_training/engine/checkpoint.py`

- [x] **Step 1: Run the complete pytest suite**

```bash
python -m pytest -q
```

Expected: exit code 0 with all existing and new tests passing.

- [x] **Step 2: Run compileall for the requested packages**

```bash
python -m compileall -q harness4h3 h3_training research tools
```

Expected: exit code 0 and no syntax errors.

- [x] **Step 3: Review diff and verify protected files were not modified**

```bash
git diff --check
git diff -- research/experiments/a0_model_evolution.py tests/unit/test_a0_model_evolution.py tools/h3_model_worker.py h3_training/engine/checkpoint.py
```

Confirm no changes to worker staging or full-state checkpoint semantics, and confirm deletion code only receives resolved paths under the campaign output root.

- [x] **Step 4: Report exact retention behavior and test results**

Report changed files, the post-evaluator lifecycle point, files deleted for rejected/failed experiments, files never deleted, and the exact pytest/compileall outcomes.
