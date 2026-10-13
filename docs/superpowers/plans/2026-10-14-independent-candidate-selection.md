# Independent Candidate Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make independent Wan evaluation measure an explicitly selected rejected Child instead of silently reusing the active baseline.

**Architecture:** Extend the existing read-only evaluation launcher with a per-run model-ID override. Keep active-model selection as the backward-compatible default, but record the exact selected model and selection mode in every summary. The evaluator and model-loading path remain unchanged.

**Tech Stack:** Python 3, argparse, multiprocessing spawn, pytest, existing Harness4H3 Wan evaluator.

## Global Constraints

- Do not change the existing evaluator semantics or H3/Wan training paths.
- Do not overwrite unrelated Student changes in the dirty worktree.
- Explicit candidate IDs must fail closed when the candidate record is missing.
- Real evidence must include the selected candidate ID and checkpoint identity.

---

### Task 1: Add candidate-selection tests

**Files:**
- Modify: `tests/unit/test_independent_evaluation.py`
- Modify: `tools/run_wan_independent_evaluation.py`

**Interfaces:**
- Add `_parse_candidate_models(values: Sequence[str]) -> dict[str, str]`.
- Extend `_active_artifacts(root: Path, candidate_model_id: str | None = None) -> tuple[CheckpointArtifact, CheckpointArtifact, str, str]`, returning `(baseline, candidate, active_id, selection_mode)`.

- [ ] **Step 1: Write tests for explicit Child selection and fail-closed lookup**

  Create baseline, active, and rejected candidate JSON records under a temporary
  run root. Assert that `_active_artifacts(root, "M0003")` returns the rejected
  candidate path and `selection_mode == "explicit"`; assert a missing `M9999`
  raises `FileNotFoundError` mentioning the candidate record.

- [ ] **Step 2: Write tests for CLI mapping validation**

  Assert `_parse_candidate_models(["fixed=M0003", "coding=M0002"])` returns the
  two mappings, while malformed input and duplicate names raise `ValueError`.

- [ ] **Step 3: Run the focused tests and confirm they fail**

  Run:

  ```bash
  .venv/bin/pytest -q tests/unit/test_independent_evaluation.py
  ```

  Expected: failures because the explicit-selection helpers are not implemented.

### Task 2: Implement explicit candidate resolution and summary provenance

**Files:**
- Modify: `tools/run_wan_independent_evaluation.py`
- Modify: `tests/unit/test_independent_evaluation.py`

**Interfaces:**
- `RunSpec` gains `candidate_model_id: str | None`.
- Add repeated CLI option `--candidate-model NAME=MODEL_ID`.
- `_evaluate_one` receives the selected model ID and writes
  `candidate_model_id`, `candidate_selection`, and `candidate_checkpoint_path`.

- [ ] **Step 1: Implement strict candidate resolution**

  Resolve `models/candidates/<candidate_model_id>.json` when an override is
  supplied. Check that it is a regular file before reading it. Only use
  `active.json` when the override is `None`.

- [ ] **Step 2: Implement mapping parsing and RunSpec wiring**

  Parse each `NAME=MODEL_ID`, reject empty names/IDs and duplicate names, then
  attach the mapping to the matching `RunSpec`. Reject an override name that is
  not present in `--run`.

- [ ] **Step 3: Record selection provenance**

  Add to each per-run summary:

  ```python
  {
      "active_model_id": active_id,
      "candidate_model_id": selected_id,
      "candidate_selection": "explicit" | "active_default",
      "candidate_checkpoint_path": candidate.checkpoint_path,
  }
  ```

- [ ] **Step 4: Run focused tests**

  Run:

  ```bash
  .venv/bin/pytest -q tests/unit/test_independent_evaluation.py
  ```

  Expected: all focused tests pass, including default compatibility and
  fail-closed explicit lookup.

### Task 3: Run the complete regression suite and commit

**Files:**
- Modify only the files from Tasks 1–2.

- [ ] **Step 1: Run targeted integration checks**

  ```bash
  .venv/bin/pytest -q tests/integration tests/unit/test_compare_evolution_runs.py
  ```

- [ ] **Step 2: Run the complete CPU suite**

  ```bash
  .venv/bin/pytest -q
  ```

  Expected: no collection errors and the existing passing baseline is not
  reduced.

- [ ] **Step 3: Commit only scoped files**

  ```bash
  git add tools/run_wan_independent_evaluation.py tests/unit/test_independent_evaluation.py
  git commit -m "fix: select rejected candidates in independent evaluation"
  git push origin codex/rsi-compression-loop
  ```

### Task 4: Re-run real held-out Child evaluations

**Files:**
- Remote outputs only; do not overwrite the previous baseline-only results.

- [ ] **Step 1: Sync the committed launcher to the remote evaluation checkout**

  Copy the committed launcher to the remote test checkout and verify its SHA256
  before starting workers.

- [ ] **Step 2: Start explicit candidate evaluations**

  Use distinct output roots and explicit model IDs:

  ```bash
  --candidate-model fixed=M0003
  --candidate-model memory=M0003
  --candidate-model rsi=M0003
  --candidate-model coding=M0002
  ```

- [ ] **Step 3: Verify each result**

  Require 16/16 cases, `offline_simulation: false`, valid videos, matching
  manifest/evaluator recipe, and `candidate_checkpoint_sha256 != baseline`.
  Compare candidate quality and hardware metrics against the corresponding
  baseline before drawing a research conclusion.
