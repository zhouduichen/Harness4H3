# Wan pipeline device placement implementation plan

> **For Codex:** Execute this plan task-by-task. After each task, run the listed tests before moving on.

**Goal:** Make real Wan video evaluation place the complete inference pipeline on the configured pipeline device, while preserving generation-device and quality-device controls and preventing stale evaluation-cache reuse.

**Design:** Add an optional `pipeline_device` to `WanVideoEvaluationConfig`; default it to `device` for backward compatibility. Load Wan components through the pipeline-level `.to()` when available, fall back to individual components for test doubles, and use actual VAE placement for decode. Include the resolved placement in the evaluation recipe digest. Keep H3, Student, Fake Wan, and training paths unchanged.

## Task 1: Configuration and cache identity

**Files:** `harness4h3/evolution/evaluator.py`, `tests/unit/test_wan_video_evaluator.py`

1. Add `pipeline_device: str | None = None` and a resolved-device property.
2. Parse the field from mappings without changing the existing `device` default.
3. Add the resolved value to the recipe digest.
4. Test default placement, explicit CPU placement, and digest changes.

## Task 2: Real pipeline placement

**Files:** `harness4h3/evolution/evaluator.py`, `tests/unit/test_wan_video_evaluator.py`

1. Add a placement helper that prefers `pipeline.to(device)` and otherwise moves available transformer/VAE/text-encoder components.
2. Keep the transformer in eval mode and retain the existing execution-device override.
3. Test a fake pipeline with `.to()` and a fallback fake pipeline without it.

## Task 3: Prompt and VAE device safety

**Files:** `harness4h3/evolution/evaluator.py`, `tests/unit/test_wan_video_evaluator.py`

1. Create prompt embeddings on the resolved pipeline device.
2. Send prompt embeddings to the generation device at invocation time.
3. Decode frames on the VAE parameter device, with the resolved pipeline device as fallback; remove hard-coded CPU placement.
4. Test the VAE-device selection and preserve existing prompt-cache behavior.

## Task 4: Runner, regression, and remote verification

**Files:** `tools/run_wan_independent_evaluation.py`, relevant runner tests

1. Add an optional `--pipeline-device`; default each run to its generation GPU.
2. Run focused evaluator/runner tests, then `.venv/bin/python -m pytest -q`.
3. Commit only focused files, push `codex/rsi-compression-loop`, and preserve unrelated worktree changes.
4. Record and stop the old remote matrix, sync the focused files to the working remote checkout, and run a fresh fixed/RSI/coding/memory matrix with a new output root and explicit GPU pipeline placement.
5. Audit generated `evaluation.json`, decision, evidence, hashes, and summaries; report incomplete real evidence as `NOT VERIFIED`.

