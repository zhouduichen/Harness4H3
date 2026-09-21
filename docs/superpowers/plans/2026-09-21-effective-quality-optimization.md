# Effective Student Quality Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the remote H3-to-Student campaign accept only valid Students that preserve at least 90% of H3 quality and improve the quality-efficiency frontier.

**Architecture:** Add a deterministic evaluation manifest and H3 baseline calibration. Extend the Metric Verifier Bank with real semantic/temporal evidence, teacher-relative normalization, and Pareto decisions. Persist the baseline, incumbent, frontier, and patience in the detached campaign state and pass bounded deltas to the LLM.

**Tech Stack:** Python 3, PyTorch/CUDA, Transformers CLIP, OpenCV, YAML, JSON/JSONL, pytest, existing H3 adapter and remote Student workers.

## Global Constraints

- All baseline generation, Student generation, evaluation, training, and aggregation run on the remote server.
- `Q_student / Q_H3 >= 0.90` is a hard floor.
- Structural brightness/black-frame evidence is diagnostic only.
- Missing real semantic or temporal evidence fails closed.
- Existing compiler, trusted workers, SSH validation, and retention boundaries remain authoritative.
- The LLM cannot change metrics, thresholds, manifests, or worker commands.
- Offline tests remain CPU-safe and do not load CLIP weights.

---

### Task 1: Teacher-relative metrics and Pareto gate

**Files:** Modify `harness4h3/student/metrics.py`; test `tests/unit/test_student_metrics.py`.

**Interfaces:** Add `TeacherRelativeMetrics`, `ParetoDecision`, `teacher_relative_metrics(...)`, `normalize_reward(...)`, and `pareto_decision(...)`, each serializable through `to_dict()`.

- [ ] **Step 1: Add failing tests.** Verify `quality_ratio=0.72/0.80=0.90`, rejection at `0.899`, bounded `clip(log(reference/value), -1, 1)` terms, reward delta `0.02`, 5% material efficiency gain, 2% regression tolerance, and no-improvement rejection.
- [ ] **Step 2: Run ` .venv/bin/python -m pytest -q tests/unit/test_student_metrics.py ` and observe failures for the new symbols.
- [ ] **Step 3: Implement the dataclasses and normalization.** Use weights quality `0.60`, latency `0.20`, memory `0.10`, size `0.05`, optional energy `0.05`; renormalize when energy is absent.
- [ ] **Step 4: Implement the Pareto decision.** Reject below floor; accept reward delta at least `0.02`, or a non-dominated candidate with one efficiency gain at least `5%` and no required regression above `2%`. Record reason and metric deltas.
- [ ] **Step 5: Run the focused metric tests and require all pass.

---

### Task 2: Fixed evaluation manifest and real quality backend

**Files:** Create `harness4h3/student/evaluation_manifest.py` and `harness4h3/student/quality.py`; modify `harness4h3/student/inference.py` and `tools/student_evaluate_worker.py`; test `tests/unit/test_student_quality.py`.

**Interfaces:** Add `build_manifest(cache_dir, output_path, case_count=4, seeds=(20260920, 20260921))`, `EvaluationManifest.from_path(path)`, `ClipTemporalQualityBackend.evaluate(video_path, caption)`, and `generate_video(..., cache_path=None, seed=...)`.

- [ ] **Step 1: Add tests for stable manifest digest, explicit cache selection, missing captions, mocked CLIP scores in `[0,1]`, and temporal aggregation without loading weights.
- [ ] **Step 2: Implement sorted cache selection and fixed seeds.** If fewer cache files exist than requested cases, reuse them with distinct stable IDs/seeds and record `unique_cache_items`; never glob-select the first item during evaluation.
- [ ] **Step 3: Implement lazy `transformers.CLIPModel`/`CLIPProcessor` loading from a local path.** Sample four frames, compute normalized image-text cosine semantic score, adjacent-frame cosine temporal score, and `0.70*semantic + 0.20*temporal + 0.10*motion`. Raise `QualityBackendUnavailable` when package/model is missing.
- [ ] **Step 4: Make the trusted evaluator generate one MP4 per case/seed and aggregate median/p95 latency and maximum peak memory. Decode validity remains a hard gate.
- [ ] **Step 5: Run `.venv/bin/python -m pytest -q tests/unit/test_student_metrics.py tests/unit/test_student_quality.py` with no network or model download.

---

### Task 3: H3 baseline and strict campaign state

**Files:** Create `tools/student_teacher_baseline_worker.py`; modify `harness4h3/student/config.py`, `harness4h3/student/remote.py`, `harness4h3/student/campaign.py`, and `tools/student_campaign_supervisor.py`; test `tests/unit/test_student_campaign.py` and `tests/unit/test_student_config.py`.

**Interfaces:** Add `TeacherBaselineResult`, `RemoteStudentBaseline.run(manifest)`, and `StudentCampaign(..., teacher_baseline, quality_policy)`.

- [ ] **Step 1: Add tests proving valid-but-dominated rounds do not succeed, `q_ratio < 0.90` records `quality_floor_failed`, and lower quality with material latency gain can enter the frontier.
- [ ] **Step 2: Validate these config fields and defaults: `evaluation_manifest`, `clip_model_path`, `quality_backend=clip_temporal`, `quality_floor_ratio=0.90`, `min_reward_delta=0.02`, `material_efficiency_gain=0.05`, `max_metric_regression=0.02`, `no_improvement_patience=3`, `max_rounds=8`.
- [ ] **Step 3: Implement the trusted baseline worker.** For each manifest case, decode cached H3 video latent through the configured H3 VAE, run the same quality backend, and atomically write `teacher-baseline.json`. Never infer teacher quality from Student output.
- [ ] **Step 4: Run baseline before the first proposal and fail closed on baseline/backend failure. Persist the baseline digest in `resume.json` and every event.
- [ ] **Step 5: Persist `teacher_baseline`, `incumbent`, `frontier`, `best_reward`, and `no_improvement_rounds`; include bounded frontier/delta summaries in the next LLM context. Stop only on `campaign_success`, `no_pareto_improvement`, or `campaign_budget_exhausted`.
- [ ] **Step 6: Run `.venv/bin/python -m pytest -q tests/unit/test_student_campaign.py tests/unit/test_student_config.py`.

---

### Task 4: Docs, configuration, and integration contract

**Files:** Modify `configs/student-campaign.example.yaml`, `docs/student-campaign.md`, and `docs/metric-verifier-bank.md`; create `tests/integration/test_student_effective_optimization.py`.

- [ ] **Step 1: Set strict example defaults: max rounds 8, patience 3, H3 floor 0.90, local CLIP path, manifest path, and trusted baseline/evaluator commands.
- [ ] **Step 2: Add a deterministic fake-backend integration test that records baseline digest, frontier, metric deltas, and `no_pareto_improvement` when no candidate improves.
- [ ] **Step 3: Document remote manifest build, validation, detached launch, result inspection, and the difference between effective optimization and a valid-video smoke run.
- [ ] **Step 4: Run `git diff --check`, `.venv/bin/python -m pytest -q tests -k 'student or campaign or metric'`, and `.venv/bin/python -m compileall -q harness4h3/student tools/student_*.py`.
- [ ] **Step 5: Commit with `feat: enforce effective student quality optimization` and push `main`.

---

### Task 5: Deploy and launch the strict server experiment

**Files:** Deploy to `/home/intern/huangjiahao/Harness4H3-effective-20260921`; use campaign root `/home/intern/huangjiahao/Harness4H3-effective-20260921/work/student-campaign-effective-20260921`.

**Interfaces:** Remote Python `/home/intern/miniconda3/envs/comfy/bin/python`; vLLM `http://127.0.0.1:8000/v1`, model `qwen3.5-controller`; H3 cache `/data/models/MiniMax-H3/harness4h3/cache`; CLIP `/data/models/clip-vit-large-patch14`.

- [ ] **Step 1: Export the pushed tree into the isolated remote root and force `PYTHONPATH` to it. Leave the dirty legacy checkout and processes untouched.
- [ ] **Step 2: Build the fixed manifest and verify digest, captions, case count, and local CLIP path.
- [ ] **Step 3: Run baseline-only preflight and require finite semantic/temporal evidence.
- [ ] **Step 4: Start detached with max rounds 8, patience 3, and H3 floor 0.90; record PID, log, root, and commit.
- [ ] **Step 5: Perform one bounded launch check for supervisor, baseline file, and first event digests; do not continuously monitor from Codex.
- [ ] **Step 6: Report the result path. Only a post-baseline Pareto improvement may be called effective optimization; valid-video-only output is `no_pareto_improvement` or another explicit failure.
