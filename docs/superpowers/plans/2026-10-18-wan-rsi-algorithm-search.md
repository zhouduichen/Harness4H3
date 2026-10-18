# Wan RSI Algorithm Search Improvement Implementation Plan

> **For agentic workers:** Execute this plan task by task. The user authorized implementation, real short-run experiments, tests, and commit in the request.

**Goal:** Repair evidence-driven multi-granularity search for Wan RSI and compare one fixed search baseline with three real LLM-driven descendants under one unchanged model and evaluation protocol.

**Architecture:** Keep the existing profiler, Algorithm Program, Sandbox, operator registry, measurement evaluator, EvolutionLoop, Wan Video Evaluator, and PromotionPolicy. Add temporal Block evidence and budget-aware Pareto selection within the current 8-measurement budget. Record Controller identity and API responses in existing algorithm/run artifacts.

**Tech Stack:** Python, PyTorch, Diffusers Wan 2.1, vLLM OpenAI-compatible endpoint, pytest, remote CUDA.

## Global Constraints

- Do not add operators, replace the Harness, or lower quality thresholds.
- Never rank invalid or non-finite candidate measurements.
- All candidates in each algorithm comparison use the same original Checkpoint, Calibration probes, and 8-measurement budget.
- Validation uses at least 3 distinct Prompts and 2 Seeds, 2 latency warmups, and 5 repeats.
- Historical experiment output roots remain read-only.

---

### Task 1: Preserve the ten-round evidence audit

**Files:**
- Modify: `docs/superpowers/reports/2026-10-17-candidate-selection-audit.md`
- Create: `docs/superpowers/reports/2026-10-18-wan-rsi-search-improvement.md`
- Modify: `docs/superpowers/reports/2026-10-17-candidate-selection-summary.json`

- [ ] Add a 10-row round table with each round's 12 generated plans, 8 attempted indices, attempted families, actual selected plan, internal validity, and external validation outcome.
- [ ] Add a plan-level table for all candidate families and exact registered operator args; distinguish budget-skipped plans from invalid measurements.
- [ ] Record A0001–A0010 parent/source lineage, AST-level behavior changes, SearchSpace P0001–P0008 status, error counts, and the limits of the separate runtime probe.
- [ ] Record per-family Pareto and independent video metrics without inferring GPU gains from runtime counters.

### Task 2: Add Block-by-timestep profiling and evidence-driven candidates

**Files:**
- Modify: `harness4h3/profiler/sensitivity.py`
- Modify: `harness4h3/profiler/layer_profiler.py`
- Modify: `harness4h3/profiler/report.py`
- Modify: `harness4h3/evolution/agent.py`
- Test: `tests/unit/test_algorithm_program_search.py`
- Test: `tests/unit/test_profiler_calibration.py`

- [ ] Measure Block removal output delta at high-noise, middle, and low-noise timesteps and persist the profile as `block_timestep_sensitivity`.
- [ ] Include measured low-sensitivity Block/Head regions, profile-backed Token/KV settings, and available timestep/Sparse/Window operations in bounded Agent context.
- [ ] Generate candidates only when the active SearchSpace and required profile evidence support them; record why candidates were omitted.
- [ ] Add a compatible `head_gating + kv_token_select + sparse_runtime` composition only when active policies can share the adapter execution path; never claim the combination will win.
- [ ] Retain the existing fixed seed algorithm as the human baseline and make its `search` entrypoint explicit so it spends the same candidate budget as LLM descendants.

### Task 3: Make budget allocation and Pareto selection evidence-safe

**Files:**
- Modify: `harness4h3/evolution/search_ranking.py`
- Modify: `harness4h3/evolution/algorithm_evaluation.py`
- Modify: `harness4h3/evolution/loop.py`
- Test: `tests/unit/test_search_ranking.py`
- Test: `tests/integration/test_standard_search_selection.py`

- [ ] Count prior executed candidates by family for the same immutable parent checkpoint and allocate up to 8 evaluations to the least-explored supported families first.
- [ ] Rank valid candidates on a shared Pareto front using calibration quality, observed efficiency reductions, and risk; keep unmeasured hardware values null.
- [ ] Return a real measured winner only from valid evidence. When every measurement is invalid, stop before compression/video evaluation and retain a complete failure trace.
- [ ] When no valid candidate passes Calibration quality, allow only the best valid Pareto near-miss to enter independent Validation with `validation_only` status; keep PromotionPolicy unchanged.
- [ ] Record generated, duplicate, unsupported, budget-skipped, attempted, valid, Pareto, and selected counts with original candidate index and rank position separately.

### Task 4: Record real LLM provenance and separated evaluation protocol

**Files:**
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/cli.py`
- Modify: `harness4h3/evolution/config.py`
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/algorithm_evaluation.py`
- Create: `configs/evolutionary-compression.wan-rsi-short-fixed-20261018.yaml`
- Create: `configs/evolutionary-compression.wan-rsi-short-llm-20261018.yaml`
- Test: `tests/unit/test_algorithm_evolution.py`

- [ ] Store provider, requested model, served model version, API call/response ID, parent algorithm ID, and parent source SHA with each algorithm candidate.
- [ ] Store the resolved evaluation protocol once at the output root, including baseline checkpoint SHA and Calibration sample seeds.
- [ ] Measure each candidate over 3 identical parent-relative Calibration probes and aggregate conservatively.
- [ ] Configure both experiment arms with the same checkpoint, profile settings, search budget, 3-Case/2-Seed Validation manifest, 2 latency warmups, and 5 repeats.

### Task 5: Verify, run the real comparison, report, and commit

**Files:**
- Modify: `docs/superpowers/reports/2026-10-18-wan-rsi-search-improvement.md`
- Test: all affected unit/integration suites

- [ ] Run focused tests, full CPU regression, `compileall`, and `git diff --check`.
- [ ] On the remote Wan host, run 1 fixed-baseline round then 3 vLLM rounds into one new output root; verify the `/v1/chat/completions` API response, real generated source, descendant source lineage, and all evaluation artifacts.
- [ ] Compute parent/child and fixed/LLM Pareto comparisons only from the same parent checkpoint, same budget, and same protocol.
- [ ] Report the five required conclusions, preserve all negative results, and commit only task-relevant files without rewriting unrelated existing work.
