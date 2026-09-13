# A1 Real Model Evolution Implementation Plan

> **For agentic workers:** This plan is executed inline. Keep the frozen Harness and A0 orchestration unchanged except for correctness-only integration wiring.

**Goal:** Connect the existing A0 external-operator seam to a real H3 model-changing worker and real H3 benchmark so the first two autonomous plans can produce and evaluate real child checkpoints.

**Architecture:** `ExternalScriptOperator` remains the Harness boundary. A new worker adapter receives its machine-readable request, invokes a trusted trainer command with `shell=False`, stages optional teacher-signal cache metadata, validates and copies the child checkpoint into the experiment artifact directory, and returns a validated `ModelState`. A new A1 runner reuses `run_campaign` with an external registry and a tiered `H3BenchmarkRunner` evaluator; no training logic enters `harness4h3/operators`.

**Tech Stack:** Python 3.9+, existing `LocalProcessExecutor`, `ExternalScriptOperator`, `H3Inspector`, `H3BenchmarkRunner`, JSON worker protocol, pytest.

## Global Constraints

- `Harness4H3-v1.0` remains frozen; no Controller schema, evaluator semantics, archive semantics, target semantics, or A0 parent/budget rules change.
- The worker never accepts a command, path, threshold, or environment mutation from the Controller request.
- Parent checkpoint bytes must remain unchanged; child checkpoint must be a new file under the experiment artifacts directory.
- A1 only reports real execution when an external trainer and real ComfyUI/evaluator are configured; contract fixtures are explicitly non-scientific.
- Baseline quality and hardware metrics must be supplied or loaded from validated evidence; they are never inferred from an offline simulator.

---

### Task 1: Implement the real worker adapter

**Files:**
- Create: `tools/h3_model_worker.py`
- Create: `configs/a1-worker.example.json`
- Create: `tests/fixtures/h3_model_trainer_fixture.py`
- Create: `tests/unit/test_h3_model_worker.py`

- [x] Implement fixed-config trainer invocation, teacher-cache manifest handling, child checkpoint staging/deployment, result validation, and structured failure output.
- [x] Test parent immutability, new child path, worker metrics, and invalid trainer results.

### Task 2: Implement real benchmark adapter

**Files:**
- Create: `experiments/a1_real_evolution.py`
- Create: `tests/unit/test_a1_real_evolution.py`

- [x] Implement tiered task selection over the existing `H3BenchmarkRunner`.
- [x] Convert real `BenchmarkSummary` to the evaluator result consumed by A0.
- [x] Force A1 bootstrap plans through Tier 3 while leaving later A0 tier selection unchanged.
- [x] Test that the second plan sees the first real result and that benchmark failures become failed experiments.

### Task 3: Add A1 CLI and evidence documentation

**Files:**
- Modify: `harness4h3/cli.py`
- Modify: `README.md`
- Modify: `docs/evogen-a0-model-evolution.md`
- Create: `docs/evogen-a1-real-model-evolution.md`
- Create: `tests/integration/test_a1_cli.py`

- [x] Add `a1-evolve` with required parent checkpoint, worker command, baseline metrics, ComfyUI endpoint, and two-experiment default.
- [x] Fail closed when parent/worker/baseline requirements are missing.
- [x] Document same-host/remote deployment requirements and teacher caching.

### Task 4: Verify and run preflight

- [x] Run worker and A1 focused tests.
- [x] Run the full offline suite and compileall.
- [x] Confirm frozen-module diff is empty.
- [x] Attempt a read-only SSH preflight to the configured RTX 5080 host; record the unavailable trainer and ComfyUI service in `docs/real-experiments/2026-09-10-a1-preflight.md` without fabricating A1 results.
