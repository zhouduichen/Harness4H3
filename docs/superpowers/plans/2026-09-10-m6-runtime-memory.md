# M6 Cross-Layer Runtime Memory Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add controlled runtime-memory operators and a real M6 acceptance runner that can reduce peak VRAM from the validated M0001 NVFP4 candidate without changing model weights or Harness Evolution.

**Architecture:** Runtime operators clone `ModelState` with a serializable `runtime_policy`; `H3BenchmarkRunner` applies that policy to a copied ComfyUI workflow and fails explicitly when a requested capability is absent. `M6ValidationRunner` evaluates one branch at a time on dev/held-out tasks, aggregates max peak memory and quality/efficiency metrics, and accepts only the strict hard gates.

**Tech Stack:** Python 3.9+, dataclasses, existing OperatorRegistry/ModelCandidate interfaces, ComfyUI HTTP adapter, independent subprocess evaluator, pytest, JSON/JSONL evidence.

## Global Constraints

- Start from validated M0001 NVFP4; parent checkpoint remains immutable.
- One primary runtime intervention per branch; no aggressive re-quantization or step reduction.
- TargetProfile quality thresholds remain immutable; strict memory gate uses peak max.
- Preserve Operator Attribution, cache reset, independent evaluator, Pareto branches, and append-only trajectory records.
- Do not implement Harness Evolution, surrogate search, kernel/compiler search, or controller training.

### Task 1: Runtime policy primitives and operators

**Files:**
- Create: `harness4h3/operators/runtime_memory.py`
- Modify: `harness4h3/operators/__init__.py`
- Modify: `harness4h3/operators/fake.py`
- Test: `tests/unit/test_runtime_memory.py`

**Interfaces:**
- `RuntimePolicyOperator(name, description, policy_kind, allowed_args)` implements `schema`, `validate`, `estimate_cost`, and `execute`.
- `build_runtime_registry(backend=None)` returns the existing fake registry plus `runtime_offload`, `vae_tiling`, and `inference_chunking`.
- `runtime_state["runtime_policy"]` contains `{ "kind": str, "args": {...} }`.

- [ ] Write tests for schemas, argument validation, immutable child cloning, and unsupported values.
- [ ] Run `PYTHONPATH=. pytest tests/unit/test_runtime_memory.py -q` and observe the new tests fail.
- [ ] Implement the three operators with deterministic policy metadata and no checkpoint mutation.
- [ ] Run the focused tests and then the full suite; expect all existing tests plus the new runtime tests to pass.
- [ ] Commit `feat: add runtime memory operators`.

### Task 2: Apply runtime policies to ComfyUI workflows

**Files:**
- Modify: `harness4h3/benchmark/h3.py`
- Modify: `harness4h3/backends/comfyui.py`
- Test: `tests/unit/test_h3_benchmark.py`

**Interfaces:**
- `H3BenchmarkRunner._workflow` reads `state.runtime_state.runtime_policy` and returns a copied workflow.
- `apply_runtime_policy(workflow, policy)` performs concrete node/input changes or raises `BackendError` with `runtime_policy_unsupported`.

- [ ] Add tests proving policy application does not mutate the template and unsupported nodes fail explicitly.
- [ ] Implement offload controls and guarded tiled/chunked node rewrites.
- [ ] Run focused and full tests.
- [ ] Commit `feat: apply runtime policies to benchmark workflows`.

### Task 3: Max-based M6 validation and controller context

**Files:**
- Create: `harness4h3/benchmark/m6.py`
- Modify: `harness4h3/benchmark/__init__.py`
- Modify: `harness4h3/controller/context.py`
- Modify: `harness4h3/controller/provider.py`
- Test: `tests/unit/test_m6_validation.py`

**Interfaces:**
- `M6ValidationRunner.run(parent, branch, tasks, target, operator_attribution, ...) -> M6ValidationResult`.
- `M6ValidationResult.to_dict()` includes per-run conditions, aggregate max/mean/median/min/std/CI, gate booleans, and branch decision.
- Controller context includes read-only validated Design Gene evidence and available runtime operator schemas.

- [ ] Write failing tests for max-memory rejection, quality/black-frame gates, and branch isolation.
- [ ] Implement aggregation and strict gate evaluation.
- [ ] Run focused/full tests.
- [ ] Commit `feat: add max-based M6 validation`.

### Task 4: Real M6 experiment entry point and evidence

**Files:**
- Create: `experiments/m6_runtime_memory.py`
- Modify: `README.md`
- Modify: `docs/real-experiments/2026-09-09-windows-rtx5080.md`
- Modify: `docs/experience/design-gene-h3-nvfp4.json`

**Interfaces:**
- CLI evaluates selected runtime branches on dev and held-out tasks, supports `--branches` and atomic incremental JSON output, and exits zero only when all selected branches satisfy M6 gates.
- Evidence records retain rejected branches and never claim TargetProfile feasibility from averages.

- [ ] Add the real runner and fixed branch definitions.
- [ ] Run unit tests and a remote RTX 5080 smoke/acceptance matrix.
- [ ] Update evidence and Design Gene status only from measured max values.
- [ ] Commit `feat: execute M6 runtime memory validation`.

