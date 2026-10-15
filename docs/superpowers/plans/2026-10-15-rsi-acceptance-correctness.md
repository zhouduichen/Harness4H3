# TICA-Prune v2 Correctness and RSI Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute the tasks in this plan task-by-task with a focused test cycle after each task.

**Goal:** Make TICA-Prune v2 fail closed on full-risk evidence, isolate Evolution from frozen Test evaluation, compare parent and child search algorithms through complete TICA searches, and provide reproducible CPU Fake and Controller protocol acceptance evidence.

**Architecture:** Preserve the existing trusted Wan operators, `TICAPruner`, `EvolutionLoop`, `AlgorithmValidator`, `AlgorithmSandbox`, `SkillHub`, and `PromotionPolicy`. Add explicit evidence fields and phase guards at their current boundaries: TICA search owns Calibration-only candidate search, AlgorithmValidator owns Calibration-search plus Validation comparison, and a new final-evaluation path reads frozen Test artifacts without writing adaptive state. The real Controller receives a mode-specific bounded schema; no arbitrary Python/PyTorch/CUDA execution is introduced.

**Tech Stack:** Python 3.9+, pytest, existing JSON artifacts, FakeWanDiT adapter, existing HTTP-compatible Controller provider.

## Global Constraints

- Do not alter historical four-arm experiment artifacts or retroactively apply the new protocol to them.
- Do not load a large Wan checkpoint or start a GPU experiment in this task; add only preflight interfaces and configuration.
- Calibration is the only input to TICA sensitivity, candidate generation, interaction analysis, and search.
- Validation is the only adaptive comparison/promotional evidence for Evolution; Test is frozen final evaluation only.
- CPU Fake evidence must never be labeled as real Wan video, GPU latency, or peak-memory evidence.
- Missing, non-finite, zero, or scale-incompatible measurements fail closed.
- All algorithm outputs remain restricted to Sandbox and trusted registered operators.

---

### Task 1: Make full TICA risk authoritative and auditable

**Files:**
- Modify: `harness4h3/evolution/tica_prune.py`
- Modify: `harness4h3/evolution/algorithm_evaluation.py`
- Test: `tests/unit/test_tica_prune_v2.py`
- Test: `tests/unit/test_tica_evaluation.py`

**Interfaces:**
- Fast measurements return `risk_metric`, `risk_value`, `risk_scale`, and `risk_is_approximate`.
- Full measurements return the same risk contract from fixed calibration inputs.
- `TICACandidate.feasible` is true only when structural validity, full validity, full-risk threshold, and interaction constraints all pass.

- [ ] Add a failing Fast-low/Full-high test and a scale-mismatch test.
- [ ] Define one normalized TICA risk contract and reject full results with missing/non-finite or incompatible scale.
- [ ] Recompute feasibility after full evaluation; never fall back to fast risk.
- [ ] Ensure Pareto and selected-candidate logic sees only full-risk-feasible candidates.
- [ ] Record fast/full risk provenance and rejection reason in candidate and budget artifacts.

### Task 2: Enforce Evolution versus frozen FinalEvaluation phases

**Files:**
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/cli.py`
- Modify: `harness4h3/evolution/config.py`
- Modify: `harness4h3/evolution/evaluator.py`
- Modify: `harness4h3/evolution/protocol.py`
- Create or modify: `harness4h3/evolution/final_evaluation.py`
- Test: `tests/unit/test_tica_protocol.py`
- Test: `tests/unit/test_evolution_protocol.py`

**Interfaces:**
- Evolution mode accepts only Calibration and Validation manifests and rejects `evaluation_split=test`.
- `FinalEvaluation` consumes frozen algorithm/model/protocol identifiers and returns results without writing `ExperienceStore`, `SkillHub`, `SearchPolicy`, ModelStore active state, or AlgorithmStore validation state.
- Final results include frozen IDs, source revision, protocol digest, and `adaptive_state_mutated: false`.

- [ ] Add phase/config validation tests that reject Test in Evolution.
- [ ] Add a final-evaluation request/result contract with immutable source and model hashes.
- [ ] Route Validation through PromotionPolicy and algorithm comparison only; route Test through FinalEvaluation only.
- [ ] Add regression tests proving poor Test metrics cannot change Algorithm ID, Model ID, SearchPolicy, or Agent context.
- [ ] Update the TICA Wan example and runbook to use the new phase-specific configuration.

### Task 3: Compare complete parent and child TICA searches

**Files:**
- Modify: `harness4h3/evolution/improvement.py`
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/algorithm_evaluation.py`
- Test: `tests/unit/test_algorithm_improvement.py`
- Test: `tests/integration/test_tica_recursive_loop.py`

**Interfaces:**
- `AlgorithmValidator` accepts a complete-search evaluator that returns candidates, Pareto front, selected plan, budget audit, and Validation measurements for both algorithms.
- Parent and child use the same immutable parent checkpoint, Calibration/Validation manifests, seed, registry, search-space digest, and budget limits.
- Validation records include search depth, candidate counts, full/fast/interaction costs, cache namespace, profiling cost, Pareto metrics, and the predeclared comparison decision.

- [ ] Add a CPU Fake test showing parent and child each execute `TICAPruner.search()` and produce independent candidate/Pareto artifacts.
- [ ] Replace the TICA algorithm-validation callback that measures only one Sandbox plan with a complete-search callback.
- [ ] Add a controlled child-improves test and a full-risk-over-limit child-rejected test.
- [ ] Keep model PromotionPolicy independent from algorithm validation status and effective-parent selection.
- [ ] Record shared profiling separately from per-algorithm search cost.

### Task 4: Make the real Controller protocol mode-specific

**Files:**
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/evolution/loop.py`
- Test: `tests/unit/test_algorithm_agent.py`
- Test: `tests/integration/test_controller_tica_protocol.py`

**Interfaces:**
- `build_algorithm_provider(..., protocol_mode="standard"|"tica")` emits a bounded mode-specific prompt and schema.
- TICA responses contain a search-rule proposal and optional initial suggestion; they cannot directly select an unvalidated final checkpoint operation.
- Standard mode preserves the existing operator schema.

- [ ] Add a Mock HTTP server test that captures the request and returns a TICA rule.
- [ ] Verify the returned rule changes search behavior in a CPU Fake search.
- [ ] Verify standard mode does not advertise or accept `tica_prune` fields.
- [ ] Classify Agent changes as parameter/rule/search-control evolution unless a SearchSpaceManager proposal is required.

### Task 5: Audit search depth, budgets, recovery, and real Wan preflight

**Files:**
- Modify: `harness4h3/evolution/tica_prune.py`
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/adapters/wan/diffusers.py`
- Modify: `harness4h3/evolution/tica_calibration.py`
- Create or modify: `tools/preflight_wan_tica.py`
- Test: `tests/unit/test_tica_prune_v2.py`
- Test: `tests/unit/test_tica_calibration.py`
- Test: `tests/unit/test_wan_tica_preflight.py`

**Interfaces:**
- Search artifacts expose configured and observed depth, per-level candidate/feasible counts, budget exhaustion reason, profiling cost, total elapsed time, and actual model executions.
- Resume reuses completed state/cache entries without charging them again and rejects stale recipe/algorithm/source/split digests.
- Wan preflight inspects an already available checkpoint/adapter, validates FFN components, hook width, global channel alignment, checkpoint identity, and cached source hashes without generating videos or mutating the model.

- [ ] Add depth-3/insufficient-budget tests with explicit observed-depth assertions.
- [ ] Separate baseline-suite accounting from main-search accounting and document shared versus independent budgets.
- [ ] Add resume tests for search state, candidate IDs, cache namespaces, and budget ledger.
- [ ] Add a no-side-effect preflight command and Fake adapter contract tests for real-Wan limitations.
- [ ] Ensure missing FFN activation evidence cannot fall back to first-N channels.

### Task 6: Run acceptance gates, document evidence, commit, and push

**Files:**
- Modify: `docs/evo-compress-next-stage.md`
- Modify: `docs/tica-wan-v2-runbook.md`
- Modify: `docs/superpowers/reports/2026-10-15-rsi-acceptance.md`

- [ ] Run focused TICA, protocol, controller, algorithm-validation, and historical-promotion tests.
- [ ] Run CPU-compatible repository tests and record optional-dependency collection failures exactly.
- [ ] Run compile and diff checks; do not start GPU Wan.
- [ ] Inspect changed-file scope, commit only task files, and push `codex/tica-prune-v2`.
- [ ] Report which requirements are code-proven, CPU-Fake-proven, preflight-only, or still require a real GPU experiment.
