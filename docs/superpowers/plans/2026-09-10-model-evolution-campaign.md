# Autonomous Model Evolution Campaign A0 Implementation Plan

> **For agentic workers:** This plan is executed inline in the current task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an execution-grounded A0 campaign that lets the Controller select model-level transformations, evaluates candidates at multiple fidelities, and records an accepted model lineage without changing the frozen Harness protocol.

**Architecture:** Add a model-evolution operator module with deterministic offline implementations for `create_student`, structured pruning, distillation, recovery fine-tuning, step distillation, and quantization. Add a separate A0 campaign entry point that reuses `ExperimentPlan`, `OperatorRegistry`, `ModelStore`, `ParetoArchive`, and `CompositeEvaluator`, but owns GPU-hour budget accounting, tier promotion, novelty checks, parent selection, and research reporting. Real training remains an external-operator integration point and is never represented as completed by the offline backend.

**Tech Stack:** Python 3.9+, dataclasses, existing Harness4H3 model/archive/evaluator interfaces, pytest, JSONL/JSON evidence.

## Global Constraints

- Harness version remains exactly `Harness4H3-v1.0` with status `frozen` and change policy `bugfix_only`.
- The campaign exposes model-level operators: `create_student`, `prune_blocks`, `prune_heads`, `prune_channels`, `distill`, `step_distill`, `recovery_finetune`, and `quantize`.
- A0 defaults to `max_gpu_hours=24`, `max_experiments=20`, `max_concurrent_experiments=1`, and `human_intervention_count=0`.
- Tier 0 is static validation, Tier 1 is cheap deterministic screening, Tier 2 is short recovery/distillation, and Tier 3 is full/held-out evaluation.
- Rejected candidates consume experiment budget but never become the next parent; execution failures consume both experiment and failure budget.
- A candidate fingerprint is `operator + canonical operator args + parent state digest + target profile id`; identical fingerprints are never executed twice.
- The offline backend is for protocol and lineage verification only; it must identify itself as offline and cannot claim real device or GPU training evidence.

---

### Task 1: Add model-evolution operators

**Files:**
- Create: `harness4h3/operators/model_evolution.py`
- Modify: `harness4h3/operators/__init__.py`
- Test: `tests/unit/test_model_evolution_operators.py`

**Interfaces:**
- Produce `ModelEvolutionBackend`, `ModelEvolutionOperator`, `build_model_evolution_registry(backend=None)`, and operators with the exact names listed above.
- Each operator implements the existing `Operator` protocol and returns an immutable child `ModelState` plus an explicit `CostEstimate`.
- The backend applies deterministic metric/architecture deltas in offline mode and stores provenance describing `offline_simulation=True`.

- [x] Write tests for operator visibility, parent immutability, student creation, pruning, training operators, quantization, validation, and cost tiers.
- [x] Implement the registry and deterministic backend using `ModelState.derive()`; never mutate parent mappings.
- [x] Run the focused operator tests through the project's available test runner.

### Task 2: Implement A0 campaign orchestration

**Files:**
- Create: `experiments/a0_model_evolution.py`
- Create: `tests/unit/test_a0_model_evolution.py`

**Interfaces:**
- Produce `A0Budget`, `A0Campaign`, `A0CampaignResult`, `run_campaign(...)`, and `build_a0_report(...)`.
- `run_campaign` accepts a `TargetProfile`, a `ControllerProvider`, an operator registry, an evaluator, and output paths; it returns a JSON-serializable result.
- Every iteration record includes parent state, plan, tier, operator result, evaluation, outcome, fingerprint, cost, and next-decision context.

- [x] Write tests proving rejected candidates do not replace the active parent, failures consume failure budget, duplicate fingerprints are blocked, tier promotion is recorded, and accepted children extend the model lineage.
- [x] Implement static Tier 0 checks, deterministic Tier 1/Tier 2 evaluation, Tier 3 promotion, GPU-hour accounting, and bounded stop conditions.
- [x] Persist `campaign.json`, `report.json`, model archive, Pareto archive, and append-only trajectory evidence.
- [x] Run the focused A0 tests.

### Task 3: Add CLI entry point and research documentation

**Files:**
- Modify: `harness4h3/cli.py`
- Modify: `README.md`
- Create: `docs/evogen-a0-model-evolution.md`
- Test: `tests/integration/test_a0_cli.py`

**Interfaces:**
- Add `harness4h3 a0-evolve` with `--controller mock|ollama`, `--target`, `--output`, `--report-output`, `--max-gpu-hours`, and `--max-experiments`.
- The command defaults to offline mock execution and exits nonzero only when the campaign is bounded without satisfying the target or encounters a fatal setup error.
- Documentation distinguishes simulated lineage from real training and shows how an external training operator can be configured later.

- [x] Add CLI parsing and a JSON summary output.
- [x] Add an integration test that runs the mock campaign without network/GPU access.
- [x] Document the campaign state machine, fidelity tiers, budget semantics, and evidence interpretation.

### Task 4: Regression and preflight

**Files:**
- Modify: none unless test fixes are required.

- [x] Run focused operator, campaign, and CLI tests.
- [x] Run the full test suite (`120 passed`).
- [x] Run `PYTHONPATH=. .venv/bin/python -m compileall -q harness4h3 experiments tests`.
- [x] Verify no changes under frozen Controller, evaluator, archive, memory, target, or schema modules beyond approved integration wiring.
- [x] Run an offline A0 preflight and inspect that the report marks `offline_simulation=true`, `human_intervention_count=0`, and records a real `M0000 -> M0001 ...` lineage.
