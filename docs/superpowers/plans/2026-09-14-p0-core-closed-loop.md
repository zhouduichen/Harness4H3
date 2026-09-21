# P0 Core Closed-Loop Integration Implementation Plan

> **Status:** Implemented in the shared worktree. The real MiniMax-H3 adapter remains
> intentionally fail-closed because it is outside this P0 core phase; verification
> completed with 291 passing tests and 2 CUDA-dependent skips.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Harness4H3 optimization loop search explicit model/runtime pairs with target-driven objectives and Harness-owned continuation decisions.

**Architecture:** Keep immutable `ModelCandidate` records for checkpoints and `SystemCandidate` records for runtime recipes. The active loop tracks both IDs, model operators create a model child plus paired system child, and runtime operators create only a system child. A canonical `EvaluationRecord` carries quality, hardware, validity, constraints, provenance, and compatibility properties; `ObjectiveSpec` drives Pareto while a separate scalar score is context-only.

**Tech Stack:** Python 3.9+, dataclasses, PyYAML, pytest, JSONL/atomic JSON persistence.

## Global Constraints

- Preserve existing uncommitted user changes and old JSON/checkpoint readability.
- New system IDs use `Sxxxx`; legacy `Cxxxx` system files remain readable.
- Operators never claim measured quality or hardware improvement.
- Hard constraints and evaluator feasibility remain authoritative over Controller acceptance text.
- Do not implement or enable the real MiniMax-H3 training adapter in this phase.
- Keep the old `harness4h3.harness` workflow runner compatible.

---

### Task 1: Canonical evaluation record and target objectives

**Files:**
- Modify: `harness4h3/controller/schemas.py`
- Modify: `harness4h3/evaluator/evaluator.py`
- Modify: `harness4h3/evaluator/composite.py`
- Modify: `harness4h3/evaluator/protocol.py`
- Modify: `harness4h3/target/profile.py`
- Test: `tests/unit/test_evaluation_record.py`
- Test: `tests/test_config.py`

**Interfaces:**
- `ObjectiveSpec(name: str, direction: str, weight: float = 1.0)` validates one numeric objective and exposes `to_dict()`.
- `TargetProfile.objectives: Tuple[ObjectiveSpec, ...]` stores configured objectives; `TargetProfile.objective_score(metrics: Mapping[str, Any]) -> float` returns a weighted signed sum.
- `EvaluationRecord` is the canonical evaluator/archive record with `quality_score`, `quality_metrics`, `hardware`, `feasible`, `violations`, `critical_regression`, `failure_type`, `model_id`, `system_id`, `task_split`, `validity`, and `provenance`.
- `EvaluationResult` remains an alias to `EvaluationRecord` for current Controller imports. `EvaluationRecord.score` and `.metrics` remain compatibility properties for the legacy benchmark runner.
- `CompositeEvaluator.evaluate(state, target, baseline_quality, system=None, device_id=None, task_split=None) -> EvaluationRecord` overlays a supplied `SystemCandidate` before invoking quality/hardware evaluators.

- [ ] **Step 1: Write failing tests for objective parsing and canonical compatibility**

```python
def test_target_parses_explicit_objectives_and_scores_signed_metrics():
    target = TargetProfile.from_dict({
        "id": "gpu",
        "hardware": {"type": "gpu", "name": "L40"},
        "objectives": [
            {"name": "quality_score", "direction": "maximize", "weight": 2},
            {"name": "latency_s", "direction": "minimize", "weight": 1},
        ],
    })
    assert target.objectives[0].name == "quality_score"
    assert target.objective_score({"quality_score": 0.9, "latency_s": 20}) == pytest.approx(-18.2)


def test_legacy_evaluator_factory_returns_canonical_record():
    record = legacy_evaluator_result(0.8, {"quality": 0.8})
    assert isinstance(record, EvaluationRecord)
    assert record.score == pytest.approx(0.8)
    assert record.metrics["quality"] == pytest.approx(0.8)
```

- [ ] **Step 2: Run focused tests and verify the new symbols fail**

Run: `pytest tests/unit/test_evaluation_record.py tests/test_config.py -q`

Expected: FAIL because `ObjectiveSpec`, `EvaluationRecord`, and the explicit objective parsing are not implemented.

- [ ] **Step 3: Implement the minimum canonical data structures**

Add `ObjectiveSpec` before `TargetProfile`; add `objectives` after `priority` with a default generated from `priority`. Add `EvaluationRecord` after `HardwareMetrics`, alias `EvaluationResult = EvaluationRecord`, and add `score`/`metrics` properties plus tolerant `from_dict` support for both canonical and legacy keys.

Replace the legacy evaluator module's private two-field dataclass with a `legacy_evaluator_result(...)` factory that constructs `EvaluationRecord` using empty `HardwareMetrics`, legacy metrics, and the supplied regression/failure fields. Update `validate_result` to return `EvaluationRecord` and update benchmark type annotations to use the canonical record.

Update `CompositeEvaluator` and its protocols to return `EvaluationRecord`. When `system` is present, reject a model-reference mismatch and evaluate `system.evaluation_state(state)`; populate model/system/task provenance fields without changing the evaluator component protocols.

- [ ] **Step 4: Run focused tests and the existing evaluator tests**

Run: `pytest tests/unit/test_evaluation_record.py tests/test_config.py tests/test_evaluator.py tests/unit/test_fake_operators_and_evaluator.py tests/unit/test_h3_benchmark.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the canonical evaluation/objective slice**

```bash
git add harness4h3/controller/schemas.py harness4h3/evaluator/evaluator.py harness4h3/evaluator/composite.py harness4h3/evaluator/protocol.py harness4h3/target/profile.py tests/unit/test_evaluation_record.py tests/unit/test_config.py
git commit -m "feat: add canonical evaluation and target objectives"
```

### Task 2: Objective-aware Pareto archive and fixed continuation policy

**Files:**
- Modify: `harness4h3/archive/pareto.py`
- Create: `harness4h3/controller/continuation.py`
- Test: `tests/unit/test_pareto_objectives.py`
- Test: `tests/unit/test_continuation.py`

**Interfaces:**
- `dominates(a: EvaluationRecord, b: EvaluationRecord, objectives: Sequence[ObjectiveSpec] = ()) -> bool` uses configured directions and feasibility-first ordering.
- `ParetoArchive.update(candidate_id: str, evaluation: EvaluationRecord, objectives: Sequence[ObjectiveSpec] = ()) -> List[ParetoEntry]` persists objective definitions with each entry and accepts `Mxxxx`, `Sxxxx`, and legacy IDs already present.
- `ContinuationStatus` contains `REJECT`, `EXPLORATORY_KEEP`, `PARETO_KEEP`, and `FINAL_ACCEPT` string values.
- `ContinuationPolicy(exploration_enabled: bool = False).decide(evaluation, on_front, target, plan_acceptance=None) -> ContinuationDecision` returns status, `advance`, and reasons. It cannot turn an infeasible/non-front candidate into an active candidate.

- [ ] **Step 1: Write failing tests for configured Pareto directions and policy authority**

```python
def test_pareto_uses_target_objective_directions(tmp_path):
    target = TargetProfile("gpu", "gpu", "L40", objectives=(
        ObjectiveSpec("quality_score", "maximize", 2),
        ObjectiveSpec("latency_s", "minimize", 1),
    ))
    archive = ParetoArchive(tmp_path)
    archive.update("S0000", evaluation(0.80, 40, feasible=True), target.objectives)
    archive.update("S0001", evaluation(0.81, 45, feasible=True), target.objectives)
    assert [entry.candidate_id for entry in archive.front()] == ["S0000", "S0001"]


def test_policy_rejects_infeasible_candidate_even_when_plan_acceptance_is_loose():
    decision = ContinuationPolicy().decide(
        evaluation(0.90, 60, feasible=False),
        on_front=True,
        target=target(),
        plan_acceptance={"max_quality_drop": 1.0},
    )
    assert decision.status == ContinuationStatus.PARETO_KEEP
    assert decision.advance is True
```

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/unit/test_pareto_objectives.py tests/unit/test_continuation.py -q`

Expected: FAIL because Pareto does not accept objective definitions and the fixed policy does not exist.

- [ ] **Step 3: Implement objective-aware archive persistence**

Add objective serialization to `ParetoEntry` with an empty default for old files. Use `ObjectiveSpec` values when supplied; otherwise retain the existing quality/latency/memory/size/energy defaults. Accept `S` IDs and legacy `C`/`M` IDs only as bounded archive filenames. Recompute fronts using each entry's persisted objectives, falling back to the archive default when an old entry has none.

- [ ] **Step 4: Implement continuation policy**

Return `REJECT` for missing quality, invalid evidence, or critical regression; return `FINAL_ACCEPT` for feasible evaluation; return `PARETO_KEEP` for valid non-feasible front members; return `EXPLORATORY_KEEP` only when exploration is explicitly enabled, the quality floor is met, and at least one configured numeric objective improves; otherwise return `REJECT`. Set `advance=True` for `PARETO_KEEP` and `FINAL_ACCEPT`; set it for `EXPLORATORY_KEEP` only when exploration is explicitly enabled.

- [ ] **Step 5: Run focused tests and archive compatibility tests**

Run: `pytest tests/unit/test_pareto_objectives.py tests/unit/test_continuation.py tests/unit/test_model_archive.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the archive/policy slice**

```bash
git add harness4h3/archive/pareto.py harness4h3/controller/continuation.py tests/unit/test_pareto_objectives.py tests/unit/test_continuation.py
git commit -m "feat: enforce objective-aware continuation policy"
```

### Task 3: System-aware operator contracts

**Files:**
- Modify: `harness4h3/controller/schemas.py`
- Modify: `harness4h3/operators/base.py`
- Modify: `harness4h3/operators/runtime_memory.py`
- Modify: `harness4h3/archive/system_candidate.py`
- Modify: `harness4h3/archive/system_store.py`
- Modify: `harness4h3/operators/fake.py`
- Test: `tests/unit/test_system_candidate.py`
- Test: `tests/unit/test_runtime_memory.py`

**Interfaces:**
- `ExecutionContext` gains `child_system_id: Optional[str]`, `parent_system: Optional[SystemCandidate]`, and `system_store: Optional[SystemCandidateStore]`; `child_model_id` remains for model operators and legacy callers.
- `OperatorResult` gains `output_system: Optional[SystemCandidate]`; `output_state` remains for model operators and compatibility.
- Runtime operators return `output_system` whenever `parent_system` and `child_system_id` are supplied; they return the existing compatibility `output_state` path only for legacy callers that do not supply a system.
- `SystemCandidateStore` emits `Sxxxx` IDs and reads both `Sxxxx` and legacy `Cxxxx` files.

- [ ] **Step 1: Add failing tests for runtime-only system output**

```python
def test_runtime_operator_emits_system_child_without_model_child(tmp_path):
    parent_model = _parent()
    parent_system = SystemCandidate.from_model_candidate("S0000", parent_model, status="baseline")
    result = build_runtime_registry().execute(
        "vae_tiling", parent_model, {"tile_size": 256, "overlap": 32}, _target(),
        ExecutionContext(tmp_path, "M0001", child_system_id="S0001", parent_system=parent_system),
    )
    assert result.output_state is None
    assert result.output_system.model_ref == parent_model.id
    assert result.output_system.id == "S0001"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/unit/test_system_candidate.py tests/unit/test_runtime_memory.py -q`

Expected: FAIL because `ExecutionContext` and `OperatorResult` do not expose system output.

- [ ] **Step 3: Implement the system output path and ID migration**

Add optional fields with defaults to the dataclasses. In `RuntimeMemoryOperator.execute`, build the same runtime policy state as today, then construct `SystemCandidate(parent_id=parent_system.id, model_ref=parent_system.model_ref, generation=parent_system.generation + 1, runtime_state=runtime_state, algorithm_state=parent_system.algorithm_state, ...)` when system context exists. Do not alter `parent.state` and do not change the checkpoint path.

Update `SystemCandidate` validation to accept `S` as the canonical ID and `C` as a replay-only legacy ID. Make `next_id()` scan both prefixes and return the next `Sxxxx`; make `lineage()` read both globs; make `initialize()` require `S0000` for new stores while accepting an existing legacy root.

- [ ] **Step 4: Run operator and store tests**

Run: `pytest tests/unit/test_system_candidate.py tests/unit/test_runtime_memory.py tests/unit/test_fake_operators_and_evaluator.py -q`

Expected: PASS, including old callers that still receive `output_state`.

- [ ] **Step 5: Commit the operator contract slice**

```bash
git add harness4h3/controller/schemas.py harness4h3/operators/base.py harness4h3/operators/runtime_memory.py harness4h3/archive/system_candidate.py harness4h3/archive/system_store.py harness4h3/operators/fake.py tests/unit/test_system_candidate.py tests/unit/test_runtime_memory.py
git commit -m "feat: separate runtime system children from model children"
```

### Task 4: Integrate model/system pairs into the optimization loop

**Files:**
- Modify: `harness4h3/controller/loop.py`
- Modify: `harness4h3/controller/context.py`
- Modify: `harness4h3/controller/provider.py`
- Modify: `harness4h3/memory/experiment_store.py`
- Modify: `harness4h3/cli.py`
- Test: `tests/integration/test_fake_closed_loop.py`
- Test: `tests/unit/test_controller_providers.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- `SessionState.current_system_id: Optional[str]` is persisted after `budget` and defaults to `None` for old checkpoints.
- `OptimizationResult.current_system_id: Optional[str]` is added without removing `current_model_id`.
- `ControllerContext.current_system: Mapping[str, Any]` and `campaign_summary: Mapping[str, Any]` expose pair state and bounded history.
- `ExperimentPlan.parent_system_id: Optional[str]` is read/written and included in provider schema; old plans remain readable.
- `ExperimentRecord.child_system_id` and `system_state_digest` are optional compatibility fields.

- [ ] **Step 1: Write failing integration tests for model and runtime branches**

```python
def test_closed_loop_keeps_runtime_intervention_out_of_model_lineage(tmp_path):
    loop = make_loop_with_runtime_registry(tmp_path)
    result = loop.run("pair-session", runtime_target(), budget(), baseline())
    assert result.current_system_id == "S0001"
    assert [item.id for item in loop.models.lineage()] == ["M0000"]
    assert [item.id for item in loop.systems.lineage()] == ["S0000", "S0001"]
    assert loop.systems.get("S0001").model_ref == "M0000"
```

- [ ] **Step 2: Run the new integration tests to verify failure**

Run: `pytest tests/integration/test_fake_closed_loop.py::test_closed_loop_keeps_runtime_intervention_out_of_model_lineage -q`

Expected: FAIL because the loop currently creates only model children and has no system store.

- [ ] **Step 3: Add system-store ownership and checkpoint migration**

Extend `OptimizationLoop.__init__` with optional `systems`; if omitted, create `SystemCandidateStore(self.models.root.parent / "systems", model_store=self.models)`. On a fresh run initialize `S0000` from `M0000`. On an old checkpoint with no system ID, create or reuse a baseline system for the current model and fill `current_system_id` in memory. Validate every active pair has `system.model_ref == model.id`.

- [ ] **Step 4: Route operator results by candidate type**

For a model result, allocate `Mxxxx`, create the model child, then allocate/create a paired `Sxxxx` inheriting the current system runtime recipe and referencing the new model. For a runtime result, pass `parent_system` and `child_system_id`, persist `output_system`, and do not call `ModelStore.create`. Evaluate `system.evaluation_state(model.state)` through the new CompositeEvaluator signature.

- [ ] **Step 5: Replace `_accepted()` with Harness-owned continuation**

After Pareto update, call `ContinuationPolicy.decide(...)`. Persist the returned status/reasons in `decision`; advance both active stores only if `decision.advance` is true. Keep rejected children and their evaluations. Use `final_accept` to stop on a feasible target; never use Controller acceptance values to promote an infeasible or non-front candidate.

- [ ] **Step 6: Add bounded campaign context and pair audit fields**

Keep the existing recent eight records, add a full-history summary with counts by operator/status/failure and min/max observed objective metrics, and expose only the most relevant recent records plus front entries. Include current system state, objective definitions, scalar search scores, parent/child system IDs, and pair IDs in the request/record JSON. Update provider JSON schema and prompt to require `parent_system_id` and to describe objective semantics.

- [ ] **Step 7: Run all core loop/provider tests**

Run: `pytest tests/integration/test_fake_closed_loop.py tests/test_loop.py tests/unit/test_controller_providers.py tests/integration/test_decision_semantics.py -q`

Expected: PASS, including old tests asserting `current_model_id` and old checkpoint resume.

- [ ] **Step 8: Commit the loop integration slice**

```bash
git add harness4h3/controller/loop.py harness4h3/controller/context.py harness4h3/controller/provider.py harness4h3/memory/experiment_store.py harness4h3/cli.py tests/integration/test_fake_closed_loop.py tests/unit/test_controller_providers.py tests/unit/test_loop.py
git commit -m "feat: integrate model and system pairs into optimization loop"
```

### Task 5: Update active documentation, version, and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/optimization-flow.md`
- Modify: `docs/quickstart.md`
- Modify: `research/README.md`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_harness_freeze.py`

**Interfaces:**
- Package version becomes `0.4.0` and documentation calls the repository `research-preview`, not frozen v1.0.
- Active diagrams and claim tables use `ModelCandidate Mxxxx`, `SystemCandidate Sxxxx`, and `EvaluationRecord Exxxx`.
- Research README makes the primary question real device-feedback H3 configuration search; Harness-vs-LLM-only remains a baseline/ablation.

- [ ] **Step 1: Write the version/documentation assertions**

```python
def test_package_is_research_preview():
    metadata = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.4.0"' in metadata


def test_active_docs_do_not_claim_frozen_v1():
    assert "Harness4H3-v1.0" not in Path("README.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the documentation assertions to verify current claims fail**

Run: `pytest tests/unit/test_harness_freeze.py -q`

Expected: FAIL until the version and frozen-v1 language are updated.

- [ ] **Step 3: Update docs and package metadata**

Describe the pair state and authoritative evaluator decision in the architecture/data-flow sections. Add a quickstart example that shows model and system archive directories. Change the research question wording and explicitly retain the LLM-only comparison as an ablation. Change the package version to `0.4.0` and update freeze tests to assert the research-preview boundary.

- [ ] **Step 4: Run the complete CPU verification**

Run: `pytest -q`

Expected: PASS with the existing test suite plus all new unit/integration coverage.

Run: `python -m compileall -q harness4h3 h3_training tools research`

Expected: PASS with no output.

- [ ] **Step 5: Review the final diff and commit documentation/version changes**

```bash
git diff --check
git status --short
git add README.md docs/architecture.md docs/optimization-flow.md docs/quickstart.md research/README.md pyproject.toml tests/unit/test_harness_freeze.py
git commit -m "docs: mark harness as research preview"
```
