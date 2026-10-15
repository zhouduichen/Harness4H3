# RSI Credible Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed algorithm proxy scores with reproducible model-level Fake measurements and make algorithm/search-space evolution evidence-driven and auditable.

**Architecture:** Keep `EvolutionLoop` and `AlgorithmValidator` as orchestration boundaries. Add one focused model-measurement helper that executes trusted compression candidates from the immutable parent checkpoint and records structure/output/cost evidence. Feed persisted round evidence into the existing Agent and SearchSpaceManager only after explicit thresholds, and classify changes through a small expression/lineage record without changing PromotionPolicy.

**Tech Stack:** Python, PyTorch FakeWanDiT, existing Wan adapter/Registry, JSON artifacts, pytest.

## Global Constraints

- Existing Model Candidate, Algorithm Candidate, SearchSpaceVersion, ExperienceStore, SkillHub, Sandbox, Evaluator, and PromotionPolicy interfaces remain the primary interfaces.
- New behavior is disabled unless the existing algorithm improvement/search-space evolution flags are enabled.
- CPU Fake evidence is never presented as real Wan video quality, GPU latency, or peak memory evidence.
- Current four Wan experiments, checkpoints, configs, evaluation rules, and historical results are read-only.
- Every generated candidate remains restricted by Sandbox and trusted operator validation.

---

### Task 1: Add reproducible model-level algorithm measurements

**Files:**
- Create: `harness4h3/evolution/algorithm_evaluation.py`
- Modify: `harness4h3/evolution/improvement.py`
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/adapters/wan/contracts.py`
- Modify: `harness4h3/adapters/wan/fake.py`
- Test: `tests/unit/test_algorithm_improvement.py`

**Interfaces:**
- Produce `AlgorithmModelEvaluator.evaluate(result: SandboxResult, profile: Mapping[str, Any]) -> Mapping[str, Any]`.
- The result must include `evaluation_status`, `evidence_level`, `parent_checkpoint_sha256`, `child_checkpoint_sha256`, `plan_digest`, `structural_change`, `parameter_count`, `parameter_reduction_ratio`, `estimated_flops`, `output_metrics`, `hardware_metrics`, `invocations`, `elapsed_s`, and `valid`.
- `AlgorithmValidator` continues to call a one-argument callback for backward compatibility; the loop supplies a scenario-bound evaluator closure.

- [ ] **Step 1: Write tests for equal-plan invariance and different-plan measurements.**
- [ ] **Step 2: Implement deterministic Fake probe support through the adapter boundary.**
- [ ] **Step 3: Implement candidate execution, reload, finite checks, output MSE/cosine, structure and cost metrics.**
- [ ] **Step 4: Replace `_algorithm_proxy_evaluator` in the loop with the scenario-bound evaluator.**
- [ ] **Step 5: Extend validation records with measured evidence and unavailable hardware fields.**
- [ ] **Step 6: Run focused algorithm tests and inspect JSON evidence.**

### Task 2: Make search-space evolution feedback-driven

**Files:**
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/operators/wan/search_space.py`
- Test: `tests/integration/test_evolutionary_compression.py`
- Test: `tests/unit/test_search_space.py`

**Interfaces:**
- Produce `EvolutionLoop._search_space_feedback(history, pareto_front) -> Mapping[str, Any]`.
- Produce `EvolutionLoop._maybe_evolve_search_space(...) -> Optional[SearchSpaceProposal]`.
- Proposal artifacts must include trigger evidence, bottleneck type, and a threshold decision.

- [ ] **Step 1: Add failing tests proving initialization does not register a proposal.**
- [ ] **Step 2: Add deterministic bottleneck extraction from persisted rounds.**
- [ ] **Step 3: Gate Agent proposals on repeated evidence and pass bounded history/evidence IDs.**
- [ ] **Step 4: Validate/register only validated proposals and preserve rejected state.**
- [ ] **Step 5: Add provider and deterministic fallback tests for no-trigger/trigger cases.**

### Task 3: Classify parameter search, algorithm rules, and actual space extensions

**Files:**
- Create: `harness4h3/evolution/taxonomy.py`
- Modify: `harness4h3/evolution/design.py`
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/operators/wan/search_space.py`
- Test: `tests/unit/test_search_space.py`
- Test: `tests/unit/test_algorithm_improvement.py`

**Interfaces:**
- Produce `classify_algorithm_change(parent_source, child_source, parent_plan, child_plan, active_search_space) -> Mapping[str, Any]`.
- The result includes `change_class`, `reachable_by_existing_closure`, `counterexample`, `limitations`, and `evidence_level`.

- [ ] **Step 1: Add tests for argument-only, rule-only, and structural counterexample cases.**
- [ ] **Step 2: Implement conservative operator-closure comparison for current trusted operators.**
- [ ] **Step 3: Mark threshold/generated-rule operators as rule evolution unless structural evidence proves otherwise.**
- [ ] **Step 4: Persist taxonomy in design, validation, proposal, decision, and SkillHub evidence.**
- [ ] **Step 5: Reject unsupported structure changes without entering the trusted registry.**

### Task 4: Select effective algorithm parents from measured evidence

**Files:**
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/store.py`
- Modify: `harness4h3/evolution/skill_hub.py`
- Test: `tests/integration/test_evolutionary_compression.py`
- Test: `tests/unit/test_algorithm_improvement.py`

**Interfaces:**
- Produce `EvolutionLoop._algorithm_parent_candidates() -> Mapping[str, Sequence[AlgorithmCandidate]]`.
- Parent selection must prefer validated measured candidates, then historical best, while retaining latest/pending as separate metadata.

- [ ] **Step 1: Add a test where latest candidate is rejected but an earlier validated candidate remains effective.**
- [ ] **Step 2: Implement status/evidence-aware parent ranking without changing ModelStore promotion.**
- [ ] **Step 3: Ensure validation failure and model promotion do not overwrite algorithm validation status.**
- [ ] **Step 4: Persist latest/effective/best/pending IDs and reason codes.**

### Task 5: End-to-end CPU verification and documentation

**Files:**
- Modify: `docs/evolutionary-compression.md`
- Modify: `docs/evo-compress-next-stage.md`
- Modify: `tests/integration/test_evolutionary_compression.py`

- [ ] **Step 1: Run focused tests and a fresh multi-round CPU Fake output root.**
- [ ] **Step 2: Run the full test suite without touching Wan experiment directories.**
- [ ] **Step 3: Verify source diff, measured algorithm evidence, taxonomy, trigger evidence, and rejected fallback artifacts.**
- [ ] **Step 4: Commit and push only the new independent branch.**
