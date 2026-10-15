# Algorithm Discovery and Search-Space Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Harness4H3 so Agent-authored source code controls real compression candidate generation and ranking, while validated runtime operators become executable capabilities in later searches.

**Architecture:** Preserve the existing `AlgorithmSandbox`, `TICAPruner`, `EvolutionLoop`, `SearchSpaceManager`, Adapter, Evaluator, Memory, and Store boundaries. Add a generic sandbox entrypoint returning JSON, an optional `search(profile, history, state, search_space)` Algorithm Program path in TICA, and a trusted `timestep_block_gate` operator whose rules persist through Fake and Diffusers Wan checkpoints. Keep `propose` and the existing fixed TICA path as compatibility fallbacks.

**Tech Stack:** Python 3.10+, dataclasses, JSON checkpoint metadata, PyTorch when available, Diffusers Wan adapter, pytest, existing subprocess Sandbox.

## Global Constraints

- Reuse the existing Sandbox; do not create a second isolation system.
- `propose(profile, history)` remains valid for existing algorithms and saved rounds.
- Algorithm Evolution compares parent and child from the same immutable checkpoint, split manifests, seed, SearchSpaceVersion, and search budget.
- `timestep_block_gate` must report zero static parameter reduction and separate dynamic execution metrics.
- Test data remains unreachable from Evolution and algorithm validation.
- Fake CPU evidence is marked `offline_simulation=true`; no GPU or real Wan performance result is fabricated.
- Every changed behavior gets a focused test before the implementation is committed.

---

### Task 1: Add a reusable Algorithm Program execution entrypoint

**Files:**
- Modify: `harness4h3/sandbox/policy.py`
- Modify: `harness4h3/sandbox/runner.py`
- Modify: `harness4h3/sandbox/__init__.py`
- Create: `tests/unit/test_algorithm_program.py`
- Modify: `tests/unit/test_algorithm_sandbox.py`

**Interfaces:**
- `validate_source(source: str, required_entrypoint: str = "propose") -> ValidatedSource` accepts either the legacy entrypoint or a requested safe entrypoint and still rejects sources with neither.
- `ProgramResult(status: str, source_sha256: str, value: Optional[Any], failure_type: Optional[str], message: str, stdout_path: Optional[str], stderr_path: Optional[str], metadata: Mapping[str, Any])` stores raw JSON output from one isolated invocation.
- `AlgorithmSandbox.run_program(source: str, *, entrypoint: str, payload: Mapping[str, Any]) -> ProgramResult` executes the requested source function with keyword arguments from `payload`.
- `AlgorithmSandbox.run(source, profile, history) -> SandboxResult` remains the compatibility wrapper and converts the raw `propose` value through `CompressionPlan.from_mapping`.

- [ ] **Step 1: Write the failing tests**

```python
def test_program_entrypoint_receives_search_state_and_returns_raw_json(tmp_path):
    source = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}

def search(profile, history, state, search_space):
    return {
        'candidates': [
            {'operator': 'prune_blocks', 'args': {'block_ids': [state['block_id']]}}
        ],
        'rank': [0],
        'stop': True,
    }
"""
    result = AlgorithmSandbox(tmp_path).run_program(
        source,
        entrypoint="search",
        payload={"profile": {}, "history": [], "state": {"block_id": "block.2"}, "search_space": {"operators": []}},
    )
    assert result.status == "success"
    assert result.value["candidates"][0]["args"]["block_ids"] == ["block.2"]
    assert result.metadata["entrypoint"] == "search"


def test_program_entrypoint_rejects_source_without_requested_function(tmp_path):
    source = "def propose(profile, history):\n    return {}\n"
    result = AlgorithmSandbox(tmp_path).run_program(source, entrypoint="search", payload={})
    assert result.status == "failed"
    assert result.failure_type == "sandbox_policy"
    assert "search" in result.message
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/unit/test_algorithm_program.py tests/unit/test_algorithm_sandbox.py`

Expected: the new tests fail because `run_program` and the requested-entrypoint policy do not exist; the existing sandbox tests continue to identify the legacy behavior that must remain green.

- [ ] **Step 3: Implement the raw program boundary**

Update `validate_source` so it checks the requested function name instead of hard-coding `propose`, while the default remains `propose`. Add `ProgramResult` and make `AlgorithmSandbox.run_program` write one payload JSON file, invoke the existing isolated runner with the entrypoint name, parse bounded JSON, and attach the same process/audit/resource metadata. Extend `RUNNER_SOURCE` to load the payload, call `namespace[entrypoint](**payload)`, and write only JSON-serializable output. Keep all trusted file I/O outside `_ALGORITHM_PHASE` and keep the existing `run` conversion and operator schema validation unchanged.

The runner invocation must remain equivalent to:

```python
result = self.executor.execute(
    (sys.executable, "-I", str(runner_path), str(source_path), str(payload_path), str(output_path), str(self.timeout_s), str(entrypoint)),
    run_root,
)
```

The runner must reject an entrypoint not present in the validated source and return `algorithm_output_invalid` for missing, oversized, or non-JSON output.

- [ ] **Step 4: Run focused tests and compatibility tests**

Run: `pytest -q tests/unit/test_algorithm_program.py tests/unit/test_algorithm_sandbox.py`

Expected: all new and existing sandbox tests pass, including filesystem/import/dunder/resource-limit rejection.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/sandbox/policy.py harness4h3/sandbox/runner.py harness4h3/sandbox/__init__.py tests/unit/test_algorithm_program.py tests/unit/test_algorithm_sandbox.py
git commit -m "feat: add sandboxed algorithm program entrypoints"
```

### Task 2: Execute Algorithm Program logic inside the actual bounded search

**Files:**
- Modify: `harness4h3/evolution/tica_prune.py`
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/evolution/design.py`
- Create: `tests/unit/test_algorithm_program_search.py`
- Modify: `tests/unit/test_tica_prune_v2.py`
- Modify: `tests/integration/test_tica_recursive_loop.py`

**Interfaces:**
- `AlgorithmSearchDecision` normalizes raw Algorithm Program output to `plans: Sequence[Mapping[str, Any]]`, `rank: Sequence[int]`, `stop: bool`, and `source_sha256`.
- Add these optional keyword parameters to the existing `TICAPruner.search` signature: `algorithm_source: Optional[str] = None`, `algorithm_history: Sequence[Mapping[str, Any]] = ()`, `search_space: Optional[Mapping[str, Any]] = None`, and `sandbox: Optional[AlgorithmSandbox] = None`. The return type remains `TICASearchResult`; the program path is used only when the source defines `search`, and all omitted arguments preserve the existing TICA path.
- `TICASearchResult` gains `algorithm_trace: Sequence[Mapping[str, Any]]` and serializes each invocation’s depth, source digest, payload digest, generated plan digests, rank, and stop decision.
- `AlgorithmAgent` deterministic sources include both a legacy `propose` fallback and a `search` function so source changes can affect more than Recipe fields.

- [ ] **Step 1: Write failing tests proving source-controlled candidate differences**

```python
def test_two_algorithm_programs_change_candidate_order_and_membership(tmp_path, fake_parent, fake_manifest, fake_profile):
    source_a = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}
def search(profile, history, state, search_space):
    return {'candidates': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
    ], 'rank': [0, 1], 'stop': True}
"""
    source_b = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}
def search(profile, history, state, search_space):
    return {'candidates': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}},
    ], 'rank': [1, 0], 'stop': True}
"""
    first = run_program_search(source_a, tmp_path / "a", fake_parent, fake_manifest, fake_profile)
    second = run_program_search(source_b, tmp_path / "b", fake_parent, fake_manifest, fake_profile)
    assert first.algorithm_trace[0]["source_sha256"] != second.algorithm_trace[0]["source_sha256"]
    assert first.algorithm_trace[0]["plan_digests"] != second.algorithm_trace[0]["plan_digests"]


def test_search_program_reads_registered_operator_and_records_runtime_trace(tmp_path, fake_parent, fake_manifest, fake_profile):
    # Use a registered extension spec and assert it appears in the payload
    # captured by the evaluator and in the resulting algorithm_trace.
    trace = run_program_search_with_registered_operator(tmp_path, fake_parent, fake_manifest, fake_profile)
    assert trace[0]["search_space_operator_names"] == ["prune_blocks", "timestep_block_gate"]
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/unit/test_algorithm_program_search.py tests/unit/test_tica_prune_v2.py`

Expected: the new tests fail because `TICAPruner` always builds roots through `_root_states()` and does not invoke source-defined `search`.

- [ ] **Step 3: Add strict Algorithm Program normalization**

Implement `_run_algorithm_search_program` in `tica_prune.py`:

```python
def _run_algorithm_search_program(
    source: str,
    sandbox: AlgorithmSandbox,
    profile: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    search_space: Mapping[str, Any],
) -> AlgorithmSearchDecision:
    raw = sandbox.run_program(
        source,
        entrypoint="search",
        payload={"profile": profile, "history": list(history), "state": dict(state), "search_space": dict(search_space)},
    )
    if raw.status != "success" or not isinstance(raw.value, Mapping):
        raise ValueError(raw.message or "algorithm_search_program_failed")
    candidates = raw.value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("algorithm_search_output_invalid: candidates must be non-empty")
    rank = raw.value.get("rank", list(range(len(candidates))))
    if not isinstance(rank, list) or sorted(set(rank)) != sorted(set(range(len(candidates)))):
        raise ValueError("algorithm_search_output_invalid: rank must cover candidate indexes")
    return AlgorithmSearchDecision(
        plans=tuple(candidates),
        rank=tuple(int(item) for item in rank),
        stop=bool(raw.value.get("stop", False)),
        source_sha256=raw.source_sha256,
    )
```

Validate every candidate with `CompressionPlan.from_mapping(candidate, operator_specs=active_specs)`, reject duplicate plan digests, and keep the current `AlgorithmSandbox` as the only execution boundary. Include `search_space_operator_names` and a stable payload digest in each trace row.

- [ ] **Step 4: Integrate source-controlled generation, order, and stop conditions**

In `TICAPruner.search`, when `algorithm_source` is present and contains `search`, call the program once per search depth with a state containing current depth, measured candidates, frontier, budget snapshot, and parent checkpoint digest. Measure generated plans through the existing `evaluate_candidate` callback and `_risk_evidence` gate. Sort each depth’s generated candidates by the returned rank before selecting the next frontier. Honor `stop` after the current batch while still recording the measured batch. Preserve `_root_states`/`_expand_state` when no program source is supplied or when the source has no `search` function.

For program candidates, use the returned plan’s operator as `TICACandidate.kind`, keep the existing full-risk and Pareto checks, and use the source’s rank only for search order; the Harness still owns feasibility and final selection. Add algorithm trace data to `TICASearchResult.to_dict()`.

Pass the active SearchSpaceVersion and Agent source from `EvolutionLoop._run_tica_search` and the complete parent/child algorithm validation evaluator. Store trace and source digests in `tica-search.json`, `decision.json`, and `design.json`.

- [ ] **Step 5: Make deterministic Agent generation and improvement source-level**

Update `_seed_source` and `_deterministic_improvement_source` so generated source contains a profile-based `search` implementation that can change block scoring, candidate set, candidate rank, and stop behavior. Keep a simple `propose` wrapper for old calls. Improvement sources must read `history` and failure reasons, and their `source_diff` must include the changed search function, not only numeric Recipe fields. Provider responses remain unrestricted in logic but are validated by the same Sandbox before execution.

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/unit/test_algorithm_program_search.py tests/unit/test_tica_prune_v2.py tests/integration/test_tica_recursive_loop.py`

Expected: source-defined programs produce different traces; legacy TICA and recursive validation tests remain green.

```bash
git add harness4h3/evolution/tica_prune.py harness4h3/evolution/loop.py harness4h3/evolution/agent.py harness4h3/evolution/design.py tests/unit/test_algorithm_program_search.py tests/unit/test_tica_prune_v2.py tests/integration/test_tica_recursive_loop.py
git commit -m "feat: execute agent algorithms inside compression search"
```

### Task 3: Implement and persist the non-structural `timestep_block_gate` operator

**Files:**
- Modify: `harness4h3/adapters/wan/contracts.py`
- Modify: `harness4h3/adapters/wan/fake.py`
- Modify: `harness4h3/adapters/wan/diffusers.py`
- Modify: `harness4h3/operators/wan/compression.py`
- Create: `harness4h3/operators/wan/runtime_gating.py`
- Modify: `harness4h3/operators/wan/__init__.py`
- Create: `tests/unit/test_timestep_block_gate.py`
- Modify: `tests/training/test_wan_adapter.py`
- Modify: `tests/integration/test_real_wan_adapter_contract.py`

**Interfaces:**
- `TimestepGateRule.from_mapping(raw: Mapping[str, Any]) -> TimestepGateRule` validates block ID and normalized `[min_timestep, max_timestep]` bounds.
- `validate_timestep_gate_plan(plan: Mapping[str, Any], block_ids: Sequence[str]) -> Sequence[TimestepGateRule]` rejects unknown blocks, overlapping duplicate rules, invalid policy, and rules that never match.
- `WanModelAdapter.apply_timestep_block_gating(model, rules) -> StructuralChange` installs runtime behavior and returns rule metadata; adapters that cannot restore the behavior raise `ValueError`.
- `TimestepBlockGatingOperator.execute(adapter, parent, plan, output_dir) -> CompressionResult` writes a child checkpoint with `parameter_reduction_ratio=0.0` and `runtime_operator="timestep_block_gate"`.

- [ ] **Step 1: Write failing Fake and persistence tests**

```python
def test_timestep_gate_changes_only_selected_denoising_steps_and_reloads(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = TimestepBlockGatingOperator().execute(
        adapter,
        parent,
        {"rules": [{"block_id": "block.0", "min_timestep": 0.0, "max_timestep": 0.3}], "policy": "skip_when_in_range"},
        tmp_path / "child",
    )
    assert result.status == "success"
    assert result.structural_change["parameter_reduction_ratio"] == 0.0
    child = adapter.reload_model(result.child_checkpoint)
    high = adapter.forward_denoising(child, adapter.tica_calibration_inputs(child, fake_manifest())[0], 0.9)
    low = adapter.forward_denoising(child, adapter.tica_calibration_inputs(child, fake_manifest())[0], 0.1)
    assert child.runtime_gate_stats["block.0"]["skipped"] == 1
    assert not torch.equal(high, low)
    assert child.runtime_gate_rules == result.structural_change["rules"]
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest -q tests/unit/test_timestep_block_gate.py`

Expected: import or execution failure because the new operator and adapter boundary do not exist.

- [ ] **Step 3: Implement shared rule validation and Fake behavior**

Create `runtime_gating.py` with the dataclass and validation functions. Add `runtime_gate_rules` and `runtime_gate_stats` to `FakeWanDiT`; its `forward(value, timestep=None)` skips a rule-matching block, increments `executed`/`skipped`, and leaves weights unchanged. `FakeWanDiTAdapter.save_model` writes the rules and empty stats to `config.json`; `load_model` restores them before returning the model. `forward_denoising` passes the normalized timestep into the model.

- [ ] **Step 4: Implement Diffusers Wan runtime restoration**

Add `DiffusersWanAdapter.apply_timestep_block_gating`. Store the normalized rules with `transformer.register_to_config(harness4h3_timestep_block_gating=[rule.to_dict() for rule in rules])`. Wrap the transformer `forward` only at the runtime object boundary to retain the raw timestep in a private attribute, and wrap each recognized block’s `forward` so it returns its input unchanged when the rule matches. On `load_model`, read the custom config value and reinstall the wrappers; on unknown block signatures or missing config, fail closed. Do not alter the state-dict parameter keys. `save_model` records the config bytes in the checkpoint hash.

- [ ] **Step 5: Implement the compression operator and metrics**

Add `TimestepBlockGatingOperator` to the Wan registry factory path. It reloads the parent, validates rules against `adapter.get_blocks`, applies the rules, saves the child, reloads it once, and returns:

```python
{
    "kind": "runtime_timestep_gating",
    "runtime_operator": "timestep_block_gate",
    "rules": [{"block_id": "block.0", "min_timestep": 0.0, "max_timestep": 0.3}],
    "parameter_reduction_ratio": 0.0,
    "dynamic_compute": True,
    "child_reload_verified": True,
}
```

Do not report this operator as a static structural reduction.

- [ ] **Step 6: Run adapter and operator tests and commit**

Run: `pytest -q tests/unit/test_timestep_block_gate.py tests/training/test_wan_adapter.py tests/integration/test_real_wan_adapter_contract.py`

Expected: Fake persistence and dynamic execution pass; real adapter contract tests pass without requiring a Wan checkpoint, and missing real runtime capability fails explicitly.

```bash
git add harness4h3/adapters/wan/contracts.py harness4h3/adapters/wan/fake.py harness4h3/adapters/wan/diffusers.py harness4h3/operators/wan/compression.py harness4h3/operators/wan/runtime_gating.py harness4h3/operators/wan/__init__.py tests/unit/test_timestep_block_gate.py tests/training/test_wan_adapter.py tests/integration/test_real_wan_adapter_contract.py
git commit -m "feat: add persistent timestep block gating operator"
```

### Task 4: Connect Agent proposals to the live SearchSpace registry

**Files:**
- Modify: `harness4h3/operators/wan/search_space.py`
- Modify: `harness4h3/sandbox/runner.py`
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/evolution/taxonomy.py`
- Modify: `tests/unit/test_search_space.py`
- Modify: `tests/unit/test_algorithm_evolution.py`

**Interfaces:**
- `OperatorSpec.argument_schema` accepts nested JSON values through the explicit type name `list[object]`; no arbitrary Python objects or executable callbacks enter the registry.
- `_TRUSTED_FACTORIES["timestep_block_gate"]` returns a spec-bound `TimestepBlockGatingOperator`.
- `SearchSpaceManager.validate` executes `implementation_source` through `AlgorithmSandbox` with the proposal spec as a temporary extension, then executes the generated plan with the trusted factory and verifies reload.
- `SearchSpaceManager.registry_for(version)` includes every validated active operator with a callable trusted factory.

- [ ] **Step 1: Write failing proposal-to-search-space tests**

```python
def test_agent_gate_proposal_validates_registers_and_is_used_by_next_registry(tmp_path):
    adapter, parent = _parent(tmp_path)
    spec = OperatorSpec(
        "timestep_block_gate", "1",
        {"rules": "list[object]", "policy": "str"},
        "runtime",
        constraints=("normalized_timestep", "no_parameter_reduction"),
        limitations=("runtime_only", "requires_real_wan_replay_for_hardware_claim"),
        trusted_factory_key="timestep_block_gate",
        implementation_source="""def propose(profile, history):
    return {'operator': 'timestep_block_gate', 'args': {
        'rules': [{'block_id': 'block.0', 'min_timestep': 0.0, 'max_timestep': 0.3}],
        'policy': 'skip_when_in_range',
    }}
""",
    )
    manager = SearchSpaceManager(tmp_path / "space")
    proposal = manager.propose(spec, "block deletion cannot express timestep selectivity", ["R0001"])
    validated = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "validation")
    assert validated.status == "validated"
    version = manager.register(validated)
    registry = manager.registry_for(version, build_wan_compression_registry())
    assert "timestep_block_gate" in registry.names()
    assert registry.get("timestep_block_gate").name == "timestep_block_gate"
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/unit/test_search_space.py`

Expected: the type schema rejects `list[object]` and the trusted factory is unavailable.

- [ ] **Step 3: Extend schema validation and trusted factories**

Add `list[object]` validation that recursively rejects unsafe keys and non-JSON values. Register the gate factory and add special validation-plan construction for its rules. The validation profile must include model block IDs and the implementation source must return the proposal’s operator, not an alias to Block/FFN/Compose. Store implementation source digest, generated plan digest, runtime kind, parameter delta, reload result, and the proposal’s rationale in `SearchSpaceProposal.validation`.

- [ ] **Step 4: Make deterministic Agent discovery propose the new capability**

When the active registry lacks `timestep_block_gate` and feedback contains a timestep-quality failure or block-space limitation, return a full `OperatorSpec` including rationale, constraints, limitations, trusted factory key, and implementation source. Keep threshold proposal support for existing tests and provider-generated proposals. Ensure `classify_search_space_proposal` labels the gate as `compression_space_extension`, with a validated structural counterexample showing that static Block/FFN/Compose cannot represent a timestep-dependent rule.

- [ ] **Step 5: Prove inheritance and classification**

Add assertions that the next `SearchSpaceManager` instance reads the active `S####` version, that the Algorithm Program receives `timestep_block_gate` in its operator names, and that a child algorithm using it is classified as `compression_space_extension`, not parameter search or a renamed rule.

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/unit/test_search_space.py tests/unit/test_algorithm_evolution.py tests/unit/test_algorithm_program_search.py`

Expected: old threshold/generated-rule tests pass and the new gate follows Proposal → validation → registration → registry → Algorithm Program use.

```bash
git add harness4h3/operators/wan/search_space.py harness4h3/sandbox/runner.py harness4h3/evolution/agent.py harness4h3/evolution/taxonomy.py tests/unit/test_search_space.py tests/unit/test_algorithm_evolution.py
git commit -m "feat: connect validated operators to algorithm search space"
```

### Task 5: Finish recursive Algorithm/Model Evolution separation in EvolutionLoop

**Files:**
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/improvement.py`
- Modify: `harness4h3/evolution/store.py`
- Modify: `harness4h3/evolution/skill_hub.py`
- Modify: `harness4h3/evolution/README.md`
- Create: `tests/integration/test_algorithm_discovery_recursive_loop.py`
- Modify: `tests/integration/test_tica_recursive_loop.py`

**Interfaces:**
- `EvolutionLoop` writes `algorithm_program_trace.json` for every source-driven search and copies its digest into `decision.json`.
- `AlgorithmValidationRecord` stores `search_space_version`, `source_diff_digest`, parent/child search traces, equal budget, and independent Validation result.
- `_record_experience` records algorithm status and model promotion separately; `_record_search_space_gene` records a validated operator only after registration.

- [ ] **Step 1: Write the failing three-generation integration test**

```python
def test_three_generations_modify_source_use_feedback_and_inherit_gate(tmp_path):
    loop = make_fake_loop(tmp_path, algorithm_improvement=True, search_space_evolution=True)
    first = loop.run_round()
    second = loop.run_round()
    third = loop.run_round()
    algorithms = [json.loads((tmp_path / "algorithms" / (item.algorithm_id + ".json")).read_text()) for item in (first, second, third)]
    assert algorithms[1]["parent_algorithm_id"] == algorithms[0]["algorithm_id"]
    assert algorithms[2]["parent_algorithm_id"] == algorithms[1]["algorithm_id"]
    assert algorithms[1]["source_diff"] and algorithms[2]["source_diff"]
    assert all(json.loads((tmp_path / "rounds" / round_id / "tica-search.json").read_text())["algorithm_trace"] for round_id in (first.round_id, second.round_id, third.round_id))
    assert json.loads((tmp_path / "rounds" / third.round_id / "search-space.json").read_text())["version_id"].startswith("S")
```

- [ ] **Step 2: Run the integration test and verify it fails**

Run: `pytest -q tests/integration/test_algorithm_discovery_recursive_loop.py`

Expected: current loop may create source lineage, but it does not persist the program trace or prove that the third generation consumes the registered gate operator.

- [ ] **Step 3: Pass Algorithm Program context through every search and validation path**

Update `_run_tica_search`, `_complete_tica_algorithm_measurement_evaluator`, and resumed-round loading to pass the algorithm source, retrieved history, active SearchSpaceVersion, and Sandbox. Ensure parent and child validation invoke independent search instances with identical limits and profile/checkpoint digests. Persist the raw trace before compression and restore it on resume instead of rerunning a completed source invocation.

- [ ] **Step 4: Keep model promotion independent**

Do not call `ModelStore.set_active` from Algorithm Validator success. Only the existing PromotionPolicy decision for the evaluated child model can promote it. Record `algorithm_validation_status`, `model_promotion_status`, `search_space_version`, and `algorithm_change_taxonomy` as separate decision fields. A rejected model still leaves a valid algorithm candidate and validation record available to SkillHub only when the algorithm validation itself passed.

- [ ] **Step 5: Make SkillHub retrieval include source and operator capabilities**

Store validated source, source diff, SearchSpaceVersion, and trace evidence in algorithm skills. Retrieval results must expose `algorithm_source`, `algorithm_metadata.search_space_version`, and the operator names used, so later Agent calls can genuinely reuse a validated gate algorithm rather than only its numeric outcome.

- [ ] **Step 6: Run recursion, legacy, and recovery tests and commit**

Run: `pytest -q tests/integration/test_algorithm_discovery_recursive_loop.py tests/integration/test_tica_recursive_loop.py tests/unit/test_algorithm_evolution.py tests/unit/test_algorithm_improvement.py`

Expected: at least three Fake CPU generations produce distinct source hashes and traces, parent/child validation uses equal budgets, model promotion remains independent, and old recursive/recovery behavior remains green.

```bash
git add harness4h3/evolution/loop.py harness4h3/evolution/improvement.py harness4h3/evolution/store.py harness4h3/evolution/skill_hub.py harness4h3/evolution/README.md tests/integration/test_algorithm_discovery_recursive_loop.py tests/integration/test_tica_recursive_loop.py
git commit -m "feat: separate recursive algorithm validation from model promotion"
```

### Task 6: Add the real Wan configuration, reports, and full verification

**Files:**
- Create: `configs/evolutionary-compression.wan-algorithm-discovery.example.yaml`
- Modify: `harness4h3/evolution/config.py`
- Modify: `harness4h3/evolution/README.md`
- Create: `docs/superpowers/reports/2026-10-15-algorithm-discovery-rsi-acceptance.md`
- Modify: `tests/integration/test_evolutionary_compression.py`

**Interfaces:**
- `configs/evolutionary-compression.wan-algorithm-discovery.example.yaml` pins Wan2.1-1.3B checkpoint, Calibration/Validation/Test manifests, source revision, operator/search-space evolution flags, bounded Sandbox timeout, and a 10–20 round command.
- The report distinguishes proven Fake CPU behavior, real Wan structural preflight, and unrun CUDA/video/latency evidence.

- [ ] **Step 1: Write the configuration contract test**

```python
def test_algorithm_discovery_wan_config_has_fixed_splits_and_bounded_budget():
    config = CompressionConfig.from_path(Path("configs/evolutionary-compression.wan-algorithm-discovery.example.yaml"))
    assert config.tica_prune is True
    assert config.algorithm_improvement is True
    assert config.search_space_evolution is True
    assert config.sandbox_timeout_s > 0
```

- [ ] **Step 2: Add the minimal Wan configuration and command**

Use this command in the README and report:

```bash
uv run --with torch --with diffusers python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.wan-algorithm-discovery.example.yaml \
  --rounds 10 --json
```

The example must require explicit real cached Calibration/Validation/Test assets and must not substitute Fake data when `backend=wan_diffusers`.

- [ ] **Step 3: Run the complete CPU verification suite**

Run:

```bash
pytest -q tests/unit/test_algorithm_program.py tests/unit/test_algorithm_program_search.py tests/unit/test_timestep_block_gate.py tests/unit/test_search_space.py tests/integration/test_algorithm_discovery_recursive_loop.py tests/integration/test_tica_recursive_loop.py
python3 -m compileall -q harness4h3 h3_training research tools
git diff --check
```

Expected: all focused tests pass, compileall exits zero, and diff check emits no whitespace errors.

- [ ] **Step 4: Run the existing regression suite**

Run: `pytest -q`

Expected: existing CPU tests pass with only their documented optional-dependency skips. If a regression is found, fix it in the smallest affected boundary and rerun the focused test before the full suite.

- [ ] **Step 5: Execute the Fake end-to-end evidence run**

Run:

```bash
uv run --with torch --with pytest python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.fake-rsi.yaml \
  --rounds 3 --json
```

Inspect the generated `algorithm.json`, `tica-search.json`, `search-space.json`, `compression.json`, `evaluation.json`, `decision.json`, and `algorithm-validation.json`. Confirm that the report cites only these artifacts and explicitly labels real Wan GPU/video results as pending when no CUDA device/assets are available.

- [ ] **Step 6: Commit report, tag the verification state, and prepare handoff**

```bash
git add configs/evolutionary-compression.wan-algorithm-discovery.example.yaml harness4h3/evolution/config.py harness4h3/evolution/README.md docs/superpowers/reports/2026-10-15-algorithm-discovery-rsi-acceptance.md tests/integration/test_evolutionary_compression.py
git commit -m "docs: record algorithm discovery RSI acceptance"
git status --short --branch
git log --oneline --decorate -8
```

The final handoff must include the independent branch name, focused/full test counts, Fake evidence paths, the real Wan command, and the exact remaining GPU validation steps. Push the branch only after all checks pass.

## Plan Self-Review

- Spec coverage: Tasks 1–2 cover executable source-controlled scoring/candidate/search logic; Task 3 covers a real non-static operator; Task 4 covers proposal validation and active registry inheritance; Task 5 covers recursive Algorithm Evolution vs Model Evolution; Task 6 covers configuration, evidence, and regression verification.
- Placeholder scan: no task depends on an unspecified implementation, future work marker, or unnamed interface.
- Type consistency: `AlgorithmSandbox.run_program` returns `ProgramResult`; TICA consumes its `value`; `TimestepBlockGatingOperator` is created only through a trusted factory; `SearchSpaceManager.registry_for` supplies the same operator specs to the Program and executor.
- Scope: one integrated plan is appropriate because the requested acceptance path cannot prove Algorithm Discovery without connecting Sandbox, TICA search, SearchSpace registry, Adapter execution, and EvolutionLoop lineage.
