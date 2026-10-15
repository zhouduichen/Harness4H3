# TICA-Prune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a human-designed TICA-Prune structural compression baseline for Wan2.1-1.3B, then expose its scoring and bounded search rules to controlled AlgorithmAgent source evolution without claiming autonomous discovery.

**Architecture:** Add a versioned calibration-input contract and a pure TICA planner beside the existing profiler and compression registry. The planner produces auditable recipes; a trusted structural operator executes recipes using existing block/FFN mutations, while a bounded evaluator measures timestep errors and interactions on fixed parent checkpoints. EvolutionLoop and AlgorithmValidator reuse existing lineage, Sandbox, SkillHub, and PromotionPolicy records, with algorithm validation kept independent from model promotion. Add only a disabled runtime-attention protocol for future SVOO-style operators.

**Tech Stack:** Python 3, PyTorch when installed, existing WanModelAdapter/DiffusersWanAdapter/FakeWanDiTAdapter, existing AlgorithmSandbox, WanCompressionRegistry, AlgorithmValidator, EvolutionLoop, pytest, JSON manifests.

## Global Constraints

- TICA-Prune is recorded as `origin=human_baseline`; it must never be described as Agent-discovered.
- Develop only on `codex/tica-prune`, based on `codex/rsi-core-verification` at `cdd5a8b`.
- Do not modify or use the four running Wan experiment GPUs, checkpoints, configurations, or output roots.
- New TICA and runtime-attention behavior is opt-in and CPU Fake tests run first.
- Reuse Model Candidate, Algorithm Candidate, SearchSpaceVersion, ExperienceStore, SkillHub, Sandbox, Evaluator, and PromotionPolicy.
- Global FFN channel retention is required while `DiffusersWanAdapter.channel_pruning_requires_all_blocks` is true; reject unequal per-layer FFN widths.
- Estimated FLOPs are structural estimates and must not be reported as GPU latency, speedup, or peak-memory reduction.
- CPU Fake output error is an offline quality proxy and must not be reported as real Wan video quality.
- Missing activation, calibration, runtime, or hardware evidence is unavailable, not zero.

---

### Task 1: Add versioned calibration-input contracts

**Files:**
- Create: `harness4h3/evolution/tica_calibration.py`
- Modify: `harness4h3/adapters/wan/contracts.py:166-203`
- Modify: `harness4h3/adapters/wan/fake.py:130-330`
- Modify: `harness4h3/adapters/wan/diffusers.py:250-310`
- Modify: `harness4h3/profiler/__init__.py`
- Test: `tests/unit/test_tica_calibration.py`

**Interfaces:**
- Produce `TICACalibrationSpec(sample_id, prompt, seed, latent_shape, timesteps, latent_path=None, encoder_hidden_states_path=None)` with `to_dict()` and `digest()`.
- Produce `TICACalibrationInput(spec, hidden_states, encoder_hidden_states, metadata)` with `to_dict_metadata()` and `digest()`; tensor payloads stay outside JSON and are loaded from explicit paths or provider output.
- Produce `TICACalibrationManifest(specs, provider_name, manifest_id)` with `digest()`, `to_dict()`, `samples(model, adapter)`, and `timesteps()`.
- Extend `WanModelAdapter` with optional-standard `tica_calibration_inputs(model, manifest) -> tuple[TICACalibrationInput, ...]` and `forward_denoising(model, calibration_input, timestep) -> Any`.

- [ ] **Step 1: Write failing contract tests.**

```python
def test_fake_tica_manifest_is_deterministic_and_cpu_only():
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline")
    manifest = TICACalibrationManifest.from_fake(
        hidden_size=8, sample_count=2, timesteps=(0.9, 0.5, 0.1), seed=17
    )
    first = adapter.tica_calibration_inputs(model, manifest)
    second = adapter.tica_calibration_inputs(model, manifest)
    assert manifest.digest() == manifest.digest()
    assert first[0].digest() == second[0].digest()
    assert all(item.metadata["device"] == "cpu" for item in first)

def test_real_manifest_requires_explicit_cached_tensors(tmp_path):
    spec = TICACalibrationSpec("s0", "prompt", 7, (1, 4, 8, 8), (0.9, 0.5))
    manifest = TICACalibrationManifest((spec,), provider_name="cached_real_wan")
    with pytest.raises(ValueError, match="cached calibration tensor"):
        manifest.samples(model=None, adapter=object())
```

- [ ] **Step 2: Run the focused tests and verify they fail because the TICA contracts do not exist.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q tests/unit/test_tica_calibration.py`

Expected: collection or attribute failures for the new types and adapter methods.

- [ ] **Step 3: Implement serializable specs and deterministic providers.**

Use explicit tensor paths for real cached inputs and a CPU-only Fake provider. The manifest digest must include sample IDs, prompt metadata, seed, latent shape, timestep order, and tensor file hashes. Do not serialize raw tensors into the algorithm source or ProfileReport.

- [ ] **Step 4: Implement adapter methods.**

Fake uses seeded CPU tensors and its existing forward path. Diffusers accepts only cached tensors already on disk, moves them to the transformer device/dtype, sets the requested timestep, and calls the transformer. If cached encoder states or tensor shape is missing, it raises `unsupported_tica_calibration_input` rather than falling back to random `profile_sample`.

- [ ] **Step 5: Run focused tests and commit.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q tests/unit/test_tica_calibration.py tests/unit/test_wan_profiler.py`

Expected: all focused tests pass and legacy profiler tests remain unchanged.

Commit: `git add harness4h3/evolution/tica_calibration.py harness4h3/adapters/wan/contracts.py harness4h3/adapters/wan/fake.py harness4h3/adapters/wan/diffusers.py harness4h3/profiler/__init__.py tests/unit/test_tica_calibration.py && git commit -m "feat: add TICA calibration input contract"`

### Task 2: Implement TICA scoring and bounded search

**Files:**
- Create: `harness4h3/evolution/tica_prune.py`
- Modify: `harness4h3/profiler/sensitivity.py:1-280`
- Modify: `harness4h3/profiler/report.py:1-90`
- Test: `tests/unit/test_tica_prune.py`

**Interfaces:**
- Produce `TICARecipe` with `time_weight_formula`, `time_weights`, `risk_limit`, `activation_weight`, `interaction_weight`, `beam_width`, `max_fast_evals`, `max_full_evals`, `max_interaction_evals`, `strategy`, and `baseline_name`.
- Produce `TICACandidate` with `candidate_id`, `kind`, `plan`, `parent_candidate_ids`, `fast_metrics`, `full_metrics`, `interaction_metrics`, `budget_cost`, and `status`.
- Produce `TICASearchResult` with `recipe`, `candidates`, `pareto_front`, `budget`, `calibration_digest`, and `evidence_level`.
- Expose `TICAPruner.search(adapter, parent, manifest, profile, recipe) -> TICASearchResult` and `TICAPruner.measure_interaction(...) -> Mapping[str, Any]`.
- Expose `build_human_tica_source(algorithm_id) -> str`, whose source returns a `tica_prune` plan based on ProfileReport values and never hard-codes a model block ID.

- [ ] **Step 1: Write failing tests for formulas and baseline labels.**

```python
def test_time_weighted_risk_changes_when_profile_formula_changes():
    profile = fake_tica_profile()
    uniform = tica_score(profile, time_weight_formula="uniform")
    early = tica_score(profile, time_weight_formula="early_log")
    assert uniform.block_risk != early.block_risk
    assert uniform.time_weights != early.time_weights

def test_global_ffn_recipe_rejects_layer_specific_widths():
    with pytest.raises(ValueError, match="global FFN"):
        TICARecipe.from_mapping({"per_layer_retained_indices": {"block.0": [0]}})

def test_human_source_is_profile_dependent_and_auditable(tmp_path):
    source = build_human_tica_source("A0001")
    sandbox = AlgorithmSandbox(tmp_path / "sandbox")
    low = sandbox.run(source, fake_tica_profile(block_bias=0.1), [])
    high = sandbox.run(source, fake_tica_profile(block_bias=0.9), [])
    assert low.status == high.status == "success"
    assert low.plan.to_dict() != high.plan.to_dict()
```

- [ ] **Step 2: Run the focused tests to verify the TICA scorer and source are absent.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q tests/unit/test_tica_prune.py`

Expected: collection or import failures for `TICARecipe`, `TICAPruner`, and `build_human_tica_source`.

- [ ] **Step 3: Implement deterministic time-weighted risk and activation contribution.**

Support `uniform`, `early_log`, and `late_log` formulas. Normalize weights to sum to one and persist the formula plus values. Compute relative denoising error as `mean(abs(candidate-reference)) / max(mean(abs(reference)), eps)`. Combine activation magnitude and paired FFN weight contribution only when activation evidence is available; otherwise preserve the unavailable flag and do not invent a value.

- [ ] **Step 4: Implement candidate generation and Greedy + Beam search.**

Generate the four named baselines from one candidate interface. Fast candidates use adapter `probe_block_removal` or equivalent local output probes. Full candidates reload/execute the actual structural plan. Every measured candidate increments a bounded budget counter; beam expansion stops before exceeding `max_full_evals` or `max_interaction_evals`. Pareto dominance uses quality error, parameter count, FLOPs estimate, and elapsed cost.

- [ ] **Step 5: Implement measured interactions.**

For selected block pairs and block+FFN recipes, execute the composed structural mutation through the existing registry. Store `joint_error`, `independent_error_sum`, `interaction_delta`, input digest, parent hash, and measured cost. A failed or unavailable joint measurement cannot be converted to zero.

- [ ] **Step 6: Run focused tests and commit.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q tests/unit/test_tica_prune.py tests/unit/test_wan_compression.py tests/unit/test_profiler_sensitivity.py`

Expected: all tests pass with no hard-coded interaction constants.

Commit: `git add harness4h3/evolution/tica_prune.py harness4h3/profiler/sensitivity.py harness4h3/profiler/report.py tests/unit/test_tica_prune.py && git commit -m "feat: add TICA risk and bounded interaction search"`

### Task 3: Add trusted TICA structural execution and algorithm-level evidence

**Files:**
- Modify: `harness4h3/operators/wan/compression.py:1-260`
- Modify: `harness4h3/operators/wan/search_space.py:1-560`
- Modify: `harness4h3/evolution/algorithm_evaluation.py:80-230`
- Modify: `harness4h3/evolution/improvement.py:90-330`
- Test: `tests/unit/test_tica_operator.py`
- Test: `tests/unit/test_algorithm_improvement.py`

**Interfaces:**
- Register `TICA_PRUNE_OPERATOR = "tica_prune"` as a trusted structural recipe operator, not a new model-structure primitive.
- `TICAPruningOperator.execute(adapter, parent, plan, output_dir) -> CompressionResult` executes only block pruning, global FFN channel pruning, or their trusted compose sequence and returns the actual structural change.
- Extend `AlgorithmModelEvaluator` to accept a calibration manifest and evaluate TICA plans across fixed timesteps, aggregating output error and preserving per-timestep records.
- Extend `AlgorithmValidationRecord` with `search_budget`, `calibration_digest`, and `interaction_evidence` while preserving old fields.

- [ ] **Step 1: Write failing operator/evidence tests.**

```python
def test_tica_operator_executes_global_ffn_recipe_and_reloads(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = build_wan_compression_registry().execute(
        "tica_prune", adapter, parent,
        {"block_ids": ["block.0"], "retained_indices": list(range(8)), "mode": "block_plus_ffn"},
        tmp_path / "child",
    )
    assert result.status == "success"
    assert adapter.inspect(adapter.reload_model(result.child_checkpoint)).ffn_width == 8

def test_tica_operator_rejects_partial_global_alignment(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = build_wan_compression_registry().execute(
        "tica_prune", adapter, parent,
        {"mode": "block_plus_ffn", "block_ids": ["block.0"],
         "per_layer_retained_indices": {"block.0": [0, 1], "block.1": [0]}},
        tmp_path / "child",
    )
    assert result.status == "failed"
    assert result.failure_type == "global_ffn_alignment_required"
```

- [ ] **Step 2: Run tests to verify the trusted TICA operator is missing.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q tests/unit/test_tica_operator.py`

Expected: registry lookup failure for `tica_prune`.

- [ ] **Step 3: Implement the operator by delegating to existing primitives.**

Validate all block IDs and global retained indices before mutation. For a mixed plan, remove blocks first, then pass the remaining block IDs and one shared retained-index tuple to the existing `FFNChannelPruningOperator`. Do not alter DiffusersWanAdapter core structure logic.

- [ ] **Step 4: Add per-timestep evaluator evidence.**

Use `tica_calibration_inputs` and `forward_denoising` when available. Record parent/child output digests, relative errors, timestep weights, parameter count, structural FLOPs, elapsed time, and hardware-unavailable reason. Keep the existing `profile_sample` evaluator path for non-TICA legacy algorithms.

- [ ] **Step 5: Add validator fairness checks.**

Require the parent and child TICA records to have the same parent checkpoint SHA-256, calibration digest, profile digest, seed conditions, and search budget. If any condition differs, return invalid evidence with a precise reason even when the child score is numerically higher.

- [ ] **Step 6: Run focused tests and commit.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q tests/unit/test_tica_operator.py tests/unit/test_algorithm_improvement.py tests/unit/test_wan_compression.py`

Expected: existing legacy evaluator tests pass and TICA evidence contains actual per-timestep measurements.

Commit: `git add harness4h3/operators/wan/compression.py harness4h3/operators/wan/search_space.py harness4h3/evolution/algorithm_evaluation.py harness4h3/evolution/improvement.py tests/unit/test_tica_operator.py tests/unit/test_algorithm_improvement.py && git commit -m "feat: execute TICA recipes with auditable evidence"`

### Task 4: Integrate the human baseline and recursive Agent revisions

**Files:**
- Modify: `harness4h3/evolution/agent.py:1-520`
- Modify: `harness4h3/evolution/loop.py:90-1180`
- Modify: `harness4h3/evolution/candidate.py:1-220`
- Modify: `harness4h3/evolution/design.py:1-160`
- Modify: `harness4h3/evolution/skill_hub.py:1-260`
- Test: `tests/integration/test_tica_evolution.py`

**Interfaces:**
- `AlgorithmCandidate.metadata["origin"] == "human_baseline"` for the initial TICA candidate.
- `EvolutionLoop(..., tica_prune=True, tica_manifest=TICACalibrationManifest(...))` is opt-in and leaves existing defaults unchanged.
- `AlgorithmAgent.improve` receives bounded TICA feedback keys: formula, time weights, interaction deltas, risk failures, search budget, and Pareto deltas.
- `algorithm-validation.json` stores `parent_algorithm_id`, source hashes/diff, calibration digest, search budget, baseline name, and parent/child TICA results.

- [ ] **Step 1: Write failing integration tests.**

```python
def test_tica_round_records_human_origin_and_four_baselines(tmp_path):
    loop = build_fake_tica_loop(tmp_path)
    outcome = loop.run_round()
    assert outcome.status in {"promoted", "rejected"}
    design = json.loads((tmp_path / "run" / "rounds" / outcome.round_id / "design.json").read_text())
    assert design["origin"] == "human_baseline"
    assert set(design["baseline_comparisons"]) == {
        "independent_block", "tica_block", "interaction_block", "tica_block_ffn"
    }

def test_child_tica_source_change_is_fairly_compared_and_not_auto_success(tmp_path):
    loop = build_fake_tica_loop(tmp_path, algorithm_improvement=True)
    first = loop.run_round()
    second = loop.run_round()
    record = json.loads((tmp_path / "run" / "rounds" / second.round_id / "algorithm-validation.json").read_text())
    assert record["parent_algorithm_id"] == first.algorithm_id
    assert record["calibration_digest"]
    assert record["search_budget"] == record["parent_metrics"]["search_budget"]
    assert record["valid"] is not True or record["evidence_level"] == "cpu_fake_quality_proxy"
```

- [ ] **Step 2: Run integration tests to verify the opt-in path and evidence fields are missing.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q tests/integration/test_tica_evolution.py`

Expected: missing `tica_prune` configuration, origin, baseline, or validation evidence failures.

- [ ] **Step 3: Add the human TICA generator.**

Use `build_human_tica_source` for the initial candidate and write its `origin`, formula, recipe digest, and baseline scope to AlgorithmCandidate metadata and DesignGene evidence. The source reads profile scores and emits a recipe; it must not embed the selected block ID.

- [ ] **Step 4: Integrate TICA execution as an opt-in EvolutionLoop path.**

Build the TICA profile from the manifest, run the four baselines under one fixed budget, select the candidate from their Pareto set, and route the selected recipe through the trusted operator and independent evaluator. Keep model promotion exactly under the existing `PromotionPolicy`.

- [ ] **Step 5: Extend Agent feedback and lineage.**

When improving a TICA parent, include source diff, prior formula, measured time-weight/risk changes, interaction deltas, budget use, and Pareto deltas in the provider payload. Deterministic fallback may change parameters for compatibility but cannot mark the result as an algorithm improvement unless the source changes and the validator confirms measured improvement.

- [ ] **Step 6: Run integration tests and commit.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q tests/integration/test_tica_evolution.py tests/integration/test_evolutionary_compression.py`

Expected: TICA tests pass; legacy EvoCompress tests pass unchanged.

Commit: `git add harness4h3/evolution/agent.py harness4h3/evolution/loop.py harness4h3/evolution/candidate.py harness4h3/evolution/design.py harness4h3/evolution/skill_hub.py tests/integration/test_tica_evolution.py && git commit -m "feat: integrate human TICA baseline into RSI"`

### Task 5: Add disabled Runtime Compression Operator boundary

**Files:**
- Create: `harness4h3/evolution/runtime_operator.py`
- Modify: `harness4h3/adapters/wan/contracts.py:166-203`
- Modify: `harness4h3/evolution/taxonomy.py:1-150`
- Modify: `harness4h3/evolution/config.py:1-100`
- Test: `tests/unit/test_runtime_compression_operator.py`
- Docs: `docs/evo-compress-next-stage.md`

**Interfaces:**
- Produce `RuntimeCompressionCapability(name, supported, evidence_level, changes_weights, requires_cuda, notes)`.
- Produce `RuntimeCompressionOperator` protocol with `prepare_runtime`, `run`, and `measure_runtime`.
- Produce `DisabledSparseAttentionProposal` that records SVOO/SparseVideo-inspired offline profile, online routing, backend, and hardware requirements without executing them.

- [ ] **Step 1: Write failing protocol and safety tests.**

```python
def test_sparse_attention_proposal_is_disabled_by_default():
    proposal = DisabledSparseAttentionProposal.for_svoo()
    assert proposal.enabled is False
    assert proposal.changes_weights is False
    assert proposal.requires_cuda is True

def test_fake_adapter_refuses_runtime_operator_without_dispatch():
    with pytest.raises(RuntimeError, match="runtime dispatch unavailable"):
        proposal.prepare(FakeWanDiTAdapter())
```

- [ ] **Step 2: Implement capability-only boundary.**

No CUDA kernel, monkey patch, third-party import, or attention replacement is added. The proposal must be classified as runtime compression, never as parameter reduction or model checkpoint shrinkage.

- [ ] **Step 3: Document research boundary and run tests.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q tests/unit/test_runtime_compression_operator.py`

Expected: all tests pass with runtime proposal disabled.

Commit: `git add harness4h3/evolution/runtime_operator.py harness4h3/adapters/wan/contracts.py harness4h3/evolution/taxonomy.py harness4h3/evolution/config.py tests/unit/test_runtime_compression_operator.py docs/evo-compress-next-stage.md && git commit -m "feat: add disabled runtime compression boundary"`

### Task 6: CPU Fake diagnostic experiment, documentation, and full verification

**Files:**
- Create: `configs/evolutionary-compression.fake-tica.yaml`
- Create: `docs/tica-prune.md`
- Modify: `harness4h3/cli.py:886-930`
- Test: `tests/integration/test_tica_diagnostics.py`

**Interfaces:**
- Add opt-in CLI/config flag `tica_prune` and a relative output root under the current worktree only.
- Persist one diagnostic artifact containing the four baseline records, common calibration digest, common search budget, Pareto front, and no-hardware-claim fields.

- [ ] **Step 1: Write failing diagnostic test.**

```python
def test_fake_tica_diagnostic_has_four_common_budget_baselines(tmp_path):
    result = run_fake_tica_diagnostic(tmp_path)
    assert result["calibration"]["offline_simulation"] is True
    assert len(result["baselines"]) == 4
    assert len({item["budget"]["max_full_evals"] for item in result["baselines"]}) == 1
    assert all(item["hardware_metrics"]["available"] is False for item in result["baselines"])
    assert any(item["interaction_metrics"] for item in result["baselines"])
```

- [ ] **Step 2: Implement CLI/config and artifact writer.**

Keep legacy configs accepted without new fields. Write only inside the supplied opt-in output root. Include source hash, manifest digest, profile digest, model checkpoint hash, seed, candidate generation counts, full/fast measurement counts, actual structural change, and Pareto results.

- [ ] **Step 3: Run the diagnostic and inspect its JSON.**

Run: `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m harness4h3 tica-diagnostic --config configs/evolutionary-compression.fake-tica.yaml --json`

Expected: successful CPU-only run with four baseline names, nonconstant measured interaction deltas, and `hardware_metrics.available=false`.

- [ ] **Step 4: Add documentation and run the complete test suite.**

Run: `git diff --check` followed by `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q`

Expected: all existing tests plus the TICA tests pass; any real Wan/GPU tests remain skipped for missing opt-in environment variables.

- [ ] **Step 5: Commit, verify isolation, and push.**

Run: `git status --short`, `ps -p 41692,41758 -o pid=,stat=,command=`, and `git push -u origin codex/tica-prune`.

Expected: clean branch, existing Wan processes unchanged, and the new branch points to the final commit.
