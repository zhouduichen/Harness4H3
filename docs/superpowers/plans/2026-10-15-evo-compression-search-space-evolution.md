# EvoCompress Search-Space Evolution Implementation Plan

> **For agentic workers:** Implement this plan task-by-task in the isolated
> worktree. Each task ends with a focused test cycle and a commit.

**Goal:** Add CPU/Fake-first multi-granularity profiling, recursive algorithm
improvement, versioned compression-search-space evolution, and staged candidate
screening without changing the default EvoCompress or real Wan execution path.

**Architecture:** Extend the existing `ProfileReport`, `AlgorithmAgent`,
`AlgorithmStore`, `WanCompressionRegistry`, `EvolutionLoop`, `SkillHub`, and
`PromotionPolicy` through additive contracts. New profiling, improvement,
search-space, and screening modules own their data models and persistence;
trusted execution stays in existing Sandbox/adapter/operator boundaries.

**Tech Stack:** Python 3.9+, dataclasses, JSON/JSONL atomic persistence,
PyTorch CPU FakeWanDiT, pytest, existing Sandbox and Wan adapter contracts.

## Global Constraints

- Work only in `/Users/huangjiahao/MinMax-H3/worktrees/Harness4H3-evo-search-space` on branch `codex/evo-search-space-evolution`.
- Never edit `/Users/huangjiahao/MinMax-H3/Harness4H3`, its dirty files, remote experiment checkouts, running output roots, model weights, or GPU jobs.
- New feature switches default to disabled and existing public call signatures remain backward compatible.
- CPU Fake/Mock tests must prove all new behavior; no test may allocate CUDA or call the real Wan evaluator.
- Every measured, estimated, unavailable, offline, and real evidence state must be explicit in persisted artifacts.
- Agent source remains subject to the existing AST/process Sandbox; an Agent may propose a search-space operator but cannot register executable code.
- Existing `PromotionPolicy` remains the only model promotion authority; CPU/Fake evidence cannot promote a real Wan model.
- Wan FFN channel alignment and checkpoint save/reload/hash validation remain enforced.
- Use `apply_patch` for source edits and commit each completed task with a focused message.

## Shared test fixtures

The snippets below refer to small test-only helpers. Define them in the test
module that first uses them; they are not production APIs:

```python
def fake_profile():
    return {
        "model": {"architecture": "WanFakeDiT", "block_count": 4, "ffn_width": 16},
        "layers": [
            {"block_id": "block.0", "estimated_flops": 10.0, "latency_ms": 1.0},
            {"block_id": "block.1", "estimated_flops": 40.0, "latency_ms": 4.0},
            {"block_id": "block.2", "estimated_flops": 20.0, "latency_ms": 2.0},
            {"block_id": "block.3", "estimated_flops": 30.0, "latency_ms": 3.0},
        ],
        "block_sensitivity": [
            {"block_id": "block.%d" % index, "importance_score": 0.1 + index * 0.1, "available": True}
            for index in range(4)
        ],
        "channel_sensitivity": [{"block_id": "block.0", "width": 16, "available": True, "channels": []}],
        "timestep_sensitivity": {
            "high_noise": {"available": True, "importance_score": 0.1},
            "middle": {"available": True, "importance_score": 0.2},
            "low_noise": {"available": True, "importance_score": 0.3},
        },
    }

def parent_source():
    return "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n"

def quality_failure():
    return {"experience_id": "R0001", "status": "rejected", "decision": {"promotion": {"reasons": ["quality_floor_failed"]}}}
```

Search-space tests use a `FakeWanDiTAdapter(block_count=4, hidden_size=8,
ffn_width=16)`, save `fake://baseline` to the test temporary directory, and
use a diagnostic callback that returns `{"valid": True,
"offline_simulation": True, "search_score": 0.9}`. Screening tests inject
the same adapter and a callback that appends to a list before returning that
record. Integration tests construct `EvolutionLoop` with separate temporary
`output_root`, `AlgorithmSandbox`, and the default structural proxy evaluator;
they never use a real Wan checkpoint.

---

### Task 1: Add deterministic calibration samples and ProfileReport v2 compatibility

**Files:**
- Create: `harness4h3/profiler/calibration.py`
- Modify: `harness4h3/profiler/report.py`
- Modify: `harness4h3/profiler/__init__.py`
- Modify: `harness4h3/profiler/layer_profiler.py`
- Test: `tests/unit/test_profiler_calibration.py`
- Test: `tests/unit/test_wan_profiler.py`

**Interfaces:**
- `CalibrationSampleSpec(prompt: str, seed: int, latent_shape: tuple[int, ...], timesteps: tuple[float, ...], sample_id: str = "")`.
- `CalibrationSampleSpec.to_dict() -> dict[str, Any]` and `.digest() -> str`.
- `CalibrationProvider.build(spec: CalibrationSampleSpec) -> CalibrationSample`.
- `CalibrationProvider.batch(specs: Sequence[CalibrationSampleSpec]) -> tuple[CalibrationSample, ...]`.
- `FakeWanCalibrationProvider(hidden_size: int, device: str = "cpu").build(...)` returns a deterministic CPU tensor/mapping without CUDA.
- `CalibrationCorpus(specs, provider).samples() -> tuple[CalibrationSample, ...]`, `.to_dict()`, and `.digest()`.
- `profile_model(..., calibration: CalibrationCorpus | None = None, advanced: bool = False)` preserves current output when omitted.
- `ProfileReport` accepts additive `calibration`, `relationships`, and `evidence` mappings; `from_dict` reads schema v1 and v2.

- [ ] **Step 1: Write failing tests for deterministic specs and v1 loading.**

```python
def test_fake_calibration_is_deterministic_and_cpu_only():
    provider = FakeWanCalibrationProvider(hidden_size=8)
    spec = CalibrationSampleSpec("a fixed prompt", 17, (1, 8), (0.9, 0.5, 0.1), "s0")
    first = provider.build(spec)
    second = provider.build(spec)
    assert first.provenance["offline_simulation"] is True
    assert first.provenance["device"] == "cpu"
    assert torch.equal(first.payload, second.payload)

def test_profile_report_reads_schema_v1_without_new_fields():
    report = ProfileReport.from_dict(existing_v1_payload())
    assert report.calibration == {}
    assert report.relationships == {}
    assert report.evidence == {}
```

Run: `.venv/bin/python -m pytest -q tests/unit/test_profiler_calibration.py tests/unit/test_wan_profiler.py`
Expected: new tests fail because calibration types and additive report fields do not exist.

- [ ] **Step 2: Implement the serializable calibration contracts.**

Use a frozen dataclass, canonical JSON through the existing `stable_digest`,
and `torch.Generator(device="cpu").manual_seed(spec.seed)` in the Fake provider.
Make the provider return the shape requested by `latent_shape`, with a final
hidden dimension of `hidden_size` when the spec supplies a one-dimensional
probe shape. Do not call `.cuda()` or inspect global GPU state.

- [ ] **Step 3: Extend `ProfileReport` without breaking schema v1.**

Add default empty mappings, write schema version 2 for new reports, accept
schema versions 1 and 2 in `from_dict`, and retain existing required fields.
Implement recursive JSON conversion for dataclass payloads and calibration
metadata. Reject versions greater than 2.

- [ ] **Step 4: Thread an optional corpus through `profile_model`.**

When `calibration` is supplied, use its first sample as the compatibility
`sample`, persist corpus specs/digest/provenance, and construct relationships
from returned block IDs. When omitted, preserve `_default_sample` and existing
provenance fields. Do not change default channel probe limits yet.

- [ ] **Step 5: Run focused tests and commit.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_profiler_calibration.py tests/unit/test_wan_profiler.py`
Expected: PASS.

```bash
git add harness4h3/profiler tests/unit/test_profiler_calibration.py tests/unit/test_wan_profiler.py
git commit -m "feat: add deterministic EvoCompress calibration corpus"
```

### Task 2: Implement full-channel chunking and explicit sensitivity evidence

**Files:**
- Modify: `harness4h3/profiler/sensitivity.py`
- Modify: `harness4h3/profiler/layer_profiler.py`
- Modify: `harness4h3/profiler/report.py`
- Test: `tests/unit/test_profiler_sensitivity.py`
- Test: `tests/unit/test_wan_profiler.py`

**Interfaces:**
- `ImportanceMeasurement(value, status, method, offline_simulation, reason=None)` as a serializable mapping/dataclass.
- `ChannelImportanceMethod.measure(adapter, model, block, samples, channel_indices) -> tuple[Mapping[str, Any], ...]`.
- `measure_channel_sensitivity(..., method="weight", chunk_size=32, sample_set=(), full=False) -> tuple[Mapping[str, Any], ...]`.
- Supported method names: `weight`, `activation`, `output_perturbation`.

- [ ] **Step 1: Add failing tests for all-channel coverage and evidence.**

```python
def test_full_channel_profile_covers_every_fake_ffn_channel_in_chunks():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=10)
    model = adapter.load_model("fake://baseline")
    rows = measure_channel_sensitivity(adapter, model, method="weight", chunk_size=3, full=True)
    assert [len(row["channels"]) for row in rows] == [10, 10]
    assert {channel["channel"] for channel in rows[0]["channels"]} == set(range(10))
    assert all(channel["evidence"]["status"] == "estimated" for channel in rows[0]["channels"])

def test_unsupported_activation_method_is_unavailable_not_zero():
    rows = measure_channel_sensitivity(FakeWanDiTAdapter(), model, method="activation", full=True)
    assert all(row["available"] is False for row in rows)
    assert all(row["evidence"]["status"] == "unavailable" for row in rows)
```

Run: `.venv/bin/python -m pytest -q tests/unit/test_profiler_sensitivity.py`
Expected: FAIL until method dispatch, chunking, and evidence records are added.

- [ ] **Step 2: Refactor weight sensitivity into a method implementation.**

Keep the existing paired-weight formula and `approximation=True`, but emit an
explicit `evidence` record with `status="estimated"`, method
`"weight_proxy"`, and `offline_simulation` from adapter structure metadata.
Do not silently treat weight magnitude as activation or output perturbation.

- [ ] **Step 3: Add chunked dispatch and fail-closed method stubs.**

Iterate `range(0, width, chunk_size)` and then every index inside each chunk,
so `full=False` retains the legacy limit and `full=True` covers all channels.
Implement Activation and Output Perturbation contracts; Fake adapter may return
unavailable for methods it cannot support, with a reason. Never write zero for
an unavailable metric.

- [ ] **Step 4: Attach evidence and hierarchy to advanced `profile_model`.**

Pass `full=advanced`, a bounded chunk size, and the calibration samples when
advanced mode is enabled. Add per-layer latency/FLOP evidence and a
block-to-channel/timestep relationship map to the report. Keep legacy records
and default calls unchanged.

- [ ] **Step 5: Run focused tests and commit.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_profiler_sensitivity.py tests/unit/test_wan_profiler.py`
Expected: PASS.

```bash
git add harness4h3/profiler tests/unit/test_profiler_sensitivity.py tests/unit/test_wan_profiler.py
git commit -m "feat: add full-channel profiling evidence"
```

### Task 3: Add immutable recursive algorithm-improvement lineage

**Files:**
- Modify: `harness4h3/evolution/candidate.py`
- Modify: `harness4h3/evolution/store.py`
- Modify: `harness4h3/evolution/agent.py`
- Create: `harness4h3/evolution/improvement.py`
- Modify: `harness4h3/evolution/skill_hub.py`
- Test: `tests/unit/test_algorithm_improvement.py`
- Test: `tests/unit/test_algorithm_sandbox.py`

**Interfaces:**
- `AlgorithmImprovementRequest(parent: AlgorithmCandidate, profile, history, failures, search_space) -> to_dict()`.
- `AlgorithmAgent.improve(algorithm_id, parent, parent_id, profile, history, failures, skills=(), search_space=None) -> AlgorithmCandidate`.
- `AlgorithmValidationRecord.to_dict()` with `valid`, `evidence_level`, source hashes, controlled-condition digest, and reasons.
- `AlgorithmValidator.validate(parent, child, sandbox, profile, history, evaluate_plan) -> AlgorithmValidationRecord`.
- `AlgorithmStore.create_child(candidate, parent_source_sha256) -> AlgorithmCandidate` verifies parent identity and source hash.

- [ ] **Step 1: Write failing tests for parent-source-aware improvement.**

```python
def test_deterministic_improvement_changes_parent_source_and_records_reason(tmp_path):
    parent = AlgorithmCandidate.from_source("A0001", parent_source(), hypothesis="baseline")
    agent = AlgorithmAgent()
    child = agent.improve(
        "A0002", parent=parent, parent_id="M0000",
        profile=fake_profile(), history=[quality_failure()],
        failures=["quality_floor_failed"], search_space={"version_id": "S0001"},
    )
    assert child.parent_algorithm_id == parent.algorithm_id
    assert child.metadata["parent_source_sha256"] == parent.source_sha256
    assert child.source != parent.source
    assert child.metadata["change_rationale"]

def test_algorithm_store_rejects_wrong_parent_hash(tmp_path):
    store = AlgorithmStore(tmp_path)
    parent = store.create(parent_candidate())
    with pytest.raises(ValueError, match="parent_source_hash_mismatch"):
        store.create_child(child_candidate(parent), "0" * 64)
```

Run: `.venv/bin/python -m pytest -q tests/unit/test_algorithm_improvement.py`
Expected: FAIL until lineage fields, `improve`, and `create_child` exist.

- [ ] **Step 2: Extend `AlgorithmCandidate` with additive lineage metadata.**

Keep constructor compatibility by storing new fields in `metadata` and exposing
read-only properties where useful. Validate `parent_source_sha256` as a SHA-256
when present. Preserve existing status values and duplicate source checks.

- [ ] **Step 3: Implement `AlgorithmAgent.improve`.**

For deterministic mode, parse the parent source only for its stable `propose`
function and select one of two bounded templates based on feedback: quality
failure yields conservative block selection; repeated efficiency/size failure
yields a profile-guided FFN threshold-style selection over all profiled blocks.
The generated source must be executable by the existing Sandbox and include a
comment describing the change. For provider mode, add bounded parent source,
hash, failure reasons, and search-space version to the provider payload while
requiring the same parent ID in the response.

- [ ] **Step 4: Implement AlgorithmValidator with controlled comparison.**

Run parent and child through the same Sandbox and `evaluate_plan` callback. Use
the same profile, history, model identity, and budget digest. Return valid only
when both plans execute and child metrics meet the supplied comparison rule;
otherwise return a durable rejected record. Never mutate candidate status or
overwrite parent source.

- [ ] **Step 5: Extend Store/SkillHub retrieval and test failures.**

Persist validation record paths and bounded source summaries. Include parent
algorithm ID, source hash, change rationale, validation status, and failure
reasons in compact SkillHub context. Ensure a failed child remains readable and
cannot replace a validated parent.

- [ ] **Step 6: Run focused tests and commit.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_algorithm_improvement.py tests/unit/test_algorithm_sandbox.py tests/unit/test_algorithm_design.py`
Expected: PASS.

```bash
git add harness4h3/evolution tests/unit/test_algorithm_improvement.py tests/unit/test_algorithm_sandbox.py
git commit -m "feat: add recursive algorithm improvement lineage"
```

### Task 4: Add versioned, trusted evolvable compression search space

**Files:**
- Create: `harness4h3/operators/wan/search_space.py`
- Modify: `harness4h3/operators/wan/compression.py`
- Modify: `harness4h3/operators/wan/__init__.py`
- Modify: `harness4h3/sandbox/runner.py`
- Modify: `harness4h3/evolution/skill_hub.py`
- Test: `tests/unit/test_search_space.py`
- Test: `tests/unit/test_wan_compression.py`

**Interfaces:**
- `OperatorSpec`, `SearchSpaceVersion`, `SearchSpaceProposal` frozen dataclasses with `to_dict/from_dict`.
- `SearchSpaceStore(root).create_proposal(...)`, `.append_version(...)`, `.read_proposals()`, `.latest()`.
- `SearchSpaceManager.propose(spec, rationale, evidence_ids) -> SearchSpaceProposal`.
- `SearchSpaceManager.validate(proposal, adapter, parent, sandbox, diagnostic) -> SearchSpaceProposal`.
- `SearchSpaceManager.register(proposal) -> SearchSpaceVersion` only for validated proposals.
- `WanCompressionRegistry.for_search_space(version) -> WanCompressionRegistry` returns a non-mutating registry view.
- Built-in trusted factory key `prune_ffn_by_weight_threshold` accepts `{block_ids, threshold}` and derives retained indices before delegating to `FFNChannelPruningOperator`.

- [ ] **Step 1: Write failing tests for proposal lifecycle and fixed-space rejection.**

```python
def test_unvalidated_operator_is_described_but_not_executable(tmp_path):
    manager = SearchSpaceManager(tmp_path / "space")
    proposal = manager.propose(threshold_spec(), "weight plateau", ["R0001"])
    registry = build_wan_compression_registry()
    assert proposal.status == "proposed"
    with pytest.raises(ValueError, match="unsupported_compression_operator"):
        registry.get("prune_ffn_by_weight_threshold")

def test_fake_threshold_operator_validates_registers_and_reloads(tmp_path):
    manager = SearchSpaceManager(tmp_path / "space")
    proposal = manager.propose(threshold_spec(), "weight plateau", ["R0001"])
    validated = manager.validate(proposal, fake_adapter_and_parent(tmp_path), diagnostic=fake_diagnostic)
    version = manager.register(validated)
    assert version.status == "validated"
    assert "prune_ffn_by_weight_threshold" in version.operator_names()
```

Run: `.venv/bin/python -m pytest -q tests/unit/test_search_space.py`
Expected: FAIL until the version/proposal/store/manager types exist.

- [ ] **Step 2: Implement immutable search-space records and atomic storage.**

Persist proposals and versions under the caller-provided root. Use stable
digests for version IDs and atomic writes. Validate proposal status transitions:
`proposed -> validating -> validated|rejected`; a rejected proposal cannot be
registered and an existing version is immutable.

- [ ] **Step 3: Implement trusted threshold operator and Registry views.**

Use only a built-in factory map. Derive weight scores from the adapter's FFN
projections, retain channels whose score is at least the threshold, reject empty
or full selections, and normalize `block_ids` to every current block when the
adapter requires all-block alignment. Execute through existing save/reload and
hash checks. Do not execute proposal source.

- [ ] **Step 4: Validate and register without changing default Registry.**

Add `for_search_space` and/or an explicit operator map copy. Existing
`build_wan_compression_registry().names()` remains exactly the legacy tuple.
Only a validated version exposes the threshold operator. Record model structure,
forward finite checks, child hash, diagnostic metrics, and offline evidence.

- [ ] **Step 5: Publish validated operator evidence to SkillHub and test rejection.**

Store proposal/version IDs and limitations as a design gene/experience context.
Invalid schema, unknown factory, partial Wan FFN blocks, incompatible Fake
model, and diagnostic failure must persist `rejected` records without registry
mutation.

- [ ] **Step 6: Run focused tests and commit.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_search_space.py tests/unit/test_wan_compression.py tests/unit/test_algorithm_sandbox.py`
Expected: PASS; legacy operator names remain unchanged.

```bash
git add harness4h3/operators/wan harness4h3/sandbox harness4h3/evolution/skill_hub.py tests/unit/test_search_space.py tests/unit/test_wan_compression.py
git commit -m "feat: add validated EvoCompress search-space versions"
```

### Task 5: Implement restartable three-stage candidate screening

**Files:**
- Create: `harness4h3/evolution/screening.py`
- Modify: `harness4h3/evolution/__init__.py`
- Modify: `harness4h3/evolution/config.py`
- Test: `tests/unit/test_candidate_screening.py`

**Interfaces:**
- `ScreeningStage` enum/string values `static`, `diagnostic`, `full`.
- `ScreeningContext(parent, profile, algorithm, plan, registry, search_space_version, diagnostic_evaluator, full_evaluator)`.
- `ScreeningResult(stage, status, cost, evidence_level, reasons, metrics, offline_simulation)`.
- `CandidateScreeningPipeline(root).run(context) -> tuple[ScreeningResult, ...]`.
- `CandidateScreeningPipeline.load(path) -> tuple[ScreeningResult, ...]`.

- [ ] **Step 1: Write failing tests for stage order, cost, and no promotion bypass.**

```python
def test_screening_stops_invalid_plan_before_diagnostic_or_full(tmp_path):
    calls = []
    context = screening_context(plan=invalid_plan(), full_evaluator=lambda _: calls.append("full"))
    results = CandidateScreeningPipeline(tmp_path).run(context)
    assert results[-1].status == "rejected"
    assert results[-1].stage == ScreeningStage.STATIC
    assert calls == []

def test_fake_diagnostic_is_not_real_promotion_evidence(tmp_path):
    results = CandidateScreeningPipeline(tmp_path).run(screening_context())
    assert all(result.evidence_level != "real_measured" for result in results[:2])
    assert results[-1].offline_simulation is True
```

Run: `.venv/bin/python -m pytest -q tests/unit/test_candidate_screening.py`
Expected: FAIL until stage records and scheduler exist.

- [ ] **Step 2: Implement static stage.**

Validate source through `validate_source`, plan through `CompressionPlan`,
operator presence in the supplied Registry view, duplicate source identity,
block/channel argument bounds and composition constraints. Record monotonic
elapsed time and `evidence_level="static"`.

- [ ] **Step 3: Implement CPU diagnostic stage.**

Use an injected diagnostic callback or Fake adapter; execute bounded forward,
finite-parameter, save/reload/hash checks and limited proxy metrics. Mark all
results `offline_simulation=true`, `evidence_level="diagnostic"`, and never
write them into the full evaluator result.

- [ ] **Step 4: Implement gated full stage and atomic resume.**

Call `full_evaluator` only when static and diagnostic stages pass. Persist
stage records atomically to `screening.json`; if a valid file exists, resume
from the last completed stage and do not repeat a completed full evaluation.

- [ ] **Step 5: Add config defaults and test interruption/recovery.**

Add `candidate_screening=False` and related feature flags to
`CompressionConfig.from_path`, while accepting existing YAML unchanged. Test
an exception after diagnostic, reload the persisted results, and ensure the
full callback runs exactly once after recovery.

- [ ] **Step 6: Run focused tests and commit.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_candidate_screening.py tests/unit/test_config.py`
Expected: PASS.

```bash
git add harness4h3/evolution harness4h3/config.py tests/unit/test_candidate_screening.py tests/unit/test_config.py
git commit -m "feat: add staged low-cost candidate screening"
```

### Task 6: Integrate optional profiling, improvement, search-space, and screening into EvolutionLoop

**Files:**
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/config.py`
- Modify: `harness4h3/evolution/__init__.py`
- Modify: `harness4h3/cli.py`
- Test: `tests/integration/test_evo_compression_evolution.py`
- Test: `tests/integration/test_evolutionary_compression.py`

**Interfaces:**
- `EvolutionLoop(..., improvement_enabled=False, search_space_manager=None, screening_pipeline=None, advanced_profiling=False)`.
- New round artifacts: `search-space.json`, `screening.json`, and optional `algorithm-validation.json`.
- Existing `EvolutionRound` fields remain compatible; `paths` may gain additive entries.
- CLI config fields map directly to the feature switches and never alter defaults when absent.

- [ ] **Step 1: Add failing integration test for the complete Fake feedback loop.**

```python
def test_fake_evolution_reads_profile_failure_improves_algorithm_and_reuses_skillhub(tmp_path):
    first = run_fake_round(tmp_path, feature_flags=all_new_features())
    second = run_fake_round(tmp_path, feature_flags=all_new_features())
    assert first.paths["profile"]
    assert Path(first.paths["algorithm"]).read_text() != Path(second.paths["algorithm"]).read_text()
    assert json.loads(Path(second.paths["decision"]).read_text())["algorithm_id"] != first.algorithm_id
    assert any(item["memory_kind"] in {"experience", "design_gene"} for item in skillhub_context(tmp_path))
```

Run: `.venv/bin/python -m pytest -q tests/integration/test_evo_compression_evolution.py`
Expected: FAIL until optional orchestration is wired.

- [ ] **Step 2: Add feature flags and active search-space initialization.**

Create a default `SearchSpaceVersion` artifact only inside the new output root
when `search_space_evolution=True`. Pass its description to Agent/SkillHub;
when false, use the existing fixed Registry and no new artifact.

- [ ] **Step 3: Add advanced profiling and algorithm improvement branches.**

When `advanced_profiling=True`, call `profile_model(..., calibration=..., advanced=True)`.
When `algorithm_improvement=True` and a prior AlgorithmCandidate exists, call
`AlgorithmAgent.improve`; otherwise preserve `generate`. Persist parent source
hash and validation records. Generation failure still follows existing failure
recording and resume behavior.

- [ ] **Step 4: Insert screening before full evaluator without changing promotion.**

Persist/reload `screening.json`. If static/diagnostic screening rejects, call
`_finish_failure` with the screening reason and record negative Memory. If full
screening runs, pass only its full evaluator record to the unchanged
`PromotionPolicy.decide` call. Never set `feasible=True` from screening alone.

- [ ] **Step 5: Integrate search-space proposals and SkillHub evidence.**

Allow the deterministic Fake path to create/validate/register the threshold
operator when enabled. Record `SearchSpaceVersion` and proposal IDs in the
round decision, algorithm metadata, DesignGene, and ExperienceRecord. Keep
real Wan configs at legacy defaults.

- [ ] **Step 6: Add CLI/config parsing and run integration tests.**

Run: `.venv/bin/python -m pytest -q tests/integration/test_evo_compression_evolution.py tests/integration/test_evolutionary_compression.py tests/unit/test_config.py`
Expected: PASS, including all prior assertions.

```bash
git add harness4h3/evolution harness4h3/config.py harness4h3/cli.py tests/integration/test_evo_compression_evolution.py tests/integration/test_evolutionary_compression.py
git commit -m "feat: wire optional EvoCompress self-evolution loop"
```

### Task 7: Add failure, duplicate, incompatibility, long-memory, and resume coverage

**Files:**
- Modify: `tests/unit/test_algorithm_improvement.py`
- Modify: `tests/unit/test_search_space.py`
- Modify: `tests/unit/test_candidate_screening.py`
- Modify: `tests/integration/test_evo_compression_evolution.py`
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/evolution/loop.py`

**Interfaces:** Existing interfaces from Tasks 3–6; no new public API.

- [ ] **Step 1: Test provider/generation failure persistence.**

Assert failed generation creates a durable algorithm/decision/experience
record, does not overwrite the parent algorithm, and resumes from the same
round directory.

- [ ] **Step 2: Test duplicate algorithm and invalid new operator.**

Assert duplicate source remains rejected, unknown operator remains unavailable
before registration, malformed proposal is stored as rejected, and Registry
names do not change after rejection.

- [ ] **Step 3: Test incompatible model and Wan alignment.**

Use a Fake adapter with invalid block/channel selection and an adapter exposing
`channel_pruning_requires_all_blocks=True`; assert validation rejects partial
plans and no child checkpoint is returned.

- [ ] **Step 4: Test interruption and long Memory.**

Write >100 bounded history records with repeated failures, run improvement and
screening, and assert prompt/context output stays within the existing compact
limits. Interrupt after a persisted static or diagnostic stage and assert
resume does not repeat prior work.

- [ ] **Step 5: Run all focused tests and commit.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_algorithm_improvement.py tests/unit/test_search_space.py tests/unit/test_candidate_screening.py tests/integration/test_evo_compression_evolution.py`
Expected: PASS.

```bash
git add harness4h3/evolution tests
git commit -m "test: cover EvoCompress failure and resume paths"
```

### Task 8: Documentation, full CPU verification, and delivery evidence

**Files:**
- Modify: `docs/evolutionary-compression.md`
- Modify: `harness4h3/evolution/README.md`
- Create: `docs/superpowers/reports/2026-10-15-evo-compression-search-space-evolution-report.md`

- [ ] **Step 1: Document feature flags and CPU Fake commands.**

Document how to enable each feature in a separate output root, how proposal
validation differs from execution registration, and why Fake/diagnostic results
cannot promote real Wan. Include exact commands for profiling, two-round Fake
evolution, and screening tests.

- [ ] **Step 2: Run static and full CPU checks.**

Run:

```bash
.venv/bin/python -m compileall -q harness4h3
.venv/bin/python -m pytest -q
git diff --check
```

Expected: compileall succeeds, all CPU tests pass, and diff check is clean.

- [ ] **Step 3: Verify isolation and branch state.**

From the original checkout, run `git status --short` and confirm its pre-existing
dirty file list is unchanged. In this worktree, verify `git status --short`,
`git log --oneline -8`, and that no new test/config writes exist outside the
worktree. Do not poll or modify remote experiment jobs beyond read-only process
observation.

- [ ] **Step 4: Write the final evidence report.**

Separate:

- implemented and CPU-tested behavior;
- features not tested on real Wan/GPU;
- features that must wait for the current four-arm experiment to finish;
- exact test commands/results, branch/worktree, commits, and known limitations.

- [ ] **Step 5: Commit documentation and report.**

```bash
git add docs/evolutionary-compression.md harness4h3/evolution/README.md docs/superpowers/reports/2026-10-15-evo-compression-search-space-evolution-report.md
git commit -m "docs: record EvoCompress self-evolution validation"
```

## Plan self-review

- Profiling requirements map to Tasks 1–2, including deterministic calibration,
  full-channel chunking, hierarchy, and evidence status.
- Recursive source improvement, parent hash, failure rationale, validation, and
  SkillHub reuse map to Task 3 and Task 6.
- Search-space version/proposal/reject/register and Fake-only new operator map
  to Task 4 and Task 6.
- Three-stage low-cost screening and PromotionPolicy independence map to Task 5
  and Task 6.
- Failure, duplicate, invalid operator, incompatibility, interruption, and long
  Memory requirements map to Task 7.
- Isolation, docs, static checks, full CPU tests, and explicit real-Wan limits
  map to Task 8.
- No step changes the original dirty checkout or running experiment roots.
