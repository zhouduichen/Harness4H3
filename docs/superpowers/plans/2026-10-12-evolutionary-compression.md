# Harness4H3 EvoCompress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution with task-by-task checkpoints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, restartable evolutionary compression pipeline for Wan-shaped DiT models while preserving the existing H3 and Wan student campaigns.

**Architecture:** Add a Wan structural adapter, compression operators, profiler, algorithm sandbox, and evolution services as independent modules. Reuse the existing evaluator, scheduler, archive, memory stores, controller provider, and no-shell process executor; use a fake Wan DiT for CPU contracts and an optional Diffusers adapter for real GPU smoke.

**Tech Stack:** Python 3.9+, dataclasses, PyTorch when available, optional Diffusers, JSON/YAML, existing `LocalProcessExecutor`, pytest.

## Global Constraints

- Do not reset, clean, or overwrite existing uncommitted files.
- Do not change existing H3 operator behavior or the current Wan student training campaign.
- Generated algorithms may return bounded JSON plans only; they may not select paths, devices, commands, imports, or evaluator settings.
- All structural changes must validate forward compatibility, finite parameters, save/reload, and parent checkpoint lineage before promotion.
- Fake measurements must carry `offline_simulation=true` and must never be reported as real Wan evidence.
- Real Diffusers/Torch imports are optional at module import time; CPU contract tests must run without them.
- Every failed operation must produce a stable failure type and remain available as experience.

---

## File Map

- Create: `harness4h3/adapters/wan/contracts.py` — model structure, block references, pruning configs, checkpoint artifacts, and adapter protocol.
- Create: `harness4h3/adapters/wan/fake.py` — deterministic Torch fake Wan DiT and structural adapter.
- Create: `harness4h3/adapters/wan/diffusers.py` — optional real Wan/Diffusers adapter with fail-closed layout discovery.
- Create: `harness4h3/adapters/wan/__init__.py` — public exports and backend factory.
- Create: `harness4h3/operators/wan/compression.py` — block and FFN channel pruning operators.
- Create: `harness4h3/operators/wan/__init__.py` — compression operator exports.
- Create: `harness4h3/profiler/layer_profiler.py` — parameter/FLOP/latency/memory profiling.
- Create: `harness4h3/profiler/sensitivity.py` — block, channel, and timestep sensitivity.
- Create: `harness4h3/profiler/report.py` — JSON-safe report model and persistence.
- Create: `harness4h3/profiler/__init__.py` — profiler exports.
- Create: `harness4h3/sandbox/policy.py` — AST/import/path policy checks.
- Create: `harness4h3/sandbox/runner.py` — fixed JSON runner using `LocalProcessExecutor`.
- Create: `harness4h3/sandbox/__init__.py` — sandbox exports.
- Create: `harness4h3/evolution/candidate.py` — algorithm candidate and compression plan records.
- Create: `harness4h3/evolution/store.py` — atomic candidate source/metadata persistence.
- Create: `harness4h3/evolution/skill_hub.py` — architecture-aware experience retrieval.
- Create: `harness4h3/evolution/agent.py` — deterministic seed agent and structured-provider adapter.
- Create: `harness4h3/evolution/policy.py` — bounded exploit/explore selection policy.
- Create: `harness4h3/evolution/loop.py` — one-round compression evolution orchestration.
- Create: `harness4h3/evolution/__init__.py` — evolution exports.
- Modify: `harness4h3/cli.py` — add profile/compress/evolve command dispatch without changing existing commands.
- Verify: `pyproject.toml` package discovery includes the new `harness4h3.*` packages; do not change dependency versions in this feature.
- Create: `tests/unit/test_wan_adapter_compression.py` — fake adapter and structural operator contracts.
- Create: `tests/unit/test_wan_profiler.py` — profiling and report contracts.
- Create: `tests/unit/test_algorithm_sandbox.py` — source policy and process execution contracts.
- Create: `tests/unit/test_algorithm_evolution.py` — candidate lineage, retrieval, loop, and promotion contracts.
- Create: `tests/integration/test_evolutionary_compression.py` — fake end-to-end CLI/service smoke.

## Interfaces shared by all tasks

```python
class WanModelAdapter(Protocol):
    architecture_name: str
    def load_model(self, checkpoint: str, *, device: str = "cpu", dtype: str = "float32") -> Any: ...
    def inspect(self, model: Any) -> ModelStructure: ...
    def get_blocks(self, model: Any) -> tuple[BlockRef, ...]: ...
    def remove_blocks(self, model: Any, block_ids: Sequence[str]) -> StructuralChange: ...
    def prune_channels(self, model: Any, config: ChannelPruningConfig) -> StructuralChange: ...
    def save_model(self, model: Any, output_dir: Path) -> CheckpointArtifact: ...
    def reload_model(self, artifact: CheckpointArtifact) -> Any: ...

class CompressionOperator(Protocol):
    name: str
    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact,
                plan: Mapping[str, Any], output_dir: Path) -> CompressionResult: ...
```

### Task 1: Add adapter contracts and deterministic fake Wan model

**Files:**
- Create: `harness4h3/adapters/wan/contracts.py`
- Create: `harness4h3/adapters/wan/fake.py`
- Create: `harness4h3/adapters/wan/__init__.py`
- Test: `tests/unit/test_wan_adapter_compression.py`

**Interfaces:**
- Consumes: a checkpoint identifier and optional Torch runtime.
- Produces: `ModelStructure`, `BlockRef`, `ChannelPruningConfig`, `StructuralChange`, `CheckpointArtifact`, `FakeWanDiTAdapter`, and `build_wan_adapter(backend, ...)`.

- [ ] **Step 1: Write failing contract tests.**

```python
def test_fake_adapter_discovers_blocks_and_ffn_width(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline", device="cpu")
    structure = adapter.inspect(model)
    assert structure.block_count == 4
    assert structure.ffn_width == 16
    assert tuple(item.block_id for item in adapter.get_blocks(model)) == ("block.0", "block.1", "block.2", "block.3")

def test_fake_adapter_save_reload_preserves_forward_shape(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline")
    artifact = adapter.save_model(model, tmp_path / "child")
    reloaded = adapter.reload_model(artifact)
    assert tuple(reloaded(torch.zeros(2, 8)).shape) == (2, 8)
```

- [ ] **Step 2: Run `pytest -q tests/unit/test_wan_adapter_compression.py` and verify the imports fail.**

- [ ] **Step 3: Implement the contracts and fake module.**

`contracts.py` must define frozen dataclasses with JSON-safe `to_dict()` methods:

```python
@dataclass(frozen=True)
class ModelStructure:
    architecture: str
    block_count: int
    hidden_size: int
    ffn_width: int
    parameter_count: int
    checkpoint_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class BlockRef:
    block_id: str
    index: int
    parameter_count: int
    module_path: str

@dataclass(frozen=True)
class ChannelPruningConfig:
    block_ids: tuple[str, ...]
    retained_indices: tuple[int, ...]
    projection_name: str = "ffn"

@dataclass(frozen=True)
class CheckpointArtifact:
    path: str
    sha256: str
    architecture: str
    structure: ModelStructure
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

The fake adapter must use `torch.nn.ModuleList` blocks and paired `up_proj`/`down_proj` linear layers. It must write `config.json` and `model.pt`, hash the files deterministically, and reconstruct from those files. The factory must reject unknown backends with `unsupported_adapter_backend`.

- [ ] **Step 4: Run focused tests and `python -m compileall -q harness4h3/adapters`.**

- [ ] **Step 5: Commit only the new adapter files and focused tests.**

```bash
git add harness4h3/adapters/wan tests/unit/test_wan_adapter_compression.py
git commit -m "feat: add Wan compression adapter contracts"
```

### Task 2: Implement structural compression operators

**Files:**
- Create: `harness4h3/operators/wan/compression.py`
- Create: `harness4h3/operators/wan/__init__.py`
- Modify: `tests/unit/test_wan_adapter_compression.py`

**Interfaces:**
- Consumes: `WanModelAdapter`, `CheckpointArtifact`, and JSON plans.
- Produces: `CompressionResult`, `BlockPruningOperator`, `FFNChannelPruningOperator`, and `build_wan_compression_registry()`.

- [ ] **Step 1: Add failing tests for valid and invalid block/channel plans.**

```python
def test_block_pruning_updates_structure_and_parent_digest(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = BlockPruningOperator().execute(adapter, parent, {"block_ids": ["block.1"]}, tmp_path / "child")
    assert result.status == "success"
    assert result.structural_change["retained_blocks"] == 3
    assert result.parent_checkpoint_sha256 == parent.sha256

def test_channel_pruning_rejects_empty_retained_indices(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = FFNChannelPruningOperator().execute(adapter, parent, {"block_ids": ["block.0"], "retained_indices": []}, tmp_path / "child")
    assert result.status == "failed"
    assert result.failure_type == "invalid_channel_selection"
```

- [ ] **Step 2: Run the focused test and verify the operator imports fail.**

- [ ] **Step 3: Implement operators with copy-on-write model loading.**

Each operator must load the parent artifact, validate the complete plan, mutate a copy, run a no-grad shape/finite check, save the child, and return a `CompressionResult`. `BlockPruningOperator` must reject deleting all blocks. `FFNChannelPruningOperator` must apply the same retained index set to `up_proj` output rows and `down_proj` input columns for every selected block. The registry must expose only `prune_blocks` and `prune_ffn_channels` and must not alter the existing H3 registry.

- [ ] **Step 4: Run `pytest -q tests/unit/test_wan_adapter_compression.py` and compile the new package.**

- [ ] **Step 5: Commit the structural operators.**

### Task 3: Add profiling and JSON reports

**Files:**
- Create: `harness4h3/profiler/layer_profiler.py`
- Create: `harness4h3/profiler/sensitivity.py`
- Create: `harness4h3/profiler/report.py`
- Create: `harness4h3/profiler/__init__.py`
- Create: `tests/unit/test_wan_profiler.py`

**Interfaces:**
- Consumes: adapter/model, optional `forward_fn`, optional `timestep_fn`, and a measurement budget.
- Produces: `ProfileReport`, `profile_layers()`, `measure_block_sensitivity()`, `measure_channel_sensitivity()`, and `measure_timestep_sensitivity()`.

- [ ] **Step 1: Write tests for JSON shape and unavailable measurements.**

```python
def test_profile_report_contains_layer_sensitivity_channel_and_timestep_sections():
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    report = profile_model(adapter, adapter.load_model("fake://baseline"), seed=7)
    payload = report.to_dict()
    assert set(payload) >= {"model", "layers", "block_sensitivity", "channel_sensitivity", "timestep_sensitivity"}
    assert payload["provenance"]["offline_simulation"] is True

def test_missing_real_measurement_is_explicitly_unavailable():
    report = profile_model(FakeWanDiTAdapter(), FakeWanDiTAdapter().load_model("fake://baseline"), measure_latency=False)
    assert report.provenance["latency_available"] is False
```

- [ ] **Step 2: Implement deterministic layer statistics and leave-one-out sensitivity.**

The profiler must use `time.perf_counter()` only when requested, never label an absent measurement as zero, and use a stable seeded input for fake runs. It must cap channel probes by `max_channel_probes`. Timestep sections must always include `high_noise`, `middle`, and `low_noise` records, with an `available` flag.

- [ ] **Step 3: Persist reports atomically and run focused tests.**

`ProfileReport.write_json(path)` must write to a sibling temporary file, fsync, and replace the target. `ProfileReport.from_dict()` must reject missing model digest or malformed sections.

- [ ] **Step 4: Commit profiling files and tests.**

### Task 4: Add algorithm candidate records and sandbox execution

**Files:**
- Create: `harness4h3/sandbox/policy.py`
- Create: `harness4h3/sandbox/runner.py`
- Create: `harness4h3/sandbox/__init__.py`
- Create: `harness4h3/evolution/candidate.py`
- Create: `harness4h3/evolution/store.py`
- Create: `tests/unit/test_algorithm_sandbox.py`

**Interfaces:**
- Consumes: source text implementing `propose(profile, history)`.
- Produces: `AlgorithmCandidate`, `CompressionPlan`, `SandboxResult`, `AlgorithmStore`, and `AlgorithmSandbox.run()`.

- [ ] **Step 1: Write tests for accepted source, forbidden import, and invalid output.**

```python
def test_sandbox_accepts_bounded_plan(tmp_path):
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}\n"
    result = AlgorithmSandbox(tmp_path).run(source, {"layers": []}, [])
    assert result.status == "success"
    assert result.plan.operator == "prune_blocks"

def test_sandbox_rejects_filesystem_import(tmp_path):
    source = "import os\ndef propose(profile, history):\n    return {}\n"
    result = AlgorithmSandbox(tmp_path).run(source, {}, [])
    assert result.status == "failed"
    assert result.failure_type == "sandbox_policy"
```

- [ ] **Step 2: Implement AST policy and fixed runner.**

The policy must allow only the four documented standard-library modules, reject `Import`, `ImportFrom`, `Call` targets named `eval`, `exec`, `compile`, `__import__`, `open`, `input`, and reject attribute roots named `os`, `sys`, `subprocess`, `socket`, `pathlib`, `shutil`, `requests`, or `torch`. The runner must use an argv tuple and `LocalProcessExecutor(shell=False)`, pass JSON through files/stdin, cap output at 64 KiB, and parse only the exact `operator`/`args` plan shape.

- [ ] **Step 3: Implement atomic candidate source/metadata storage.**

`AlgorithmStore.create()` assigns IDs `A0001`, `A0002`, etc., stores source in `candidates/<id>.py`, metadata in `candidates/<id>.json`, and rejects duplicate source digests. `AlgorithmCandidate` records `parent_id`, `source_sha256`, `status`, `experiment_ids`, and failure summaries.

- [ ] **Step 4: Run sandbox tests and commit.**

### Task 5: Add Skill Hub, agent normalization, and search policy

**Files:**
- Create: `harness4h3/evolution/skill_hub.py`
- Create: `harness4h3/evolution/agent.py`
- Create: `harness4h3/evolution/policy.py`
- Create: `tests/unit/test_algorithm_evolution.py`

**Interfaces:**
- Consumes: `ExperienceStore`, `DesignGeneStore`, profile JSON, target constraints, and an optional structured controller.
- Produces: `SkillHub.search()`, `AlgorithmAgent.generate()`, `SearchPolicy.choose()`, and normalized `AlgorithmCandidate` source.

- [ ] **Step 1: Add tests for architecture-aware retrieval and deterministic explore/exploit.**

```python
def test_skill_hub_filters_by_architecture_and_operator_family(tmp_path):
    hub = SkillHub(tmp_path / "experience.jsonl", tmp_path / "genes.jsonl")
    hub.record(_experience("wan", "prune_blocks", reward=0.4))
    hub.record(_experience("h3", "prune_blocks", reward=0.9))
    matches = hub.search(architecture="wan", operator="prune_blocks", limit=5)
    assert len(matches) == 1
    assert matches[0]["architecture"] == "wan"

def test_search_policy_uses_seeded_explore_ratio():
    policy = SearchPolicy(exploit_ratio=0.7, seed=3)
    choices = [policy.choose(has_best=True) for _ in range(10)]
    assert all(item in {"exploit", "explore"} for item in choices)
    assert policy.to_dict()["exploit_ratio"] == 0.7
```

- [ ] **Step 2: Implement deterministic fallback generation.**

The offline agent must generate a real Python function from a fixed template using the profile's highest-sensitivity removable block and must attach a hypothesis, parent ID, and source digest. The provider-backed agent must accept only structured fields `source`, `hypothesis`, and `parent_id`; any path, command, device, or unknown field is rejected.

- [ ] **Step 3: Implement Skill Hub ranking.**

Rank exact architecture and operator matches before partial matches, then by reward descending and newest record. Failed records are returned with a `failure_reason` so the agent can avoid repeated fingerprints.

- [ ] **Step 4: Run focused tests and commit.**

### Task 6: Implement the restartable evolution loop

**Files:**
- Create: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/__init__.py`
- Create: `tests/integration/test_evolutionary_compression.py`

**Interfaces:**
- Consumes: adapter, compression registry, profiler, algorithm agent, sandbox, evaluator, `ModelStore`, `ExperienceStore`, and `SearchPolicy`.
- Produces: `EvolutionRound`, `EvolutionLoop.run_round()`, and durable round artifacts.

- [ ] **Step 1: Write a fake end-to-end test.**

```python
def test_evolution_loop_records_failed_and_successful_algorithm_rounds(tmp_path):
    loop = make_fake_loop(tmp_path)
    first = loop.run_round()
    second = loop.run_round()
    assert first.algorithm_id.startswith("A")
    assert second.parent_model_id in {"M0000", "M0001"}
    assert (tmp_path / "rounds" / first.round_id / "decision.json").exists()
    assert list(ExperienceStore(tmp_path / "experiences.jsonl").read())
```

- [ ] **Step 2: Implement one-round ordering and fail-closed promotion.**

The loop must write `profile.json`, `algorithm.json`, `sandbox.json`, `compression.json`, `evaluation.json`, and `decision.json` atomically per round. It must create a child `ModelCandidate` only after a valid checkpoint artifact exists; call the existing evaluator with explicit fake provenance in fake mode; append `ExperienceRecord` exactly once by source URI/hash; and call `ModelStore.set_active()` only when evaluation is feasible and no critical regression exists.

- [ ] **Step 3: Implement resume and duplicate fingerprint handling.**

On startup, complete round directories are loaded. A partial round is marked failed and never retried with the same algorithm/source/parent fingerprint. The loop resumes from the active model and retains all prior failures in Skill Hub.

- [ ] **Step 4: Run the integration smoke, the full unit suite, and commit.**

### Task 7: Add CLI commands and configuration

**Files:**
- Modify: `harness4h3/cli.py`
- Create: `configs/evolutionary-compression.fake.yaml`
- Create: `configs/evolutionary-compression.wan.example.yaml`
- Modify: `tests/test_cli.py`
- Modify: `tests/integration/test_evolutionary_compression.py`

**Interfaces:**
- Consumes: YAML configuration and the public adapter/evolution services.
- Produces: `profile-wan`, `compress-wan`, and `evolve-compression` commands with JSON summaries and non-zero exit codes on failed gates.

- [ ] **Step 1: Add CLI parser tests.**

```python
def test_profile_wan_fake_cli_writes_json(tmp_path):
    result = run_cli("profile-wan", "--config", str(fake_config), "--output", str(tmp_path / "profile.json"))
    assert result.returncode == 0
    assert json.loads((tmp_path / "profile.json").read_text())["model"]["architecture"] == "WanFakeDiT"
```

- [ ] **Step 2: Implement trusted config parsing.**

The config must validate backend, absolute/contained output directories, positive budgets, adapter backend, and evaluator mode. Generated algorithm source cannot modify any of these values.

- [ ] **Step 3: Add fake and real-example configs without embedding machine-specific paths.**

The fake config must run on CPU with a temporary output root. The real example must use explicit placeholder-free environment-independent configuration fields and state that the checkpoint path is supplied by the operator; it must not claim checked-in real evidence.

- [ ] **Step 4: Run `python -m harness4h3 profile-wan --help`, fake CLI smoke, and the full test suite.**

### Task 8: Real Wan smoke and final verification

**Files:**
- Modify: `docs/quickstart.md`
- Create: `docs/evolutionary-compression.md`
- Create: `tests/integration/test_real_wan_adapter_contract.py`

- [ ] **Step 1: Add a skip-safe real adapter contract test.**

The test skips only when Diffusers/Torch or an explicitly supplied checkpoint is unavailable. When enabled, it must inspect the real block list, execute a one-block prune in an isolated directory, reload the child, run one shape/finite forward, and assert parent/child digests differ.

- [ ] **Step 2: Document fake verification and real GPU smoke commands.**

- [ ] **Step 3: Run the full CPU suite, compileall, fake profile/compress/evolve commands, and the real smoke when the required environment is present.**

- [ ] **Step 4: Perform a requirement-by-requirement audit against the design and keep the goal active for any missing real evidence.**

## Plan self-review

- The design requirements map to Tasks 1–3 (Wan foundation and profiling), Task 4 (sandbox), Tasks 5–6 (algorithm and experience evolution), Task 7 (operational entrypoints), and Task 8 (deployment evidence).
- No task relies on H3 architecture mutation or changes existing uncommitted student files.
- All shared names are defined above or in the task that produces them: `WanModelAdapter`, `CompressionResult`, `AlgorithmCandidate`, `CompressionPlan`, `SandboxResult`, `ProfileReport`, `EvolutionRound`, and `SearchPolicy`.
- Fake and real evidence are separated by explicit provenance and gates.
