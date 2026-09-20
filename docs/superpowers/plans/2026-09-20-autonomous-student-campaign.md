# Autonomous H3 Student Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable autonomous campaign in which a local LLM proposes legal 1B–2B video Student architectures, the Harness validates/compiles them, a fixed remote worker distills from real H3, and real video/quality/VRAM evidence feeds the next round.

**Architecture:** Add a focused `harness4h3.student` package. `StudentProposal` is a versioned JSON contract; `StudentCompiler` constructs only registered modules on `meta`/fake tensors and emits a signed compile manifest; `StudentCampaign` drives the proposal→compile→SSH worker→evaluation→experience loop. Existing H3 adapters, ComfyUI benchmark adapters, SSH transport, controller providers, experience memory, and checkpoint retention remain the integration boundary.

**Tech Stack:** Python 3.9+, PyTorch 2.2+, PyYAML, safetensors, existing `harness4h3` SSH/ComfyUI infrastructure, JSONL append-only records, pytest.

## Global Constraints

- The LLM emits JSON data only; it never emits Python, shell, executable paths, or arbitrary remote commands.
- The compiler accepts only registered video model modules and rejects unknown fields, invalid topology, invalid shapes, non-finite values, and parameter counts outside 1B–2B for production targets.
- A compile/validation failure is recorded before SSH launch; no invalid proposal reaches a worker.
- Real H3, real Student checkpoint mutation, real video decoding, quality, latency, and peak VRAM are required for remote acceptance; TinyH3/fake backends are contract tests only.
- Remote commands are fixed argv vectors from trusted YAML; proposal values travel in immutable JSON manifests.
- Metadata and failure evidence are append-only; only active/best/direct-parent/in-flight weight payloads are retained by default.
- Existing unrelated working-tree changes belong to the user and must not be reverted or staged.
- Every task ends with a focused test command and a commit containing only that task's files.

## File map

Create the following focused units:

- `harness4h3/student/__init__.py` — public student interfaces.
- `harness4h3/student/proposal.py` — typed proposal/target contracts and validation errors.
- `harness4h3/student/model.py` — registered video latent DiT modules and deterministic parameter counting.
- `harness4h3/student/compiler.py` — meta/fake compilation, shape report, memory estimate, and manifest digest.
- `harness4h3/student/worker.py` — teacher/student training contract, changed-child verification, and result record.
- `harness4h3/student/evaluator.py` — decoded-video validity, quality/hardware result normalization, and typed failures.
- `harness4h3/student/campaign.py` — bounded local-LLM context, round state, proposal deduplication, and resumable loop.
- `harness4h3/student/retention.py` — safe local/remote candidate retention decision logic.
- `tools/student_train_worker.py` — fixed remote entrypoint with JSON result output.
- `configs/student-campaign.example.yaml` — trusted campaign/SSH/teacher/evaluator configuration.
- `tests/unit/test_student_proposal.py`, `tests/unit/test_student_compiler.py`, `tests/unit/test_student_campaign.py` — offline contracts.
- `tests/training/test_student_worker.py` — fake teacher/student training and checkpoint evidence.
- `tests/integration/test_student_campaign.py` — two-round proposal→failure→revision loop.

Modify only the following existing integration points:

- `harness4h3/controller/provider.py` — add a strict StudentProposal request schema/provider method without weakening the existing ExperimentPlan schema.
- `harness4h3/remote/config.py` — parse the student campaign section and fixed worker/evaluator paths.
- `harness4h3/cli.py` — add `student-campaign validate`, `student-campaign compile`, and `student-campaign run` commands.
- `pyproject.toml` — expose the new package and keep PyTorch/safetensors under the existing `training` extra.
- `README.md`, `docs/quickstart.md`, `docs/operator-contract.md` — document the new primary path and its real-evidence boundary.

---

### Task 1: Add the Student proposal contract and target validation

**Files:**
- Create: `harness4h3/student/__init__.py`
- Create: `harness4h3/student/proposal.py`
- Create: `tests/unit/test_student_proposal.py`

**Interfaces:**
- Consumes: JSON mappings from the configured local LLM.
- Produces: `ArchitectureSpec`, `TrainingSpec`, `DeploymentSpec`, `StudentProposal`, `StudentTarget`, `ProposalValidationError`, and `canonical_digest()` for Tasks 2–6.

- [ ] **Step 1: Write failing contract tests**

```python
def test_two_different_proposals_are_valid_and_digest_stable():
    first = StudentProposal.from_dict(valid_payload(hidden_size=2048, depth=24))
    second = StudentProposal.from_dict(valid_payload(hidden_size=1792, depth=32))
    target = StudentTarget(min_params=1_000_000_000, max_params=2_000_000_000)
    first_report = first.validate(target)
    second_report = second.validate(target)
    assert first_report.errors == ()
    assert second_report.errors == ()
    assert first.digest != second.digest
    assert StudentProposal.from_dict(first.to_dict()).digest == first.digest


def test_unknown_fields_and_invalid_head_divisibility_are_rejected():
    payload = valid_payload(hidden_size=2000, depth=24)
    payload["architecture"]["unknown"] = 1
    with pytest.raises(ProposalValidationError, match="unknown"):
        StudentProposal.from_dict(payload)
    payload = valid_payload(hidden_size=2048, depth=24)
    payload["architecture"]["num_heads"] = 30
    assert any("divisible" in error for error in StudentProposal.from_dict(payload).validate(StudentTarget()).errors)


def test_training_and_deployment_bounds_are_strict():
    payload = valid_payload()
    payload["training"]["learning_rate"] = float("nan")
    with pytest.raises(ProposalValidationError, match="finite"):
        StudentProposal.from_dict(payload)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest -q tests/unit/test_student_proposal.py`

Expected: FAIL because `harness4h3.student.proposal` does not yet exist.

- [ ] **Step 3: Implement the immutable data contract**

Implement frozen dataclasses with these exact fields:

```python
@dataclass(frozen=True)
class StudentTarget:
    min_params: int = 1_000_000_000
    max_params: int = 2_000_000_000
    max_peak_memory_gb: float = 72.0
    latent_channels: int = 24
    latent_frames: int = 5
    latent_height: int = 32
    latent_width: int = 32

@dataclass(frozen=True)
class ArchitectureSpec:
    family: str
    latent_channels: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    spatial_patch: int
    temporal_patch: int
    temporal_layers: Tuple[int, ...]
    conditioning: str
    norm: str
    activation: str

@dataclass(frozen=True)
class StudentProposal:
    schema_version: int
    proposal_id: str
    parent_proposal_id: Optional[str]
    teacher: Mapping[str, str]
    architecture: ArchitectureSpec
    training: TrainingSpec
    deployment: DeploymentSpec
```

`from_dict` must reject missing top-level keys, unknown keys, bool-as-number values, non-finite floats, empty IDs, unsupported enum values, invalid layer indices, and malformed mappings. `validate(target)` must return a deterministic `ValidationReport` containing `errors`, `estimated_params`, `estimated_peak_memory_gb`, and `duplicate_key`; it must not silently clamp or repair an LLM proposal. `to_dict()` must use JSON-safe lists instead of tuples, and `digest` must hash canonical sorted JSON.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest -q tests/unit/test_student_proposal.py`

Expected: PASS with at least three tests.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/student/__init__.py harness4h3/student/proposal.py tests/unit/test_student_proposal.py
git commit -m "feat: add autonomous student proposal contract"
```

### Task 2: Compile legal proposals into registered video Student graphs

**Files:**
- Create: `harness4h3/student/model.py`
- Create: `harness4h3/student/compiler.py`
- Create: `tests/unit/test_student_compiler.py`

**Interfaces:**
- Consumes: `StudentProposal`, `StudentTarget` from Task 1.
- Produces: `build_student(proposal, device)`, `StudentCompiler.compile(proposal, output_dir)`, `CompileManifest`, `CompileError`, and exact parameter/shape/memory evidence for the campaign.

- [ ] **Step 1: Write failing compiler tests**

```python
def test_production_proposal_compiles_on_meta_without_allocating_weights(tmp_path):
    proposal = StudentProposal.from_dict(valid_payload(hidden_size=2048, depth=24))
    manifest = StudentCompiler(StudentTarget()).compile(proposal, tmp_path)
    assert 1_000_000_000 <= manifest.parameter_count <= 2_000_000_000
    assert manifest.graph_status == "compiled"
    assert manifest.output_shape == manifest.input_shape
    assert Path(manifest.path).is_file()


def test_invalid_memory_budget_fails_before_graph_build(tmp_path):
    target = StudentTarget(max_peak_memory_gb=1.0)
    with pytest.raises(CompileError, match="peak memory"):
        StudentCompiler(target).compile(StudentProposal.from_dict(valid_payload()), tmp_path)


def test_materially_different_architectures_produce_different_graph_digests(tmp_path):
    first = StudentCompiler(StudentTarget()).compile(
        StudentProposal.from_dict(valid_payload(hidden_size=2048, depth=24)), tmp_path / "a"
    )
    second = StudentCompiler(StudentTarget()).compile(
        StudentProposal.from_dict(valid_payload(hidden_size=1792, depth=32)), tmp_path / "b"
    )
    assert first.graph_digest != second.graph_digest
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest -q tests/unit/test_student_compiler.py`

Expected: FAIL because no registered Student graph or compiler exists.

- [ ] **Step 3: Implement the registered graph**

Implement only these modules in `model.py`: `VideoPatchEmbed`, `RMSNorm`, `SpatialDiTBlock`, `TemporalDiTBlock`, `AdaNormZero`, `ConditionProjector`, and `VideoUnpatchify`. `VideoLatentDiT` must construct blocks from the proposal, use `ModuleList`, keep temporal-layer selection static, accept `(video_latents, conditioning, timestep)`, and return a tensor with the same latent shape. No `eval`, imports, callbacks, or dynamic module names from proposal data are allowed.

The block dimensions must make the exact constructed count authoritative. With a standard qkv/out attention plus MLP, the legal baseline is `hidden_size=2048`, `depth=24`, `mlp_ratio=4.0`, and `num_heads=32`; the compiler must calculate the actual count rather than trusting an LLM estimate.

- [ ] **Step 4: Implement meta/fake compilation and immutable manifest**

`StudentCompiler.compile` must:

1. call proposal validation and fail on any error;
2. construct the model under `torch.device("meta")`;
3. run a meta forward with `StudentTarget` latent and conditioning shapes;
4. run `torch.fx.symbolic_trace` on the static graph;
5. calculate exact parameters and a conservative bf16 + AdamW peak-memory estimate;
6. serialize `compile_manifest.json` with proposal digest, graph digest, parameter count, input/output shapes, memory estimate, compiler version, and JSON-safe module list;
7. atomically write the manifest and refuse to overwrite a different digest.

For CPU smoke tests, expose `build_smoke_student(proposal, scale=0.125)` using the same registered classes and topology with reduced dimensions; this is explicitly marked `smoke_only` and is never accepted as 1B–2B evidence.

- [ ] **Step 5: Run tests and compile check**

Run: `python -m pytest -q tests/unit/test_student_proposal.py tests/unit/test_student_compiler.py && python -m compileall -q harness4h3/student`

Expected: PASS; the production graph is built on `meta`, so the test does not allocate 1B parameters.

- [ ] **Step 6: Commit**

```bash
git add harness4h3/student/model.py harness4h3/student/compiler.py tests/unit/test_student_compiler.py
git commit -m "feat: compile declarative student video graphs"
```

### Task 3: Add a fixed H3→Student training worker and checkpoint evidence

**Files:**
- Create: `harness4h3/student/worker.py`
- Create: `tools/student_train_worker.py`
- Create: `tests/training/test_student_worker.py`
- Modify: `h3_training/adapters/base.py` only if a narrow adapter method is required by the new worker.

**Interfaces:**
- Consumes: `CompileManifest`, trusted teacher checkpoint path, fixed training config, and real/fake `DenoisingModelAdapter`.
- Produces: `TrainingResult`, `TrainingFailureRecord`, changed child checkpoint, result JSON, and `python tools/student_train_worker.py --manifest ... --teacher ... --output ... --result ...`.

- [ ] **Step 1: Write failing worker contract tests**

```python
def test_worker_writes_changed_child_and_training_evidence(tmp_path, fake_teacher):
    manifest = compile_smoke_manifest(tmp_path)
    result = StudentTrainWorker(fake_teacher).run(manifest, tmp_path / "child", max_steps=2)
    assert result.status == "success"
    assert Path(result.child_checkpoint).is_file()
    assert result.parent_sha256 != result.child_sha256
    assert result.optimizer_steps >= 1
    assert result.offline_simulation is True


def test_worker_refuses_manifest_digest_mismatch(tmp_path, fake_teacher):
    manifest = compile_smoke_manifest(tmp_path)
    manifest_path = Path(manifest.path)
    payload = json.loads(manifest_path.read_text())
    payload["proposal"]["architecture"]["depth"] += 1
    manifest_path.write_text(json.dumps(payload))
    result = StudentTrainWorker(fake_teacher).run(manifest, tmp_path / "child", max_steps=1)
    assert result.failure_code == "manifest_digest_mismatch"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest -q tests/training/test_student_worker.py`

Expected: FAIL because the worker and fixed CLI entrypoint do not exist.

- [ ] **Step 3: Implement the narrow worker contract**

`StudentTrainWorker.run` must reload and verify the compile manifest, load the teacher through an injected adapter, build the Student, run a fixed latent/velocity distillation loop with the existing `TrainerEngine`, and save a child checkpoint atomically. The worker must record parent/child SHA-256, parameter count, proposal/compiler digests, steps, initial/final loss, gradient norm, wall time, CUDA peak memory, and `offline_simulation`.

The fake adapter is used only in tests. The real path must use `RealMiniMaxH3Adapter` and fail with `h3_adapter_unavailable` when the declared ComfyUI/H3 loader is not present. The worker must never copy the teacher checkpoint as a successful child; compare hashes and at least one trainable tensor before returning success.

- [ ] **Step 4: Implement the fixed JSON CLI entrypoint**

The CLI parses only trusted file paths and bounded numeric values, constructs the worker from a configured adapter name, and always writes one result JSON even on failure:

```bash
python tools/student_train_worker.py \
  --manifest /data/campaign/student_0001/compile_manifest.json \
  --teacher /data/models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --output /data/campaign/student_0001 \
  --result /data/campaign/student_0001/training-result.json
```

Exit 0 is reserved for a verified changed child; all typed failures exit non-zero without deleting the result record.

- [ ] **Step 5: Run worker tests and CLI help**

Run: `python -m pytest -q tests/training/test_student_worker.py && python tools/student_train_worker.py --help`

Expected: PASS and help lists only manifest/teacher/output/result/config arguments.

- [ ] **Step 6: Commit**

```bash
git add harness4h3/student/worker.py tools/student_train_worker.py tests/training/test_student_worker.py
git commit -m "feat: add fixed H3 student training worker"
```

### Task 4: Normalize real video evaluation, experience, and retention

**Files:**
- Create: `harness4h3/student/evaluator.py`
- Create: `harness4h3/student/retention.py`
- Create: `tests/unit/test_student_evaluator.py`
- Create: `tests/unit/test_student_retention.py`
- Modify: `harness4h3/memory/experience.py` only to add a `student_proposal` source kind if existing validation rejects it.

**Interfaces:**
- Consumes: generated video paths, quality evaluator output, hardware sampler output, training result, and candidate identities.
- Produces: `StudentEvaluation`, stable failure codes, `StudentExperienceRecord`, and a retention decision with safe deletion targets.

- [ ] **Step 1: Write failing evaluation/retention tests**

```python
def test_invalid_video_is_hard_failure(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not-video")
    result = StudentEvaluator().evaluate(path, quality={"score": 0.9}, hardware={"peak_vram_gb": 4.0})
    assert result.valid is False
    assert result.failure_code == "video_decode_failed"
    assert result.promotable is False


def test_retention_preserves_active_parent_and_deletes_only_rejected_child(tmp_path):
    parent = tmp_path / "M0000.safetensors"
    child = tmp_path / "M0001.safetensors"
    parent.write_bytes(b"parent")
    child.write_bytes(b"child")
    decision = retain_after_evaluation(child, "M0001", outcome="rejected", protected={parent})
    assert decision.delete_path == child
    assert parent.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/unit/test_student_evaluator.py tests/unit/test_student_retention.py`

Expected: FAIL because the new evaluator and retention contracts do not exist.

- [ ] **Step 3: Implement validity and metric normalization**

Use OpenCV/imageio already available to decode a bounded sample of frames. Require a readable container, positive frame count/dimensions/duration, finite pixel statistics, black-frame ratio at or below the configured threshold, and a completed generation within timeout. Merge external quality and hardware measurements without allowing them to override validity. Return explicit codes: `video_missing`, `video_decode_failed`, `video_empty`, `video_nonfinite`, `video_black_frames`, `generation_timeout`, `generation_oom`, `quality_regression`, and `evaluation_ok`.

- [ ] **Step 4: Implement append-only experience and safe retention decision**

Experience records must include proposal/compiler/teacher digests, parent/child paths, training metrics, evaluation metrics, outcome, failure code, diagnosis, and next-round hints. Raw video/log fields are URI references. Retention must protect active, direct parent, best feasible, and in-flight paths; it may return a deletion target only after evidence has been fsynced by the caller.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/unit/test_student_evaluator.py tests/unit/test_student_retention.py`

Expected: PASS.

```bash
git add harness4h3/student/evaluator.py harness4h3/student/retention.py tests/unit/test_student_evaluator.py tests/unit/test_student_retention.py
git commit -m "feat: record student video evidence and retention"
```

### Task 5: Add the autonomous proposal→train→evaluate→revise campaign

**Files:**
- Create: `harness4h3/student/campaign.py`
- Create: `tests/unit/test_student_campaign.py`
- Create: `tests/integration/test_student_campaign.py`
- Modify: `harness4h3/controller/provider.py`

**Interfaces:**
- Consumes: goal text, `StudentTarget`, strict local LLM provider, compiler, trusted worker launcher, evaluator, and retention.
- Produces: `StudentCampaign.run(max_rounds)`, durable `campaign-events.jsonl`, `resume.json`, bounded next-round context, and `CampaignResult`.

- [ ] **Step 1: Write failing campaign tests**

```python
def test_two_round_campaign_passes_failure_to_revised_proposal(tmp_path):
    provider = SequenceStudentProvider([valid_payload(depth=24), valid_payload(depth=32)])
    worker = ScriptedWorker([Failure("video_decode_failed"), Success(video="ok.mp4")])
    campaign = StudentCampaign(provider, compiler, worker, evaluator, output_root=tmp_path)
    result = campaign.run(max_rounds=2)
    assert result.rounds_completed == 2
    assert result.status == "success"
    events = [json.loads(line) for line in (tmp_path / "campaign-events.jsonl").read_text().splitlines()]
    assert events[1]["failure_code"] == "video_decode_failed"
    assert "video_decode_failed" in events[2]["llm_context"]["failures"]


def test_duplicate_proposal_is_rejected_without_worker_launch(tmp_path):
    provider = SequenceStudentProvider([valid_payload(depth=24), valid_payload(depth=24)])
    worker = CountingWorker()
    result = StudentCampaign(provider, compiler, worker, evaluator, output_root=tmp_path).run(max_rounds=2)
    assert worker.calls == 1
    assert result.failure_code == "duplicate_proposal"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/unit/test_student_campaign.py tests/integration/test_student_campaign.py`

Expected: FAIL because no Student campaign runner or strict provider method exists.

- [ ] **Step 3: Add strict provider schema and bounded context**

Add `student_proposal_json_schema(target)` and `StudentProposalProvider.propose(context)`. The schema must require the exact proposal fields, set the proposal ID for the current round, disallow additional properties, constrain enum values, and require the architecture dimensions to be integers/numbers. The prompt must contain only the goal, target limits, teacher manifest summary, recent bounded evidence, and typed failures. The provider must parse one JSON object and surface malformed output as `proposal_invalid` with no fallback to a rule-based architecture in production mode.

- [ ] **Step 4: Implement durable campaign state and round sequence**

`StudentCampaign` must atomically write `resume.json` before each external action, append events with fsync, and make a round idempotent by proposal digest. On restart, a compiled but unlaunched round resumes from its manifest; a launched round imports the existing result file before retrying. A failed proposal/compile/evaluation is appended and passed into the next bounded context. The runner stops on accepted valid student, configured max rounds, max failures, wall time, or unrecoverable infrastructure error.

- [ ] **Step 5: Run offline integration tests and commit**

Run: `python -m pytest -q tests/unit/test_student_campaign.py tests/integration/test_student_campaign.py`

Expected: PASS with two rounds and no manual proposal selection.

```bash
git add harness4h3/student/campaign.py harness4h3/controller/provider.py tests/unit/test_student_campaign.py tests/integration/test_student_campaign.py
git commit -m "feat: add autonomous student campaign loop"
```

### Task 6: Connect trusted SSH execution, detached remote operation, and CLI

**Files:**
- Create: `configs/student-campaign.example.yaml`
- Modify: `harness4h3/remote/config.py`
- Modify: `harness4h3/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/unit/test_student_config.py`
- Create: `tests/integration/test_student_cli.py`

**Interfaces:**
- Consumes: existing `SSHClient`, remote campaign config, local LLM URL/model, teacher checkpoint, ComfyUI benchmark recipe.
- Produces: commands `harness4h3 student-campaign validate`, `compile`, `run`, and an SSH-safe detached launcher.

- [ ] **Step 1: Write failing config/CLI tests**

```python
def test_student_campaign_config_requires_fixed_worker_and_real_evaluator(tmp_path):
    config = load_student_campaign_config(example_path(tmp_path))
    assert config.remote.worker_entrypoint.endswith("student_train_worker.py")
    assert config.target.min_params == 1_000_000_000
    with pytest.raises(ConfigError, match="worker_entrypoint"):
        load_student_campaign_config(config_without_worker(tmp_path))


def test_validate_and_compile_cli_are_offline(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "harness4h3", "student-campaign", "compile", "--proposal", str(proposal), "--output", str(tmp_path / "out"), "--json"),
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["graph_status"] == "compiled"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest -q tests/unit/test_student_config.py tests/integration/test_student_cli.py`

Expected: FAIL because the student campaign config and CLI namespace do not exist.

- [ ] **Step 3: Parse trusted remote config**

Add a `StudentCampaignConfig` containing goal, target, teacher checkpoint/adapter, local controller provider/model/URL, remote worker entrypoint/Python, campaign root, ComfyUI/evaluator recipe, retention cap, and budgets. Reject relative worker paths, missing teacher adapter, missing evaluator recipe, shell command strings, and retention caps below one. Reuse the existing `SSHClient` path validation and remote result import rather than adding a second SSH implementation.

- [ ] **Step 4: Add CLI commands and detached launcher**

Implement:

```bash
harness4h3 --config configs/student-campaign.example.yaml student-campaign validate --json
harness4h3 --config configs/student-campaign.example.yaml student-campaign compile \
  --proposal var/student/proposal.json --output var/student/compile --json
harness4h3 --config configs/student-campaign.example.yaml student-campaign run \
  --max-rounds 4 --detach --json
```

`run` must build fixed argv for the worker, upload manifest/config through `SSHClient.write_json`, and launch a remote `nohup`/`tmux` wrapper that writes a PID and result path under the configured campaign root. `--detach` returns after the remote process is confirmed alive; it must not poll indefinitely. A separate `status` path may read the existing result/state once but is not required for the campaign itself to continue.

- [ ] **Step 5: Verify CLI, config, and package discovery**

Run: `python -m pytest -q tests/unit/test_student_config.py tests/integration/test_student_cli.py && python -m compileall -q harness4h3/student tools/student_train_worker.py && python -m harness4h3 student-campaign --help`

Expected: PASS; help lists validate/compile/run and no arbitrary command flag.

- [ ] **Step 6: Commit**

```bash
git add configs/student-campaign.example.yaml harness4h3/remote/config.py harness4h3/cli.py pyproject.toml tests/unit/test_student_config.py tests/integration/test_student_cli.py
git commit -m "feat: expose detached student campaign over SSH"
```

### Task 7: Document the primary path and perform the full contract gate

**Files:**
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/operator-contract.md`
- Create: `docs/validation/student-campaign-acceptance.md`
- Create: `tests/integration/test_student_full_contract.py`

**Interfaces:**
- Consumes: all previous tasks and the example config.
- Produces: operator commands, explicit evidence checklist, and a two-round end-to-end contract test.

- [ ] **Step 1: Write the full offline contract test**

```python
def test_full_contract_has_legal_architecture_compile_training_failure_revision(tmp_path):
    result = run_two_round_student_campaign(tmp_path)
    assert 1_000_000_000 <= result.rounds[0].compile.parameter_count <= 2_000_000_000
    assert result.rounds[0].training.status == "success"
    assert result.rounds[0].evaluation.failure_code == "video_decode_failed"
    assert result.rounds[1].proposal.parent_proposal_id == result.rounds[0].proposal.proposal_id
    assert result.rounds[1].evaluation.valid is True
    assert result.retained_checkpoints == {"active", "parent", "best"}
```

- [ ] **Step 2: Run the full existing and new test suites**

Run: `python -m pytest -q`

Expected: all existing tests and the new Student contract tests pass. Existing remote tests must remain unchanged in behavior.

- [ ] **Step 3: Run static and package checks**

Run: `python -m compileall -q harness4h3 h3_training tools research && python -m harness4h3 --config configs/student-campaign.example.yaml student-campaign validate --json`

Expected: no compile errors and a JSON payload with `valid: true`.

- [ ] **Step 4: Update operator documentation**

Document local LLM setup, proposal schema, offline compile command, remote detached launch, resume/status behavior, result paths, failure codes, retention rules, and the exact distinction between contract evidence and real H3/video evidence. State clearly that the acceptance gate is not passed by TinyH3 or fake ComfyUI.

- [ ] **Step 5: Commit documentation and gate**

```bash
git add README.md docs/quickstart.md docs/operator-contract.md docs/validation/student-campaign-acceptance.md tests/integration/test_student_full_contract.py
git commit -m "docs: define autonomous student campaign acceptance"
```

## Execution handoff

The plan is ready for inline execution in this same thread. Execute Tasks 1–2 first and stop at their focused tests; then execute Tasks 3–5; finally perform the SSH/CLI and full contract gate. Do not claim the overall goal is complete until remote evidence contains a real H3 teacher load, a changed Student checkpoint, a real generated video, quality/VRAM measurements, and two persisted rounds.
