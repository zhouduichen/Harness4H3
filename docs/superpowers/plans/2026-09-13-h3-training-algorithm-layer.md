# H3 Training Algorithm Layer Implementation Plan

> **Execution mode:** Implement this plan task-by-task in the current task. The optional plan-execution skills are not installed in this workspace, so the same checkpoints and verification gates are followed directly. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the complete hardware-independent H3 training algorithm layer, including real TinyH3 training, recovery fine-tuning, binary progressive distillation, content-addressed caches, a DMD2 reference implementation, exact resume, child evidence, and the existing Harness worker/lineage loop.

**Architecture:** Add a top-level `h3_training` package outside the frozen `harness4h3` core. `TrainerEngine` owns loop mechanics; `TrainingMethod` subclasses own role models, losses, and optimizer schedules; adapters own parameterization and schedules. A trusted TinyH3 worker implements the existing external operator contract, while the H3 adapter remains a fail-closed contract until L40 integration.

**Tech Stack:** Python 3.9+, PyTorch 2.x, safetensors, pytest, existing Harness4H3 dataclasses and external worker contract.

## Global Constraints

- Do not modify Harness4H3-v1.0 Controller, evaluator authority, archive semantics, acceptance policy, or TargetProfile meaning.
- Do not modify or stage the user's existing changes in `research/experiments/m6_campaign.py` or `research/experiments/m6_runtime_recipe.py`.
- Ordinary Harness installation and imports must work without PyTorch.
- TinyH3 and DMD2 are reference implementations, not evidence of real MiniMax-H3 training.
- Every successful child must preserve the parent, change intended trainable tensors, preserve frozen tensors, reload, and emit measured evidence.
- One `step_distill` invocation performs one binary NFE stage; stage promotion remains an evaluator decision.

---

### Task 1: Training package contracts and data schema

**Files:**
- Modify: `pyproject.toml`
- Create: `h3_training/__init__.py`
- Create: `h3_training/data/__init__.py`
- Create: `h3_training/data/schema.py`
- Create: `h3_training/adapters/__init__.py`
- Create: `h3_training/adapters/base.py`
- Create: `h3_training/algorithms/__init__.py`
- Create: `h3_training/algorithms/base.py`
- Create: `h3_training/engine/__init__.py`
- Create: `h3_training/engine/state.py`
- Test: `tests/training/test_schema_and_contracts.py`

**Interfaces:**
- Consumes: no new project interfaces.
- Produces: `ModalLatents`, `ModalTimesteps`, `ModalInterval`, `ModalSchedule`, `Conditioning`, `TrainingSample`, `PreparedBatch`, `ModelRole`, `StepOutput`, `TrainingMethod`, and `DenoisingModelAdapter`.

- [ ] **Step 1: Write failing schema and import-isolation tests**

```python
def test_modal_latents_requires_a_modality():
    with pytest.raises(ValueError, match="at least one modality"):
        ModalLatents()

def test_training_sample_keeps_raw_fields_optional():
    sample = TrainingSample("s0", "a prompt", 7)
    assert sample.latents is None
    assert sample.text_embedding is None

def test_plain_harness_import_does_not_import_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; import harness4h3",
        ],
        cwd=ROOT,
        check=True,
    )
```

- [ ] **Step 2: Run the focused test and verify missing-package failure**

Run: `.venv/bin/python -m pytest tests/training/test_schema_and_contracts.py -q`

Expected: collection fails because `h3_training` does not exist.

- [ ] **Step 3: Add the optional dependency and package contracts**

```toml
[project.optional-dependencies]
test = ["pytest>=8,<9"]
training = [
  "torch>=2.2,<3",
  "safetensors>=0.4,<1",
]

[tool.setuptools.packages.find]
include = ["harness4h3*", "h3_training*", "research*"]
```

Implement frozen dataclasses that validate modality presence, matching batch
sizes, finite floating-point tensors, monotonic schedules, and positive NFE.
Define `TrainingMethod` as an abstract `torch.nn.Module` with
`training_step`, `optimizers`, `grad_clip_targets`, `checkpoint_state`, and
`load_checkpoint_state`. Define `DenoisingModelAdapter` with the exact
methods from the approved design.

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/training/test_schema_and_contracts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the contracts**

```bash
git add pyproject.toml h3_training tests/training/test_schema_and_contracts.py
git commit -m "feat: define training algorithm contracts"
```

### Task 2: TinyH3 model, schedule adapter, deterministic data, and evaluator

**Files:**
- Create: `h3_training/tiny/__init__.py`
- Create: `h3_training/tiny/model.py`
- Create: `h3_training/tiny/factory.py`
- Create: `h3_training/adapters/tiny.py`
- Create: `h3_training/data/dataset.py`
- Create: `h3_training/tiny/evaluator.py`
- Test: `tests/training/test_tiny_model.py`
- Test: `tests/training/test_tiny_evaluator.py`

**Interfaces:**
- Consumes: Task 1 data and adapter contracts.
- Produces: `TinyH3Config`, `TinyH3Model`, `TinyH3Adapter`, `SyntheticH3Dataset`, `create_tiny_checkpoint`, and `TinyCheckpointEvaluator`.

- [ ] **Step 1: Write failing real-forward and schedule tests**

```python
def test_tiny_h3_joint_forward_has_real_gradients():
    model, batch = tiny_fixture(seed=3)
    output = model(batch.latents, batch.conditioning, batch.timesteps)
    loss = output.video.square().mean() + output.audio.square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())

def test_h3_style_schedule_has_separate_audio_video_sigmas():
    schedule = TinyH3Adapter().schedule(4)
    assert len(schedule.video_sigmas) == 5
    assert len(schedule.audio_sigmas) == 5
    assert schedule.video_sigmas[1] != schedule.audio_sigmas[1]
    assert schedule.video_sigmas[-1] == schedule.audio_sigmas[-1] == 0.0
```

- [ ] **Step 2: Verify the tests fail because TinyH3 is missing**

Run: `.venv/bin/python -m pytest tests/training/test_tiny_model.py -q`

Expected: import failure for `h3_training.tiny`.

- [ ] **Step 3: Implement TinyH3 and its adapter**

Use a small batch-first Transformer encoder with modality projections,
conditioning/time embeddings, and separate video/audio velocity heads. The
adapter implements data-ward rectified flow:

```python
def add_noise(clean, noise, timestep):
    return timestep * clean + (1.0 - timestep) * noise

def prediction_to_clean(noisy, velocity, timestep):
    sigma = 1.0 - timestep
    return noisy + sigma * velocity

def shifted_sigma(base_sigma, shift):
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)
```

Use video shift 12 and audio shift 3. Checkpoints contain model config,
state dict, model ID, parent ID, sampling NFE, and provenance.

- [ ] **Step 4: Implement deterministic synthetic samples and evaluator**

`SyntheticH3Dataset[index]` derives every tensor from `base_seed + index`
without global RNG. `TinyCheckpointEvaluator.evaluate` reloads the model,
measures held-out velocity MSE and forward latency, computes actual checkpoint
size, and returns the existing `EvaluationResult`. Quality is
`1 / (1 + heldout_mse)`; feasibility uses the supplied TargetProfile and
baseline quality.

- [ ] **Step 5: Run model and evaluator tests**

Run: `.venv/bin/python -m pytest tests/training/test_tiny_model.py tests/training/test_tiny_evaluator.py -q`

Expected: all tests pass with non-zero gradients and finite measured metrics.

- [ ] **Step 6: Commit TinyH3**

```bash
git add h3_training/tiny h3_training/adapters/tiny.py h3_training/data/dataset.py tests/training/test_tiny_model.py tests/training/test_tiny_evaluator.py
git commit -m "feat: add real TinyH3 reference model"
```

### Task 3: Trainer engine, optimizer mechanics, checkpoint, and child evidence

**Files:**
- Create: `h3_training/engine/optimizer.py`
- Create: `h3_training/engine/checkpoint.py`
- Create: `h3_training/engine/evidence.py`
- Create: `h3_training/engine/trainer.py`
- Test: `tests/training/test_trainer_engine.py`
- Test: `tests/training/test_checkpoint_resume.py`
- Test: `tests/training/test_child_evidence.py`

**Interfaces:**
- Consumes: `TrainingMethod`, `ModelRole`, TinyH3 save/reload, and `PreparedBatch`.
- Produces: `TrainerConfig`, `LoopState`, `TrainerEngine`, `TrainingCheckpoint`, `ParentEvidence`, `ChildEvidence`, and stable `TrainingFailure` codes.

- [ ] **Step 1: Write failing accumulation, clipping, and atomic-checkpoint tests**

```python
def test_accumulation_steps_once_for_two_microbatches(tmp_path):
    method = CountingMethod()
    result = TrainerEngine(TrainerConfig(gradient_accumulation_steps=2)).run(
        method, four_batches(), max_steps=1
    )
    assert method.optimizer_steps == 1
    assert result.loop_state.microbatches_consumed == 2

def test_checkpoint_rejects_parent_mismatch(tmp_path):
    checkpoint = save_test_checkpoint(tmp_path, parent_sha256="a" * 64)
    with pytest.raises(TrainingFailure, match="resume_mismatch"):
        load_test_checkpoint(checkpoint, expected_parent_sha256="b" * 64)
```

- [ ] **Step 2: Run tests and verify missing-engine failures**

Run: `.venv/bin/python -m pytest tests/training/test_trainer_engine.py tests/training/test_checkpoint_resume.py tests/training/test_child_evidence.py -q`

Expected: import failures for engine modules.

- [ ] **Step 3: Implement the trainer loop and stable failures**

For each optimizer iteration, consume exactly the configured number of
micro-batches, divide `total_loss` before backward, validate finite scalar
losses, clip the method's named targets, step only optimizers returned for the
iteration, then zero those gradients. Record initial/final loss, maximum
gradient norm, role step counts, wall time, and CPU/CUDA peak memory when
available.

- [ ] **Step 4: Implement atomic full-state checkpoint/resume**

Save a versioned dictionary containing method state, optimizer and scheduler
states, EMA, loop and sampler position, CPU/CUDA RNG, algorithm generator,
parent hash, and config digest. Write to a sibling temporary file, fsync, then
`os.replace`. Load with `torch.load(..., weights_only=True)` and validate all
resume-critical fields before mutation.

- [ ] **Step 5: Implement child evidence**

Snapshot parent named tensors before training. On save, compare every student
tensor with the snapshot, reject trainable-no-change or frozen changes, verify
the parent file hash, reload the child through its adapter, and write a JSON
manifest with measured hashes and counts.

- [ ] **Step 6: Run engine tests**

Run: `.venv/bin/python -m pytest tests/training/test_trainer_engine.py tests/training/test_checkpoint_resume.py tests/training/test_child_evidence.py -q`

Expected: all tests pass, including parent immutability and corrupted
checkpoint rejection.

- [ ] **Step 7: Commit the engine**

```bash
git add h3_training/engine tests/training/test_trainer_engine.py tests/training/test_checkpoint_resume.py tests/training/test_child_evidence.py
git commit -m "feat: add resumable training engine"
```

### Task 4: Recovery fine-tuning

**Files:**
- Create: `h3_training/algorithms/recovery_finetune.py`
- Test: `tests/training/test_recovery_finetune.py`

**Interfaces:**
- Consumes: engine, TinyH3 adapter, model roles, and cached-latent batches.
- Produces: `RecoveryConfig` and `RecoveryFineTune`.

- [ ] **Step 1: Write failing recovery tests**

```python
def test_recovery_updates_only_heads_and_decreases_loss(tmp_path):
    method, batches = recovery_fixture(trainable_scope="heads")
    before = clone_named_parameters(method.student.model)
    result = TrainerEngine(test_config()).run(method, batches, max_steps=8)
    after = clone_named_parameters(method.student.model)
    assert result.final_loss < result.initial_loss
    assert changed_names(before, after)
    assert changed_names(before, after) <= method.trainable_parameter_names

def test_empty_freeze_policy_fails_before_forward():
    with pytest.raises(TrainingFailure, match="no_trainable_parameters"):
        recovery_fixture(trainable_scope="does-not-exist")
```

- [ ] **Step 2: Verify focused failures**

Run: `.venv/bin/python -m pytest tests/training/test_recovery_finetune.py -q`

Expected: import failure for `RecoveryFineTune`.

- [ ] **Step 3: Implement recovery loss and freeze evidence**

Compute adapter-native supervised flow loss plus optional frozen-teacher drift
loss, weighted independently for video and audio. Resolve `all`, `heads`, and
explicit-prefix freeze policies into exact parameter-name sets before building
AdamW. Teacher prediction is always under `torch.no_grad()` and teacher
parameters always have `requires_grad=False`.

- [ ] **Step 4: Add exact recovery resume test**

Train four steps, save, reconstruct, resume for four steps, and compare against
one eight-step run. Assert identical student parameters, optimizer state,
loop state, and final metrics.

- [ ] **Step 5: Run and commit recovery**

Run: `.venv/bin/python -m pytest tests/training/test_recovery_finetune.py -q`

Expected: all tests pass.

```bash
git add h3_training/algorithms/recovery_finetune.py tests/training/test_recovery_finetune.py
git commit -m "feat: implement recovery fine-tuning"
```

### Task 5: Binary progressive distillation

**Files:**
- Create: `h3_training/algorithms/progressive_distillation.py`
- Test: `tests/training/test_progressive_distillation.py`

**Interfaces:**
- Consumes: TinyH3 schedule/transition methods and TrainerEngine.
- Produces: `DistillationStage`, `plan_binary_stages`, `ProgressiveDistillationConfig`, and `ProgressiveDistillation`.

- [ ] **Step 1: Write failing stage and loss tests**

```python
def test_binary_stage_requires_two_to_one_nfe():
    with pytest.raises(ValueError, match="twice"):
        DistillationStage(teacher_nfe=12, student_nfe=8)

def test_plan_binary_stages():
    assert plan_binary_stages(16, 4) == (
        DistillationStage(16, 8),
        DistillationStage(8, 4),
    )

def test_progressive_distillation_updates_student_not_teacher():
    method, batches = progressive_fixture(teacher_nfe=4, student_nfe=2)
    teacher_before = clone_named_parameters(method.teacher.model)
    student_before = clone_named_parameters(method.student.model)
    TrainerEngine(test_config()).run(method, batches, max_steps=4)
    assert clone_named_parameters(method.teacher.model) == teacher_before
    assert clone_named_parameters(method.student.model) != student_before
```

- [ ] **Step 2: Run the focused test and verify import failure**

Run: `.venv/bin/python -m pytest tests/training/test_progressive_distillation.py -q`

Expected: import failure for progressive distillation.

- [ ] **Step 3: Implement aligned two-step teacher and one-step student targets**

Build teacher and student modality schedules from explicit NFE. Sample a
student interval, construct its noisy start state, run two teacher transitions
under `no_grad`, and run one differentiable student transition over the same
outer endpoints. Reject any modality whose endpoint sigma does not align.
Compute weighted endpoint MSE for video and audio.

- [ ] **Step 4: Add exact resume and modality-weight tests**

Verify video-only, audio-only, and joint losses; zero weights for missing
modalities; invalid grids; frozen teacher; non-zero student gradients; child
reload; and uninterrupted versus resumed equivalence.

- [ ] **Step 5: Run and commit progressive distillation**

Run: `.venv/bin/python -m pytest tests/training/test_progressive_distillation.py -q`

Expected: all tests pass.

```bash
git add h3_training/algorithms/progressive_distillation.py tests/training/test_progressive_distillation.py
git commit -m "feat: implement binary progressive distillation"
```

### Task 6: Content-addressed tensor cache

**Files:**
- Create: `h3_training/data/cache.py`
- Test: `tests/training/test_training_cache.py`

**Interfaces:**
- Consumes: training data dataclasses and safetensors.
- Produces: `CacheKey`, `CacheManifest`, and `TrainingCache`.

- [ ] **Step 1: Write failing key and corruption tests**

```python
def test_teacher_cache_key_changes_with_teacher_and_schedule(tmp_path):
    cache = TrainingCache(tmp_path)
    base = cache.key(kind="teacher_prediction", sample_digest="s", model_digest="m1", schedule=[1.0, 0.0], seed=1)
    assert base != cache.key(kind="teacher_prediction", sample_digest="s", model_digest="m2", schedule=[1.0, 0.0], seed=1)
    assert base != cache.key(kind="teacher_prediction", sample_digest="s", model_digest="m1", schedule=[1.0, 0.5, 0.0], seed=1)

def test_corrupt_payload_fails_closed(tmp_path):
    cache = populated_cache(tmp_path)
    cache.payload_path.write_bytes(b"broken")
    with pytest.raises(TrainingFailure, match="cache_corrupt"):
        cache.load(cache.key_value)
```

- [ ] **Step 2: Verify missing-cache failure**

Run: `.venv/bin/python -m pytest tests/training/test_training_cache.py -q`

Expected: import failure for `TrainingCache`.

- [ ] **Step 3: Implement canonical keys and atomic entries**

Serialize key material with sorted canonical JSON and SHA-256. Store tensors
in safetensors and metadata in a manifest containing schema, tensor names,
shapes, dtypes, and payload SHA-256. Publish a completed entry by atomically
renaming a temporary directory; if another process wins the same key, verify
and reuse its entry.

- [ ] **Step 4: Cover all invalidation dimensions and commit**

Run: `.venv/bin/python -m pytest tests/training/test_training_cache.py -q`

Expected: tests cover source, model, preprocessing, modality, dtype, shape,
schedule, timestep, conditioning, and seed changes.

```bash
git add h3_training/data/cache.py tests/training/test_training_cache.py
git commit -m "feat: add content-addressed training cache"
```

### Task 7: DMD2 multi-role reference implementation

**Files:**
- Create: `h3_training/algorithms/dmd2.py`
- Test: `tests/training/test_dmd2.py`

**Interfaces:**
- Consumes: TrainerEngine, TinyH3 roles, adapter noise/prediction conversion, and real/text-only batches.
- Produces: `DMD2Config`, `TimestepNoiseSampler`, `StudentEMA`, and `DMD2`.

- [ ] **Step 1: Write failing role/update tests**

```python
def test_dmd2_updates_critic_every_step_and_student_on_interval():
    method, batches = dmd2_fixture(generator_update_interval=2)
    result = TrainerEngine(test_config()).run(method, batches, max_steps=4)
    assert result.optimizer_steps["critic"] == 4
    assert result.optimizer_steps["student"] == 2
    assert method.ema.num_updates == 2

def test_dmd2_loss_is_finite_and_gradients_nonzero():
    method, batch = dmd2_single_batch_fixture()
    output = method.training_step(batch, iteration=2)
    output.losses["total_loss"].backward()
    assert torch.isfinite(output.losses["total_loss"])
    assert nonzero_grad(method.student.model)
    assert nonzero_grad(method.critic.model)
```

- [ ] **Step 2: Run focused tests and verify import failure**

Run: `.venv/bin/python -m pytest tests/training/test_dmd2.py -q`

Expected: import failure for `DMD2`.

- [ ] **Step 3: Implement critic and distribution-matching losses**

Generate student samples from text-only noise or real-latent rollout. Train
the critic on detached noised student samples with adapter-native denoising
targets. Compute teacher and critic clean estimates under `no_grad`, normalize
their difference with finite guards, and expose the prescribed student
gradient using a detached pseudo-target MSE.

- [ ] **Step 4: Implement alternating updates, EMA, GAN, and regression anchor**

Return the critic optimizer every iteration and the student optimizer only on
the configured interval. Advance EMA only when the student steps. Add focused
real-latent tests with non-zero adversarial weight and a separate regression
anchor test; keep both weights zero by default.

- [ ] **Step 5: Add exact DMD2 resume test**

Compare uninterrupted and resumed student, critic, both optimizers, EMA,
sampler generator, counters, and final losses on CPU.

- [ ] **Step 6: Run and commit DMD2**

Run: `.venv/bin/python -m pytest tests/training/test_dmd2.py -q`

Expected: all tests pass.

```bash
git add h3_training/algorithms/dmd2.py tests/training/test_dmd2.py
git commit -m "feat: add DMD2 reference trainer"
```

### Task 8: Fail-closed H3 contract and trusted Tiny worker

**Files:**
- Create: `h3_training/adapters/h3_contract.py`
- Create: `tools/tiny_training_worker.py`
- Test: `tests/training/test_h3_contract.py`
- Test: `tests/training/test_tiny_training_worker.py`
- Modify: `tests/unit/test_h3_model_worker.py`

**Interfaces:**
- Consumes: approved H3 schedule facts, recovery/progressive methods, TrainerEngine, and existing worker JSON.
- Produces: `H3AdapterContract` and a trainer CLI accepting `--request` and `--result`.

- [ ] **Step 1: Write failing H3 and worker tests**

```python
def test_h3_contract_fails_without_real_loader():
    with pytest.raises(TrainingFailure, match="h3_adapter_unavailable"):
        H3AdapterContract().load_role(Path("parent"), trainable=False)

def test_tiny_worker_returns_real_child_and_metrics(tmp_path):
    completed, result = run_tiny_worker(tmp_path, operator="recovery_finetune")
    assert completed.returncode == 0
    assert result["status"] == "success"
    assert result["metrics"]["real_worker"] is True
    assert result["metrics"]["optimizer_steps"] > 0
    assert result["metrics"]["parent_sha256"] != result["metrics"]["child_sha256"]
```

- [ ] **Step 2: Verify missing-worker failures**

Run: `.venv/bin/python -m pytest tests/training/test_h3_contract.py tests/training/test_tiny_training_worker.py -q`

Expected: import or missing-script failures.

- [ ] **Step 3: Implement the explicit H3 contract**

Expose confirmed data-ward velocity, `t = 1 - sigma`, shifts 12/3, and
separate modality schedules. All load/train/save calls raise
`h3_adapter_unavailable` with the missing concrete capability.

- [ ] **Step 4: Implement Tiny worker request/result behavior**

Support only `recovery_finetune` and `step_distill`. Validate parent/child
identity and bounded numeric arguments. Resolve recovery `training_steps`
from operator args and progressive training steps from trusted CLI config;
infer progressive teacher NFE from the parent and require target NFE to be
exactly half. Write failed JSON for every stable TrainingFailure and never
emit partial success.

- [ ] **Step 5: Exercise the existing two-process adapter chain**

Configure `h3_model_worker.py` with `tiny_training_worker.py` as its fixed
trainer command. Assert parent bytes are unchanged, the staged child reloads,
and all authenticity metrics survive adapter normalization.

- [ ] **Step 6: Run and commit worker integration**

Run: `.venv/bin/python -m pytest tests/training/test_h3_contract.py tests/training/test_tiny_training_worker.py tests/unit/test_h3_model_worker.py -q`

Expected: all tests pass.

```bash
git add h3_training/adapters/h3_contract.py tools/tiny_training_worker.py tests/training/test_h3_contract.py tests/training/test_tiny_training_worker.py tests/unit/test_h3_model_worker.py
git commit -m "feat: connect TinyH3 training worker"
```

### Task 9: Real PyTorch Harness closed loop and completion audit

**Files:**
- Create: `research/experiments/tiny_real_closed_loop.py`
- Create: `tests/integration/test_tiny_real_closed_loop.py`
- Modify: `README.md`
- Modify: `docs/operator-contract.md`

**Interfaces:**
- Consumes: existing campaign runner, ExternalScriptOperator registry, Tiny worker, TinyCheckpointEvaluator, and model store.
- Produces: `run_tiny_real_closed_loop` plus a documented CPU validation command.

- [ ] **Step 1: Write the failing M0001/M0002 integration test**

```python
def test_real_tiny_training_creates_evaluated_lineage(tmp_path):
    result = run_tiny_real_closed_loop(tmp_path, max_experiments=2, seed=11)
    lineage = result.model_store.lineage()
    assert [item.id for item in lineage] == ["M0000", "M0001", "M0002"]
    assert lineage[1].parent_id == "M0000"
    assert lineage[2].parent_id == "M0001"
    assert all(Path(item.checkpoint_path).is_file() for item in lineage)
    assert all(item.state.provenance.get("offline_simulation") is False for item in lineage[1:])
    assert result.report["total_experiments"] == 2
```

- [ ] **Step 2: Run it and verify missing experiment failure**

Run: `.venv/bin/python -m pytest tests/integration/test_tiny_real_closed_loop.py -q`

Expected: import failure for the experiment module.

- [ ] **Step 3: Implement the deterministic closed loop**

Create M0000 with a real TinyH3 checkpoint. Use a bounded deterministic
Controller to request recovery for M0001 and one 2:1 step-distillation stage
for M0002. Execute both through ExternalScriptOperator and the two worker
processes. Evaluate every child by reloading its checkpoint and measuring the
held-out dataset. Return the campaign result and model store for inspection.

- [ ] **Step 4: Document scope and commands**

Add a README capability row for the Tiny real training loop and keep real H3
training marked unavailable. Extend the operator contract with the Tiny
reference command and an explicit warning that it is protocol/algorithm
evidence, not MiniMax-H3 evidence.

- [ ] **Step 5: Run the focused closed loop**

Run: `.venv/bin/python -m pytest tests/integration/test_tiny_real_closed_loop.py -q`

Expected: M0000 -> M0001 -> M0002 is produced and evaluated from real
PyTorch checkpoints.

- [ ] **Step 6: Run failure and exact-resume suites**

Run: `.venv/bin/python -m pytest tests/training -q`

Expected: all training tests pass, including NaN, zero-gradient, corrupt
cache/checkpoint, parent mutation, frozen mutation, unchanged child, reload
failure, and exact resume.

- [ ] **Step 7: Verify optional-dependency isolation**

Run: `python3 -m pytest -q`

Expected: ordinary Harness suite passes in an environment without PyTorch,
with training-only tests skipped using `pytest.importorskip("torch")` rather
than importing training modules during collection.

- [ ] **Step 8: Run full training environment regression checks**

Run: `.venv/bin/python -m pytest -q`

Expected: full suite passes.

Run: `.venv/bin/python -m compileall -q harness4h3 h3_training research tools tests`

Expected: exit status 0 with no output.

- [ ] **Step 9: Audit scope and user changes**

Run: `git status --short`

Expected: the user's two pre-existing M6 modifications remain unstaged and
unchanged; no run artifacts or caches are staged.

Run: `git diff --check HEAD`

Expected: no whitespace errors.

- [ ] **Step 10: Commit documentation and integration**

```bash
git add research/experiments/tiny_real_closed_loop.py tests/integration/test_tiny_real_closed_loop.py README.md docs/operator-contract.md
git commit -m "feat: validate real TinyH3 evolution loop"
```
