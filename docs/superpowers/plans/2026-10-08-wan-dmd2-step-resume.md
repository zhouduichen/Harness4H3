# Wan DMD2 and Step-level Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution with task-by-task checkpoints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the existing DMD2, TrainerEngine checkpoint, fingerprint, and ExperienceStore modules to enable memory-gated Wan DMD2 and exact step-level recovery.

**Architecture:** The Wan adapter remains the only Wan-specific boundary. The worker will construct the existing `DMD2` with a Wan Student, a copied Wan critic, and the existing H3 RGB-boundary endpoint Teacher. The existing `TrainerEngine` will gain an optimizer-boundary checkpoint callback, while the Wan campaign will use the existing canonical fingerprint and append one normalized experience record per round.

**Tech Stack:** PyTorch, existing `h3_training.algorithms.DMD2`, `h3_training.engine.TrainerEngine`, `h3_training.engine.checkpoint`, `harness4h3.memory.fingerprint`, `harness4h3.memory.experience`, pytest.

## Global Constraints

- `DMD2` remains the only DMD2 implementation; no Wan-specific DMD2 algorithm is allowed.
- The Wan Student and Critic must use the same trusted Wan checkpoint family; the H3 Teacher remains real and crosses the RGB→Wan VAE boundary.
- DMD2 is executable only when both `wan_dmd2_adapter_ready=true` and `dmd2_memory_admitted=true` are trusted config values.
- Checkpoints are written only at optimizer boundaries and must preserve Student, Critic, both optimizers, EMA, sampler RNG, CPU/CUDA RNG, loop state, parent digest, algorithm name, and config digest.
- Existing dirty worktree changes are preserved; only the files in this plan are modified.

---

## Task 1: Enable the existing DMD2 algorithm through the Wan adapter

**Files:**
- Modify: `tools/wan_student_train_worker.py`
- Modify: `h3_training/adapters/wan.py`
- Test: `tests/training/test_wan_adapter.py`
- Test: `tests/unit/test_wan_student_train_worker.py`

**Interfaces:**
- Consumes: `WanTrainingAdapter`, `WanEndpointTeacher`, `DMD2`, `DMD2Config`, and validated Wan policy fields.
- Produces: a real `DMD2` method with `student`, `teacher`, `critic`, `student_optimizer`, `critic_optimizer`, `ema`, and `sampler` states bound to Wan tensors.

- [ ] **Step 1: Add the failing DMD2 dispatch contract.**

```python
def test_dmd2_policy_constructs_shared_method(monkeypatch):
    seen = {}
    def fake_run(self, method, batches, max_steps, resume_from=None):
        seen["name"] = method.algorithm_name
        seen["roles"] = tuple(method.optimizer_map())
        return fake_training_result()
    monkeypatch.setattr(TrainerEngine, "run", fake_run)
    result = run_with_fake_wan_worker(method="dmd2", dmd2_memory_admitted=True, wan_dmd2_adapter_ready=True)
    assert seen == {"name": "dmd2", "roles": ("critic", "student")}
    assert result["algorithm_path"].endswith("DMD2")
```

- [ ] **Step 2: Run the focused test and verify current Wan worker rejects DMD2.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_wan_student_train_worker.py -k dmd2_policy_constructs`

Expected: FAIL because the worker currently raises `capability_unavailable` after the memory gate.

- [ ] **Step 3: Replace the unconditional DMD2 rejection with the shared constructor.**

Use this trusted branch after the memory gate:

```python
critic = copy.deepcopy(components.transformer)
method = DMD2(
    components.transformer,
    WanEndpointTeacher(),
    critic,
    adapter,
    DMD2Config(
        student_learning_rate=policy.learning_rate,
        critic_learning_rate=policy.critic_learning_rate,
        data_mode="real_latent",
        regression_weight=policy.clean_anchor_weight,
        generator_update_interval=2,
    ),
)
result = TrainerEngine(trainer_config).run(method, training_batches, max_steps=steps, resume_from=resume_path)
```

The adapter must distinguish the parameter-free `WanEndpointTeacher` from Student/Critic roles by `isinstance(role.model, WanEndpointTeacher)`, so DMD2 teacher calls use the detached H3 RGB-boundary target while critic calls use the Wan Transformer. Record `algorithm_name="dmd2"`, critic parameter count, `critic_updates`, `student_updates`, and EMA update count in the result. If `copy.deepcopy` raises OOM or the configured free-memory gate fails, raise `capability_unavailable` and do not fall back to another algorithm.

- [ ] **Step 4: Run DMD2 contract tests.**

Run: `.venv/bin/python -m pytest -q tests/training/test_dmd2.py tests/training/test_wan_adapter.py tests/unit/test_wan_student_train_worker.py`

Expected: existing DMD2 resume tests remain green, Wan role dispatch tests pass, and no CUDA is required for the tiny contract.

- [ ] **Step 5: Commit the DMD2 dispatch.**

```bash
git add tools/wan_student_train_worker.py h3_training/adapters/wan.py tests/training/test_wan_adapter.py tests/unit/test_wan_student_train_worker.py
git commit -m "feat: enable memory-gated Wan DMD2 dispatch"
```

## Task 2: Add optimizer-boundary checkpoint callbacks to TrainerEngine

**Files:**
- Modify: `h3_training/engine/trainer.py`
- Test: `tests/training/test_checkpoint_resume.py`
- Test: `tests/training/test_dmd2.py`

**Interfaces:**
- Consumes: existing `TrainerEngine.save_checkpoint`, `LoopState`, and `load_training_checkpoint`.
- Produces: optional `checkpoint_path` and `checkpoint_every_steps` arguments on `TrainerEngine.run` that save only after `optimizer_step` and return the same `TrainingRunResult` as before.

- [ ] **Step 1: Add a failing callback test.**

```python
def test_engine_writes_optimizer_boundary_checkpoint(tmp_path):
    method = CountingMethod()
    engine = TrainerEngine(TrainerConfig(seed=9))
    result = engine.run(method, batches(), max_steps=3, checkpoint_path=tmp_path / "step.pt", checkpoint_every_steps=2)
    assert result.loop_state.global_step == 3
    assert (tmp_path / "step.pt").is_file()
```

- [ ] **Step 2: Run it and verify `TrainerEngine.run` has no checkpoint arguments.**

Run: `.venv/bin/python -m pytest -q tests/training/test_checkpoint_resume.py -k optimizer_boundary`

Expected: FAIL with an unexpected keyword argument.

- [ ] **Step 3: Add the minimal optional arguments and callback.**

Extend the signature with:

```python
checkpoint_path: Optional[Path] = None,
checkpoint_every_steps: int = 0,
```

Validate `checkpoint_every_steps >= 0`; after `state.global_step = iteration` and `state.accumulation_position = 0`, call `self.save_checkpoint(method, state, checkpoint_path)` when the interval divides the global step. Preserve all existing call sites and return values.

- [ ] **Step 4: Verify exact DMD2 resume with the callback.**

Run: `.venv/bin/python -m pytest -q tests/training/test_checkpoint_resume.py tests/training/test_dmd2.py`

Expected: all existing resume tests pass and the new checkpoint is loadable by `load_training_checkpoint`.

- [ ] **Step 5: Commit the generic checkpoint callback.**

```bash
git add h3_training/engine/trainer.py tests/training/test_checkpoint_resume.py tests/training/test_dmd2.py
git commit -m "feat: checkpoint TrainerEngine at optimizer boundaries"
```

## Task 3: Connect Wan worker resume to the existing checkpoint format

**Files:**
- Modify: `tools/wan_student_train_worker.py`
- Modify: `tools/wan_vllm_campaign.py`
- Modify: `configs/student-campaign.wan-vllm-loop.yaml`
- Test: `tests/unit/test_wan_student_train_worker.py`
- Test: `tests/integration/test_wan_vllm_algorithm_loop.py`

**Interfaces:**
- Consumes: Task 1 DMD2 method and Task 2 `TrainerEngine.run(... checkpoint_path=..., resume_from=...)`.
- Produces: trusted `--resume-training-checkpoint`, `--checkpoint-every-steps`, result metadata, and a complete step-level recovery path.

- [ ] **Step 1: Add a failing worker command/resume contract.**

```python
def test_training_command_passes_step_checkpoint_options(tmp_path):
    command = build_training_command(config, proposal, output_dir=tmp_path / "round", result_path=tmp_path / "result.json", parent_checkpoint=tmp_path / "parent.safetensors")
    assert "--checkpoint-every-steps" in command
    assert "--resume-training-checkpoint" in command
```

- [ ] **Step 2: Add trusted CLI/config binding.**

The supervisor supplies only paths under the round directory and the existing parent binding. The worker accepts:

```text
--training-checkpoint PATH
--checkpoint-every-steps INTEGER
```

The worker computes a deterministic TrainerConfig digest from parent SHA256, algorithm policy, Wan config digest, and fidelity policy. It passes the checkpoint path to `TrainerEngine.run`, and when the file exists passes it as `resume_from`. A checkpoint with a mismatched parent or config digest fails with the existing `TrainingFailure("resume_mismatch", ...)` path.

- [ ] **Step 3: Record and validate checkpoint evidence.**

Add `training_checkpoint`, `training_checkpoint_sha256`, `resumed_from_step`, `checkpoint_every_steps`, and `resume_config_digest` to `training-result.json` and final safetensors metadata. The supervisor must reject a success result whose checkpoint metadata does not match the trusted parent or policy fingerprint.

- [ ] **Step 4: Run worker integration contracts.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_wan_student_train_worker.py tests/unit/test_wan_vllm_campaign.py tests/integration/test_wan_vllm_algorithm_loop.py`

Expected: command binding, checkpoint metadata, and accepted-parent state tests pass.

- [ ] **Step 5: Commit Wan worker resume integration.**

```bash
git add tools/wan_student_train_worker.py tools/wan_vllm_campaign.py configs/student-campaign.wan-vllm-loop.yaml tests/unit/test_wan_student_train_worker.py tests/unit/test_wan_vllm_campaign.py tests/integration/test_wan_vllm_algorithm_loop.py
git commit -m "feat: resume Wan training from optimizer checkpoints"
```

## Task 4: Use the existing fingerprint and ExperienceStore memory

**Files:**
- Modify: `harness4h3/student/wan_campaign.py`
- Modify: `tools/wan_vllm_campaign.py`
- Test: `tests/unit/test_wan_campaign.py`
- Test: `tests/integration/test_wan_vllm_algorithm_loop.py`

**Interfaces:**
- Consumes: normalized Wan proposal, base/capability/evaluator digests, round artifacts, and candidate decisions.
- Produces: canonical `experiment_fingerprint`, append-only `ExperienceStore` record, and the existing campaign state pointing to that record.

- [ ] **Step 1: Add a failing memory compatibility test.**

```python
def test_wan_round_memory_is_readable_by_experience_store(tmp_path):
    result = run_fake_wan_campaign(tmp_path)
    records = list(ExperienceStore(tmp_path / "experiences.jsonl").read())
    assert len(records) == 1
    assert records[0].fingerprint == result["rounds"][0]["fingerprint"]
    assert records[0].status == "accepted"
```

- [ ] **Step 2: Replace the custom policy hash with `experiment_fingerprint`.**

Call:

```python
fingerprint = experiment_fingerprint(
    parent_model_id=parent_digest,
    parent_system_id=identity["base_digest"],
    operator=proposal.training.method,
    operator_args=proposal.to_dict(),
    target_id="wan-30s-clip-temporal-v1",
    device_id=str(config.get("device")),
    benchmark_recipe={"manifest": manifest_digest, "evaluator": evaluator_digest, "capability": capability_digest},
)
```

Keep the existing duplicate check, but store this canonical value. It must change whenever the parent, method, policy, benchmark, device, or capability snapshot changes.

- [ ] **Step 3: Append one `ExperienceRecord` per completed round.**

Use `ExperienceStore(path).append(ExperienceRecord(...))` with status `accepted`, `rejected`, or `failed`; use the round directory as `source_uri` and its deterministic artifact digest as `source_sha256`. Include training policy, checkpoint evidence, evaluation metrics, decision, and provenance. Do not remove `events.jsonl` or `campaign-state.json`; those remain operational logs/state.

- [ ] **Step 4: Verify restart, duplicate, and experience import behavior.**

Run: `.venv/bin/python -m pytest -q tests/unit/test_wan_campaign.py tests/integration/test_wan_vllm_algorithm_loop.py`

Expected: restart does not call vLLM for completed rounds, duplicate fingerprints are rejected, and ExperienceStore can read every emitted record.

- [ ] **Step 5: Commit unified experiment memory.**

```bash
git add harness4h3/student/wan_campaign.py tools/wan_vllm_campaign.py tests/unit/test_wan_campaign.py tests/integration/test_wan_vllm_algorithm_loop.py
git commit -m "feat: record Wan rounds in shared experiment memory"
```

## Task 5: Final verification and capability documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-10-08-wan-native-algorithm-adaptation-design.md`
- Test: `tests/training/test_dmd2.py`
- Test: `tests/training/test_checkpoint_resume.py`
- Test: `tests/integration/test_wan_vllm_algorithm_loop.py`

- [ ] **Step 1: Run all focused regression suites.**

```bash
.venv/bin/python -m pytest -q tests/training/test_dmd2.py tests/training/test_checkpoint_resume.py
.venv/bin/python -m pytest -q tests/training/test_wan_adapter.py tests/unit/test_wan_student_train_worker.py tests/unit/test_wan_campaign.py tests/unit/test_wan_vllm_campaign.py tests/integration/test_wan_vllm_algorithm_loop.py
```

Expected: all CPU/tiny tests pass; CUDA-only tests may be skipped only when CUDA is unavailable.

- [ ] **Step 2: Run static checks and inspect the diff.**

```bash
.venv/bin/python -m py_compile h3_training/engine/trainer.py h3_training/adapters/wan.py tools/wan_student_train_worker.py tools/wan_vllm_campaign.py harness4h3/student/wan_campaign.py
git diff --check
```

Expected: no syntax or whitespace errors and no unrelated files staged.

- [ ] **Step 3: Update the spec with verified DMD2/resume/memory status.**

Document whether DMD2 was enabled by contract tests, whether step resume passed, and whether a real GPU smoke was available. Do not claim semantic 30-second improvement without real evaluation evidence.

- [ ] **Step 4: Commit final status documentation.**

```bash
git add docs/superpowers/specs/2026-10-08-wan-native-algorithm-adaptation-design.md
git commit -m "docs: record Wan DMD2 resume verification"
```

## Self-review checklist

- Every DMD2 behavior is delegated to the existing `DMD2` class; the Wan code only constructs roles and passes trusted tensors.
- Every training checkpoint uses the existing `TrainerEngine`/`checkpoint.py` schema; no second optimizer or EMA serialization format is introduced.
- Existing `experiment_fingerprint` and `ExperienceStore` are used for durable experiment memory; custom files remain only as operational campaign state/logs.
- DMD2 remains unavailable when either trusted memory or adapter evidence is absent.
- Real GPU semantic claims are explicitly separated from CPU/tiny contract tests.
