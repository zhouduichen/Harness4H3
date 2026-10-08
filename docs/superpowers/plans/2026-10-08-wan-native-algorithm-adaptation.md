# Wan-native Student Algorithm Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution with task-by-task checkpoints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已有 H3 Progressive Distillation、DMD2、rollout、F1→F2→F3 和 lineage 机制接入真实 Wan2.1-T2V-1.3B Student，并让 vLLM 只能在受信任搜索空间内选择策略、持久化实验记忆并可恢复。

**Architecture:** 训练算法仍由 `h3_training.algorithms` 提供；新增的 Wan adapter 只负责 Wan VAE、Flow-Match、Transformer forward、scheduler 和 checkpoint 边界。现有 H3 Teacher 继续在 RGB 边界生成真实目标，Wan adapter 将该目标编码为 Wan latent，随后通过现有 `ProgressiveDistillation` 或 `DMD2` 运行。Wan vLLM campaign 将旧 flat proposal 规范化为 nested schema v2，经过 capability gate 后才生成 trusted worker command，并用 campaign state、round artifacts 和 fingerprint index 实现恢复与去重。

**Tech Stack:** Python 3.11+, PyTorch, Diffusers WanPipeline, safetensors, PyYAML, pytest, OpenAI-compatible vLLM HTTP API。

## Global Constraints

- 保持真实 H3 Teacher、Wan 1–2B Student、30 秒 overlap-window evaluator 和 checkpoint lineage 不变。
- 不新增 Student 架构，不把 H3 latent 直接当作 Wan latent；Teacher 输出必须经过 RGB 边界后再由 Wan VAE 编码。
- vLLM 只能产生 bounded policy fields，不能输出路径、shell、设备、模型身份、Teacher 身份或代码。
- `offline_simulation` 必须为 `false`；训练结果必须包含 parent/Teacher digest、algorithm dispatch、fidelity 和 continuation evidence。
- Progressive Distillation 只接受可二分的 source/target NFE；DMD2 只有在 Wan critic memory gate 通过时可选。
- 任何现有未提交文件都属于用户已有工作，实施只改本计划列出的文件和新增测试，不执行 reset、checkout 或清理。

---

## File Map

- Create: `h3_training/adapters/wan.py` — Wan-native `DenoisingModelAdapter`，包括 Flow-Match 数学、Wan role forward、scope 解析和 boundary teacher。
- Modify: `h3_training/adapters/__init__.py` — 导出 `WanTrainingAdapter`。
- Modify: `harness4h3/student/wan_training.py` — 提供 adapter 所需的组件加载、帧→Wan latent、checkpoint save/load 和 FP32 loss reduction 配置。
- Modify: `tools/wan_student_train_worker.py` — 将当前独立 `wan_velocity_loss` 循环改为共享算法 dispatch，并保留 H3 Teacher RGB 生成。
- Modify: `harness4h3/student/wan_campaign.py` — proposal v1/v2 规范化、capability gate、fingerprint、campaign state。
- Modify: `tools/wan_vllm_campaign.py` — v2 JSON schema、v2→trusted command 映射、启动恢复、round artifacts 和 history context。
- Modify: `configs/student-campaign.wan-vllm-loop.yaml` — 明确算法能力、30 秒 continuation、state/fingerprint 文件路径。
- Create: `tests/training/test_wan_adapter.py` — adapter 数学、role forward、scope、checkpoint contract 测试。
- Modify: `tests/unit/test_wan_training.py` — Wan training config 允许算法策略字段并验证 FP32 loss reduction。
- Modify: `tests/unit/test_wan_campaign.py` — nested proposal、capability rejection、fingerprint 和 state recovery 测试。
- Modify: `tests/unit/test_wan_vllm_campaign.py` — v2 schema、command dispatch、resume path 和安全字段测试。

## Task 1: Add the shared Wan adapter boundary

**Files:**
- Create: `h3_training/adapters/wan.py`
- Modify: `h3_training/adapters/__init__.py`
- Test: `tests/training/test_wan_adapter.py`

**Interfaces:**
- Consumes: `WanTrainingComponents`, `WanTrainingSpec`, an RGB Teacher mapping with keys `frames` and `prompt`, and the existing `DenoisingModelAdapter`/`Modal*` dataclasses.
- Produces: `WanTrainingAdapter`, `WanEndpointTeacher`, and a `PreparedBatch` accepted by `ProgressiveDistillation` and `DMD2`.

- [ ] **Step 1: Write failing adapter tests.**

```python
def test_wan_adapter_flow_match_and_clean_equation():
    adapter = make_adapter()
    clean = ModalLatents(video=torch.zeros(1, 4, 2, 2, 2))
    noise = ModalLatents(video=torch.ones_like(clean.video))
    noisy = adapter.add_noise(clean, noise, ModalTimesteps(video=torch.tensor([0.25])))
    assert torch.allclose(noisy.video, torch.full_like(clean.video, 0.25))
    prediction = ModalPrediction(video=torch.ones_like(clean.video))
    recovered = adapter.prediction_to_clean(noisy, prediction, ModalTimesteps(video=torch.tensor([0.25])))
    assert torch.allclose(recovered.video, torch.zeros_like(clean.video))

def test_wan_adapter_schedule_is_binary_aligned():
    adapter = make_adapter()
    teacher = adapter.schedule(8).video_sigmas
    student = adapter.schedule(4).video_sigmas
    assert teacher is not None and student is not None
    assert all(abs(student[index] - teacher[2 * index]) < 1e-8 for index in range(5))

def test_wan_adapter_prepares_rgb_boundary_and_native_conditioning():
    adapter = make_adapter()
    batch = adapter.prepare_batch({"sample_id": "s0", "prompt": "red train", "frames": torch.zeros(2, 8, 8, 3)}, torch.Generator().manual_seed(7))
    assert batch.latents.video.shape == (1, 4, 2, 2, 2)
    assert batch.conditioning.text.shape[0] == 1
    assert batch.sample_ids == ("s0",)

def test_wan_adapter_scope_rejects_empty_selection():
    adapter = make_adapter()
    role = ModelRole("student", adapter.transformer, adapter, True)
    with pytest.raises(TrainingFailure, match="no trainable"):
        tuple(adapter.resolve_trainable_parameters(role, "missing.*"))
```

- [ ] **Step 2: Run the focused tests and verify they fail because the adapter does not exist.**

Run: `pytest -q tests/training/test_wan_adapter.py`

Expected: collection or import failure mentioning `WanTrainingAdapter`.

- [ ] **Step 3: Implement the adapter with the existing contracts.**

Implement these exact behaviors in `h3_training/adapters/wan.py`:

```python
class WanTrainingAdapter(DenoisingModelAdapter):
    generation_start_timestep = 1.0

    def add_noise(self, clean, noise, timestep):
        sigma = timestep.video.to(clean.video.device, clean.video.dtype).view(-1, 1, 1, 1, 1)
        return ModalLatents(video=(1.0 - sigma) * clean.video + sigma * noise.video)

    def prediction_to_clean(self, noisy, prediction, timestep):
        sigma = timestep.video.to(noisy.video.device, noisy.video.dtype).view(-1, 1, 1, 1, 1)
        return ModalLatents(video=noisy.video - sigma * prediction.video)

    def scheduler_step(self, role, latent, prediction, interval):
        return ModalLatents(video=latent.video + (interval.video[0] - interval.video[1]) * prediction.video)

    def schedule(self, num_model_evaluations):
        if int(num_model_evaluations) <= 0:
            raise ValueError("number of model evaluations must be positive")
        values = tuple(1.0 - index / float(num_model_evaluations) for index in range(num_model_evaluations + 1))
        return ModalSchedule(video_sigmas=values)
```

`predict` must pass `hidden_states`, `timestep * scheduler.config.num_train_timesteps`, and native `encoder_hidden_states` to the Wan transformer for Student/Critic roles. For the Teacher role it must use `WanEndpointTeacher.predict(noisy, sigma, clean_target)` so the H3 Teacher RGB target is represented as a detached Wan endpoint; it must never reinterpret H3 latent rows as Wan latent channels. `prepare_batch` must call the injected frame encoder and prompt encoder, create a CPU-generator noise tensor, and return `PreparedBatch` with `ModalLatents(video=clean)` and `ModalLatents(video=noise)`. `resolve_trainable_parameters` must support `all`, `heads`, and comma-separated `fnmatch` patterns. `save_role` must delegate to `save_wan_checkpoint`; `reload_role` must delegate to the trusted Wan loader.

- [ ] **Step 4: Export the adapter and run the focused tests.**

Run: `pytest -q tests/training/test_wan_adapter.py tests/unit/test_wan_training.py`

Expected: all adapter and existing Wan helper tests pass.

- [ ] **Step 5: Commit the adapter boundary.**

```bash
git add h3_training/adapters/wan.py h3_training/adapters/__init__.py tests/training/test_wan_adapter.py
git commit -m "feat: add shared Wan denoising adapter"
```

## Task 2: Normalize the bounded Wan strategy schema

**Files:**
- Modify: `harness4h3/student/wan_campaign.py`
- Modify: `tools/wan_vllm_campaign.py`
- Test: `tests/unit/test_wan_campaign.py`
- Test: `tests/unit/test_wan_vllm_campaign.py`

**Interfaces:**
- Consumes: legacy v1 flat proposals and v2 nested JSON returned by vLLM.
- Produces: immutable `WanTrainingProposal` with `training`, `initialization`, `precision`, `curriculum`, `data`, and `continuation` policy objects, plus `to_dict()`, `fingerprint()`, `capability_requirements()`, and `validate_capabilities()`.

- [ ] **Step 1: Add failing tests for v2 normalization and unsafe-field rejection.**

```python
def test_v2_proposal_normalizes_nested_training_policy():
    parent = "a" * 64
    proposal = WanTrainingProposal.from_mapping(v2_payload(parent), expected_parent_sha256=parent, capabilities=progressive_capabilities())
    assert proposal.schema_version == 2
    assert proposal.training.method == "progressive_distillation"
    assert proposal.training.rollout_steps == 4
    assert proposal.continuation.overlap_latent_frames == 7

def test_v1_proposal_remains_readable_as_v2_compatible_policy():
    parent = "a" * 64
    proposal = WanTrainingProposal.from_mapping(v1_payload(parent), expected_parent_sha256=parent, capabilities=progressive_capabilities())
    assert proposal.schema_version == 1
    assert proposal.training.method == "progressive_distillation"

def test_v2_rejects_unavailable_dmd2_and_paths():
    parent = "a" * 64
    with pytest.raises(WanProposalError, match="capability_unavailable"):
        WanTrainingProposal.from_mapping(v2_payload(parent, method="dmd2", shell="python evil.py"), expected_parent_sha256=parent, capabilities=progressive_capabilities())
```

- [ ] **Step 2: Run the focused tests and verify the current flat parser rejects nested v2.**

Run: `pytest -q tests/unit/test_wan_campaign.py tests/unit/test_wan_vllm_campaign.py`

Expected: the new v2 tests fail while existing v1 tests remain green.

- [ ] **Step 3: Implement schema v2 using frozen dataclasses and explicit bounds.**

Add frozen policy dataclasses with these fields and limits:

```python
training.method in {"progressive_distillation", "dmd2"}
training.source_steps in [1, 256]
training.target_steps in [1, 64] and target_steps <= source_steps
training.train_steps in [1, 100000]
training.learning_rate, critic_learning_rate in (0, 0.01]
training.rollout_steps in {0, 2, 4}; rollout_every in [1, 16]
training.trainable_scope in {"all", "heads"} or a validated comma-separated prefix pattern
initialization.mode in {"inherit", "generation_head_reset", "dit_zero"}
precision.loss_reduction in {"bf16", "fp32"}; precision.optimizer_state in {"bf16", "fp32"}
curriculum.fidelity_schedule is a non-empty ordered prefix of ["F1", "F2", "F3"]
data.prompt_policy in {"balanced", "fixed_manifest"}; data.seed_policy in {"fixed", "rotating"}
continuation.overlap_latent_frames in [6, 12]; continuation.anchor_schedule in {"constant", "linear", "cosine"}
```

`from_mapping` must reject unknown keys before reading values, bind the lowercase parent SHA-256, enforce binary stages via `plan_binary_stages`, and reject `dmd2` unless the capability snapshot has `available=true` for `wan_dmd2`. It must preserve v1 parsing by mapping `steps→target_steps`, `teacher_reuse_steps→rollout_every`, and `learning_rate→training.learning_rate` with `method="progressive_distillation"`. `fingerprint()` must hash canonical JSON of policy fields, parent digest, manifest digest, capability digest, evaluator digest, and code digest; no path string may be used as a substitute for a content digest.

- [ ] **Step 4: Update the vLLM JSON schema and prompt to emit only v2 bounded policy fields.**

`proposal_json_schema()` must set `schema_version` to `2`, define the nested objects above with `additionalProperties: false`, and omit shell/path/device/model/Teacher properties. The controller prompt must say: “Choose one executable policy from the provided capability snapshot; preserve real H3 Teacher, Wan Student, `offline_simulation=false`, and the 30-second overlap contract.”

- [ ] **Step 5: Run proposal/security tests and commit.**

Run: `pytest -q tests/unit/test_wan_campaign.py tests/unit/test_wan_vllm_campaign.py`

Expected: v1 compatibility, v2 validation, capability rejection, and no-path/no-shell schema tests pass.

```bash
git add harness4h3/student/wan_campaign.py tools/wan_vllm_campaign.py tests/unit/test_wan_campaign.py tests/unit/test_wan_vllm_campaign.py
git commit -m "feat: add bounded Wan strategy schema v2"
```

## Task 3: Dispatch shared Progressive/DMD2 algorithms in the Wan worker

**Files:**
- Modify: `harness4h3/student/wan_training.py`
- Modify: `tools/wan_student_train_worker.py`
- Modify: `configs/student-campaign.wan-vllm-loop.yaml`
- Test: `tests/unit/test_wan_student_train_worker.py`
- Test: `tests/training/test_wan_adapter.py`

**Interfaces:**
- Consumes: validated `WanTrainingProposal` policy fields and the existing `_teacher_video` RGB boundary.
- Produces: real Wan checkpoint/result metadata containing `algorithm_name`, `algorithm_path`, `rollout_policy`, `initialization_mode`, `precision_policy`, `stage_lineage`, `parent_checkpoint_sha256`, and `teacher_checkpoint_sha256`.

- [ ] **Step 1: Add dispatch tests before changing the worker.**

```python
def test_worker_progressive_dispatch_uses_shared_algorithm(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("tools.wan_student_train_worker.TrainerEngine.run", lambda self, method, batches, max_steps: seen.setdefault("method", method.algorithm_name) or fake_result())
    result = run_worker_with_fake_wan(tmp_path, method="progressive_distillation")
    assert seen["method"] == "progressive_distillation"
    assert result["algorithm_path"].endswith("ProgressiveDistillation")

def test_worker_rejects_dmd2_without_critic_memory_capability(tmp_path):
    with pytest.raises(WanStudentError, match="dmd2"):
        run_worker_with_fake_wan(tmp_path, method="dmd2", dmd2_memory_admitted=False)
```

- [ ] **Step 2: Implement shared dispatch without modifying algorithm source.**

In `wan_student_train_worker.py`, keep `_teacher_video` and use its frames as raw adapter samples. Construct one `WanTrainingAdapter` per loaded Wan component set, one `WanEndpointTeacher` per batch boundary, and call:

```python
stages = plan_binary_stages(proposal.training.source_steps, proposal.training.target_steps)
method = ProgressiveDistillation(
    components.transformer,
    endpoint_teacher,
    adapter,
    ProgressiveDistillationConfig(
        stage=stage,
        learning_rate=proposal.training.learning_rate,
        trainable_scope=proposal.training.trainable_scope,
        velocity_weight=proposal.training.velocity_weight,
        clean_anchor_weight=proposal.training.clean_anchor_weight,
        rollout_steps=proposal.training.rollout_steps,
        rollout_every=proposal.training.rollout_every,
        rollout_endpoint_weight=proposal.training.rollout_endpoint_weight,
        rollout_clean_weight=proposal.training.rollout_clean_weight,
        fp32_optimizer_state=proposal.precision.optimizer_state == "fp32",
    ),
)
```

Use `TrainerEngine` once per stage, save each stage through the adapter, and record promoted stage checkpoint SHA-256 in `stage_lineage`. For DMD2, admit only when the trusted config says `dmd2_memory_admitted: true`; make a deep copy of the Wan transformer, construct `DMD2Config` from bounded values, and fail closed with `capability_unavailable` before loading a critic if the memory gate is false. The direct `wan_velocity_loss` loop remains callable only through a compatibility flag for existing v1 artifacts; new v2 proposals must never dispatch to it.

- [ ] **Step 3: Add FP32 loss reduction and trusted initialization handling.**

Use `torch.float32` for all adapter MSE reductions while retaining the Wan transformer dtype. Apply `inherit`, `generation_head_reset`, or `dit_zero` only through the existing Wan checkpoint and parameter reset helpers; record the selected mode in checkpoint metadata. Do not enable FP32 optimizer state unless the config memory admission says it fits.

- [ ] **Step 4: Update the Wan loop config to expose trusted capability gates.**

Add `proposal_schema_version: 2`, `algorithm_capabilities: [progressive_distillation]`, `dmd2_memory_admitted: false`, `state_path`, `fingerprint_path`, `fidelity_schedule: [F1, F2, F3]`, `continuation_contract: 30s`, and `offline_simulation: false`. Keep all existing absolute trusted paths and GPU handoff fields unchanged.

- [ ] **Step 5: Run worker and adapter tests, then commit.**

Run: `pytest -q tests/unit/test_wan_student_train_worker.py tests/training/test_wan_adapter.py tests/unit/test_wan_training.py`

Expected: dispatch metadata proves the shared algorithm path, DMD2 is fail-closed without memory admission, and no v2 path uses the legacy direct-loss loop.

```bash
git add harness4h3/student/wan_training.py tools/wan_student_train_worker.py configs/student-campaign.wan-vllm-loop.yaml tests/unit/test_wan_student_train_worker.py tests/training/test_wan_adapter.py
git commit -m "feat: dispatch shared distillation algorithms for Wan"
```

## Task 4: Make the Wan vLLM campaign durable and strategy-aware

**Files:**
- Modify: `harness4h3/student/wan_campaign.py`
- Modify: `tools/wan_vllm_campaign.py`
- Test: `tests/unit/test_wan_campaign.py`
- Test: `tests/unit/test_wan_vllm_campaign.py`

**Interfaces:**
- Consumes: normalized proposals, trusted config digests, per-round artifacts, and existing evaluation/training results.
- Produces: `campaign-base.json`, `campaign-state.json`, `fingerprints.jsonl`, append-only `events.jsonl`, and resumable `campaign-result.json`.

- [ ] **Step 1: Add failing resume and duplicate tests.**

```python
def test_campaign_state_resumes_last_accepted_parent(tmp_path):
    write_complete_round(tmp_path, round_index=0, accepted=True, child_sha256="b" * 64)
    state = load_campaign_state(tmp_path)
    assert state["next_round"] == 1
    assert state["parent_checkpoint_sha256"] == "b" * 64

def test_campaign_marks_partial_round_failed_and_does_not_repeat_fingerprint(tmp_path):
    write_partial_round(tmp_path, round_index=1)
    record = register_fingerprint(tmp_path, "fp-1", parent_sha256="a" * 64)
    assert record["duplicate"] is True
    assert read_round_decision(tmp_path, 1)["reason"] == "incomplete_round"

def test_campaign_base_digest_change_requires_new_root(tmp_path):
    write_campaign_base(tmp_path, digest="base-a")
    with pytest.raises(ValueError, match="new campaign root"):
        ensure_campaign_base(tmp_path, digest="base-b")
```

- [ ] **Step 2: Run these tests and verify state helpers are missing.**

Run: `pytest -q tests/unit/test_wan_campaign.py tests/unit/test_wan_vllm_campaign.py`

Expected: the new state/fingerprint tests fail with missing helper or missing artifact behavior.

- [ ] **Step 3: Implement atomic state and append-only fingerprint helpers.**

Add `load_campaign_state(root)`, `write_campaign_state(root, state)`, `load_complete_rounds(root)`, and `register_fingerprint(root, fingerprint, metadata)`. State must contain `base_digest`, `last_complete_round`, `next_round`, `parent_checkpoint`, `parent_checkpoint_sha256`, `accepted_history`, and `failed_rounds`. Every JSON write uses the existing `atomic_write_json`; every event uses `append_event`. On startup, a round is complete only if proposal, training-result, evaluation-result, and decision files all exist and parse. A partial round gets one `round.incomplete` event and a deterministic `incomplete_round` decision; it is never reused as a parent.

- [ ] **Step 4: Thread state and capability context through `run_campaign`.**

At startup compute trusted digests for the config, training entrypoint, Wan config, Teacher checkpoint, evaluation manifest, CLIP model, capability snapshot, and evaluator recipe. Load or create `campaign-base.json`; reject a changed digest in the same output root. Before vLLM request, pass current parent evidence, accepted history, failed fingerprints, and capability snapshot. After proposal validation, skip a duplicate fingerprint and record `proposal.duplicate`; otherwise write `proposal.json`, run trusted training/evaluation, write all round artifacts, call existing deterministic `candidate_decision`, promote only accepted children, and atomically update `campaign-state.json`.

- [ ] **Step 5: Verify restart behavior and commit.**

Run: `pytest -q tests/unit/test_wan_campaign.py tests/unit/test_wan_vllm_campaign.py`

Expected: completed rounds resume without another vLLM call, partial rounds are recorded as failed, duplicate policy fingerprints are skipped, and base digest changes fail closed.

```bash
git add harness4h3/student/wan_campaign.py tools/wan_vllm_campaign.py tests/unit/test_wan_campaign.py tests/unit/test_wan_vllm_campaign.py
git commit -m "feat: persist and resume Wan strategy campaigns"
```

## Task 5: Add end-to-end contracts and documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-10-08-wan-native-algorithm-adaptation-design.md`
- Modify: `configs/student-campaign.wan-vllm-loop.yaml`
- Create: `tests/integration/test_wan_vllm_algorithm_loop.py`

**Interfaces:**
- Consumes: all previous tasks and fake vLLM/training/evaluation fixtures.
- Produces: a deterministic offline unit/integration contract proving that a v2 vLLM policy reaches the shared Wan algorithm path while preserving security and lineage fields. The test does not claim a real 30-second semantic gain without GPU evidence.

- [ ] **Step 1: Add an integration test with a local fake controller and fake worker.**

The test must assert:

```python
assert result["offline_simulation"] is False
assert result["algorithm_path"].endswith(("ProgressiveDistillation", "DMD2"))
assert result["parent_checkpoint_sha256"] == parent_digest
assert result["teacher_checkpoint_sha256"] == teacher_digest
assert result["continuation_verified"] is True
assert (output_root / "campaign-state.json").is_file()
assert (output_root / "fingerprints.jsonl").is_file()
```

- [ ] **Step 2: Run the integration contract and the relevant existing suites.**

Run:

```bash
pytest -q tests/training/test_progressive_distillation.py tests/training/test_dmd2.py
pytest -q tests/training/test_wan_adapter.py tests/unit/test_wan_training.py tests/unit/test_wan_campaign.py tests/unit/test_wan_vllm_campaign.py
pytest -q tests/integration/test_wan_vllm_algorithm_loop.py
```

Expected: all shared-algorithm regression tests and the Wan closed-loop contract pass. A real GPU smoke remains a separate evidence step and is reported explicitly if the local host lacks the four-GPU H3/Student setup.

- [ ] **Step 3: Update the approved design with implementation status and commit the final contract.**

Record only verified facts: which methods are executable for Wan, whether DMD2 is memory-gated, the exact schema version, and the commands used for validation.

```bash
git add docs/superpowers/specs/2026-10-08-wan-native-algorithm-adaptation-design.md configs/student-campaign.wan-vllm-loop.yaml tests/integration/test_wan_vllm_algorithm_loop.py
git commit -m "test: verify Wan vLLM shared-algorithm loop"
```

## Self-review checklist

- Progressive Distillation, DMD2, endpoint/velocity/clean-anchor, rollout, F1→F2→F3, initialization, precision, trainable scopes, continuation, lineage, and resume each have a named implementation task.
- The plan never adds a second training algorithm; Wan code is restricted to adapter, trusted boundary binding, schema, and supervisor state.
- v1 artifacts remain readable, while v2 is the only new controller output.
- No step relies on an unbounded vLLM path, command, device, model, Teacher, or code field.
- Tests cover the high-risk boundaries before implementation: latent geometry, binary schedule alignment, capability rejection, algorithm dispatch, checkpoint lineage, and restart deduplication.
