# Wan2.1-T2V-1.3B 5 秒真实训练与蒸馏 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start a real 5-second Wan2.1-T2V-1.3B training run initialized from the official pretrained checkpoint and supervised by the real MiniMax-H3 Teacher.

**Architecture:** Keep the existing W0 Wan inference path unchanged. Add a separate Wan-native training worker that loads WanPipeline components, freezes the VAE/text encoder, trains the Wan Transformer with Flow-Match velocity loss, and obtains a real H3 Teacher target through the existing collective Teacher service and H3 VAE decode/re-encode boundary. Add an isolated 5-second remote experiment config and launch command so the existing H3 Student/30-second campaign remains untouched.

**Tech Stack:** Python 3.9+, PyTorch, Diffusers WanPipeline/WanTransformer3DModel/AutoencoderKLWan, safetensors, PyYAML, existing SSHClient/TeacherService, pytest.

## Global Constraints

- Wan model ID is `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
- Training contract is 81 frames at 16 FPS, 256×256, approximately 5.06 seconds.
- The Student starts from the real pretrained Wan checkpoint with lineage root `W0000`.
- The H3 Teacher must be real, collective, and online; `offline_simulation` must remain false.
- The Wan VAE and text encoder are frozen; only the Wan Transformer is trainable in the first run.
- Existing H3 Student and 30-second files, checkpoints, and output directories are not overwritten.
- No random-init fallback, synthetic cache, or fixed teacher target may be reported as a successful run.

---

### Task 1: Add Wan-native training primitives

**Files:**
- Create: `harness4h3/student/wan_training.py`
- Modify: `harness4h3/student/wan.py`
- Test: `tests/unit/test_wan_training.py`

**Interfaces:**
- `WanTrainingSpec.from_mapping(raw) -> WanTrainingSpec`
- `load_wan_training_components(spec, device) -> WanTrainingComponents`
- `encode_wan_video(vae, frames, spec, device) -> Tensor`
- `flow_match_batch(clean_latent, generator, scheduler, device) -> (noisy, timestep, target, sigma)`
- `wan_velocity_loss(prediction, target) -> Tensor`
- `save_wan_checkpoint(transformer, path, metadata) -> str`

- [ ] **Step 1: Write failing tests for the 5-second contract and Flow-Match target.**

```python
def test_wan_training_spec_uses_native_five_second_contract():
    spec = WanTrainingSpec.from_mapping({"model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"})
    assert spec.num_frames == 81
    assert spec.native_fps == 16.0
    assert 5.0 < spec.duration_s < 5.1


def test_flow_match_batch_target_is_noise_minus_clean():
    clean = torch.zeros(1, 16, 21, 32, 32)
    noise = torch.ones_like(clean)
    noisy, timestep, target, sigma = flow_match_batch(
        clean, generator=torch.Generator().manual_seed(1), scheduler=FakeScheduler(), device=torch.device("cpu"),
        noise=noise, sigma=torch.tensor([0.25]),
    )
    assert torch.allclose(noisy, torch.full_like(clean, 0.25))
    assert torch.allclose(target, torch.ones_like(clean))
    assert timestep.shape == (1,)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the Wan training module is absent.**

Run: `Harness4H3/.venv/bin/python -m pytest Harness4H3/tests/unit/test_wan_training.py -q`

Expected: FAIL with an import/definition error for the new Wan training interfaces.

- [ ] **Step 3: Implement the Wan-native component loader and Flow-Match helpers.**

Implementation requirements:

```python
def flow_match_batch(clean_latent, generator, scheduler, device, *, noise=None, sigma=None):
    noise = noise if noise is not None else torch.randn_like(clean_latent, generator=generator)
    scheduler.set_timesteps(1000, device=device)
    sigma = sigma if sigma is not None else torch.rand((clean_latent.shape[0],), device=device)
    timestep = sigma * float(scheduler.config.num_train_timesteps)
    broadcast = sigma.to(clean_latent.dtype).view(-1, 1, 1, 1, 1)
    noisy = (1.0 - broadcast) * clean_latent + broadcast * noise
    target = noise - clean_latent
    return noisy, timestep, target, sigma
```

The loader must call `WanPipeline.from_pretrained` using the configured local snapshot/model ID, put the transformer on the Student device, freeze VAE/text encoder, enable gradient checkpointing when available, and fail if the model cannot be loaded. It must not instantiate a random `WanTransformer3DModel`.

- [ ] **Step 4: Run focused tests and verify they pass.**

Run: `Harness4H3/.venv/bin/python -m pytest Harness4H3/tests/unit/test_wan_training.py -q`

Expected: PASS without downloading or loading a 1.3B checkpoint.

- [ ] **Step 5: Commit the primitive and unit tests.**

```bash
git add harness4h3/student/wan_training.py harness4h3/student/wan.py tests/unit/test_wan_training.py
git commit -m "feat: add Wan native training primitives"
```

### Task 2: Add the real Wan/H3 training worker

**Files:**
- Create: `tools/wan_student_train_worker.py`
- Create: `tests/integration/test_wan_student_train_worker.py`

**Interfaces:**
- CLI entrypoint `tools/wan_student_train_worker.py --config ... --teacher ... --cache-dir ... --output ... --result ...`
- Result JSON fields: `status`, `model_id`, `pretrained`, `lineage_root`, `teacher_checkpoint_sha256`, `student_checkpoint_sha256`, `optimizer_steps`, `teacher_forward_count`, `offline_simulation`, `frame_contract`, `loss`

- [ ] **Step 1: Write a failing worker contract test.**

The test invokes the worker with injected fake component factories, verifies that the real-mode boundary rejects an offline Teacher, and verifies that a successful fake run records `offline_simulation=false`, a changed parameter digest, and a positive Teacher forward count.

- [ ] **Step 2: Implement trusted argument/config validation.**

Reject any config whose frame contract is not 81/16 FPS/256×256, whose model ID is not the Wan model, whose Teacher world size is not three, or whose output path overlaps the existing H3/30-second result roots.

- [ ] **Step 3: Implement H3 Teacher target construction.**

Use the existing `TeacherService` and H3 cache item to obtain the cached H3 clean latent and prompt conditioning. Make one real H3 forward on the configured noisy latent, reconstruct a clean H3 latent with the H3 velocity convention, decode it through `decode_video_latent`, convert the decoded `[T,H,W,C]` RGB tensor to `[B,C,T,H,W]`, and encode it with Wan VAE. Keep the Teacher under `torch.inference_mode()` and record the service forward count.

- [ ] **Step 4: Implement one-step/full-step Wan training.**

For each step, encode the Teacher RGB clip once, sample a Flow-Match sigma/noise pair, compute Wan text embeddings with the frozen text encoder, run the trainable Wan Transformer, compute MSE against `noise-clean`, backpropagate, clip gradients, and update AdamW. Save the Transformer state as a safetensors child checkpoint with the W0000 parent digest and Teacher digest.

- [ ] **Step 5: Add checkpoint/result validation.**

Fail if no parameters changed, no online Teacher forward occurred, any tensor is non-finite, or `offline_simulation` is true. Write the result atomically even on failure.

- [ ] **Step 6: Run worker contract tests.**

Run: `Harness4H3/.venv/bin/python -m pytest Harness4H3/tests/integration/test_wan_student_train_worker.py -q`

Expected: PASS without loading the remote 1.3B checkpoint.

- [ ] **Step 7: Commit the worker.**

```bash
git add tools/wan_student_train_worker.py tests/integration/test_wan_student_train_worker.py
git commit -m "feat: train Wan Student with real H3 Teacher"
```

### Task 3: Add isolated 5-second experiment configuration and CLI

**Files:**
- Create: `configs/wan-student-5s-train.yaml`
- Create: `configs/student-campaign.wan-5s.yaml`
- Modify: `harness4h3/cli.py`
- Modify: `harness4h3/student/wan_experiment.py`
- Test: `tests/unit/test_wan_training_config.py`

**Interfaces:**
- CLI: `harness4h3 wan-student train --config configs/student-campaign.wan-5s.yaml --json`
- Config parser returns an isolated `WanTrainingExperimentConfig`.

- [ ] **Step 1: Write failing config and CLI tests.**

Assert the parser preserves the Wan model ID, W0000 lineage, 81-frame/16-FPS contract, real Teacher path, and a remote output root named `wan-student-5s`.

- [ ] **Step 2: Implement the isolated config parser and fixed remote command builder.**

Use the existing `RemoteConfig` path guards and `SSHClient`; construct only fixed worker arguments from the trusted YAML. Do not reuse the long-video Student config or its proposal/latent geometry.

- [ ] **Step 3: Add `wan-student train` to the CLI.**

The command validates the config, starts the remote worker, reads the result JSON, and returns non-zero on a failed experiment. Keep `wan-student run` as the existing W0 baseline command.

- [ ] **Step 4: Run config/CLI tests and commit.**

Run: `Harness4H3/.venv/bin/python -m pytest Harness4H3/tests/unit/test_wan_training_config.py -q`

```bash
git add configs/wan-student-5s-train.yaml configs/student-campaign.wan-5s.yaml harness4h3/cli.py harness4h3/student/wan_experiment.py tests/unit/test_wan_training_config.py
git commit -m "feat: add Wan 5s training experiment entrypoint"
```

### Task 4: Remote preflight and direct experiment start

**Files:**
- Modify only the new remote Wan 5-second files under the configured remote harness root.
- Create: `var/wan-student-5s/launch.json`

- [ ] **Step 1: Run the full local regression suite relevant to the changed paths.**

Run: `Harness4H3/.venv/bin/python -m pytest Harness4H3/tests/unit Harness4H3/tests/integration/test_wan_student_train_worker.py Harness4H3/tests/unit/test_wan_training_config.py -q`

Expected: all selected tests pass; existing H3 contract failures are reported separately and not “fixed” by changing their files.

- [ ] **Step 2: Verify the remote host, Python, Wan cache, H3 checkpoint, ComfyUI, and free GPU allocation.**

Use the configured SSH host and explicit paths. Do not start training if the Wan snapshot or H3 checkpoint is missing, if the Teacher cannot reserve three GPUs, or if the Student GPU has insufficient free memory.

- [ ] **Step 3: Copy only the new Wan training files and configs to the remote harness checkout.**

Verify remote SHA-256 values before launch. Do not copy or overwrite existing H3 campaign files.

- [ ] **Step 4: Start the first real Wan 5-second training run.**

Launch the worker with `--steps 1` as the direct smoke experiment first. The launch must use the real Wan pretrained snapshot, the real H3 Teacher checkpoint, three Teacher GPUs plus one separate Student GPU, and an isolated output root.

- [ ] **Step 5: Record the launch target and initial goal.**

The user-facing goal is: “从 `Wan2.1-T2V-1.3B-Diffusers` 的 W0000 预训练权重出发，在 Wan 原生 81 帧/16 FPS/256×256 空间，对真实 MiniMax-H3 Teacher 做一次真实在线蒸馏训练；验证至少一次 Wan Transformer 参数更新、至少一次 H3 Teacher forward，并产出可恢复的 5 秒 Student checkpoint。”

