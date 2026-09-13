# 4×L40 Training Preflight Design

Status: approved direction; implementation pending.

## Purpose

Prepare Harness4H3 for a future four-GPU NVIDIA L40 training host without
claiming that the unconfigured machine can already run a real model-changing
experiment. This phase creates executable preflight checks and configuration
contracts only. It does not implement an H3 trainer or start A1-T0.

## Known and unknown facts

Known from the operator:

- the intended host has four NVIDIA L40 GPUs;
- available memory was described informally as approximately 40 GiB per GPU;
- host operating system, actual per-GPU memory, system RAM, CUDA/PyTorch
  versions, filesystem paths, services, and interconnect have not been
  configured or measured.

The checked-in configuration therefore records requirements and expected
topology, not measured results. Values observed during host setup from `nvidia-smi`, the
operating system, and installed runtimes replace the unverified status only
after preflight evidence is persisted.

## Scope

Create three configuration surfaces and extend the existing read-only
preflight:

1. `configs/devices/l40x4-server.yaml` describes expected hardware,
   environment requirements, paths, services, and disabled capabilities.
2. `configs/experiments/a1-t0-l40x4.yaml` fixes the first distributed
   recovery-fine-tuning smoke recipe.
3. `configs/a1-worker.l40x4.example.json` supplies a fixed Linux `torchrun`
   argv and trusted artifact/deployment paths for the existing
   `tools/h3_model_worker.py` adapter.
4. `tools/device_preflight.py` gains local GPU, RAM, runtime, and operator-path
   checks without accepting arbitrary commands from YAML.

No Harness protocol, Controller schema, evaluator, archive, trajectory,
TargetProfile, or model-worker contract changes are included.

## Device profile

`configs/devices/l40x4-server.yaml` uses the existing device-profile schema and
adds explicit verification metadata:

```yaml
schema_version: 1
id: linux-l40x4-training-server
verification:
  status: pending
  evidence_path: var/preflight/l40x4.json
platform:
  os: linux
  python: unverified
  pytorch: unverified
  cuda: unverified
hardware:
  gpu_name_contains: NVIDIA L40
  gpu_count: 4
  min_vram_gib_per_gpu: 40
  min_system_ram_gib: 128
  distributed_backend: nccl
```

The RAM value is a minimum deployment requirement for loading/consolidating a
roughly 61.73 GiB BF16 transformer with headroom; it is not a claim about the
unconfigured host. The GPU memory field is a minimum accepted value, not an
assertion that a standard L40 has exactly 40 GiB.

Paths are operation-scoped:

- repository and official Diffusers H3 model directory are required for
  `recovery_finetune`;
- trainer executable and smoke recipe are required for
  `recovery_finetune`;
- ComfyUI and deployment directories are required for `benchmark`;
- artifact and teacher-cache directories may be created by trusted setup but
  are still checked before execution when marked required.

All model-changing capabilities remain disabled:

```yaml
capabilities:
  recovery_finetune:
    enabled: false
    reason: pending measured hardware/runtime preflight and real trainer deployment
  prune:
    enabled: false
    reason: no real H3 pruning backend is registered
  distill:
    enabled: false
    reason: no real H3 teacher/student training backend is registered
```

Passing hardware checks does not automatically enable a capability. Enabling
`recovery_finetune` is a separate reviewed edit after the real trainer exists
and its isolated smoke test passes.

## A1-T0 distributed smoke recipe

`configs/experiments/a1-t0-l40x4.yaml` declares a correctness experiment, not
a quality-improvement campaign:

```yaml
experiment_id: A1-T0-L40X4-SMOKE
operator: recovery_finetune
model:
  implementation: diffusers_minimax_h3
  variant: FL2VA
  checkpoint_format: diffusers
  dtype: bfloat16
  trainable_scope: heads
distributed:
  launcher: torchrun
  strategy: fsdp_full_shard
  world_size: 4
  backend: nccl
  use_orig_params: true
memory:
  micro_batch_size: 1
  gradient_accumulation_steps: 1
  gradient_checkpointing: true
  precompute_text_embeddings: true
  precompute_vae_latents: true
optimization:
  optimizer: adamw
  learning_rate: 0.000001
  max_steps: 1
  seed: 20260913
data:
  sample_count: 1
  num_workers: 0
checkpoint:
  save_distributed: sharded
  consolidate_child: true
  export_format: safetensors
  require_diffusers_reload: true
  require_comfyui_reload: true
```

FSDP full sharding is required because ordinary DDP replicates the complete
BF16 transformer on every GPU. Only the actual model implementation may
resolve `trainable_scope: heads`; configuration must not guess module names.

Required trainer evidence is finite initial/final loss, at least one non-zero
gradient norm, one optimizer step, trainable parameter count, peak memory per
rank, wall/GPU time, unchanged parent hash, changed expected tensors, no
unexpected frozen-tensor changes, child hashes, and successful Diffusers plus
ComfyUI reload.

## Worker configuration

`configs/a1-worker.l40x4.example.json` uses an explicit Linux deployment
layout under `/opt/Harness4H3`, `/opt/h3-training`, and `/opt/ComfyUI`. The
trainer command is a fixed argv equivalent to:

```text
/opt/h3-training/.venv/bin/torchrun
--standalone
--nproc_per_node=4
/opt/h3-training/train_worker.py
--config
/opt/Harness4H3/configs/experiments/a1-t0-l40x4.yaml
```

`h3_model_worker.py` appends `--request` and `--result`. The example remains
non-runnable until those explicit paths exist. It never falls back to the
test fixture trainer.

## Preflight behavior

The existing command remains:

```text
python tools/device_preflight.py --profile PROFILE --operator OPERATOR --json
```

For profiles declaring local hardware requirements, preflight performs these
additional read-only checks:

1. run the fixed argv `nvidia-smi --query-gpu=name,memory.total
   --format=csv,noheader,nounits`;
2. require exactly the declared GPU count;
3. require every GPU name to contain the declared model text;
4. require every GPU to meet `min_vram_gib_per_gpu`, converting reported MiB
   to GiB;
5. on Linux, read `/proc/meminfo` and require `min_system_ram_gib`;
6. import PyTorch in the active Python process and record version, CUDA
   availability, CUDA build, and visible device count;
7. apply `required_for` filtering to path checks, matching existing service
   filtering;
8. confirm that a requested disabled capability remains a blocking result.

The preflight uses no profile-supplied executable command. Missing
`nvidia-smi`, malformed output, insufficient/mismatched GPUs, insufficient
RAM, unavailable CUDA, missing paths, unreachable services, or disabled
capability produce stable failed checks. No check starts training or mutates a
checkpoint.

`--skip-services` skips HTTP probes only; it does not skip hardware, runtime,
path, or capability checks. A separate `--skip-hardware` option is allowed for
schema/unit validation and must emit skipped checks rather than `passed`; a
profile with required hardware checks remains `blocked` when they are skipped.

## Stable check names

The JSON result extends the existing checks with:

```text
hardware.gpu_count
hardware.gpu_name
hardware.gpu_vram
hardware.system_ram
runtime.torch
path.<name>
capability.<operator>
service.<name>
```

Overall status remains `ready` only when no check is failed and no required
check is skipped. Skipped checks do not turn a device or disabled capability
into ready.

## Testing

Unit tests use injected/mocked `nvidia-smi`, `/proc/meminfo`, and PyTorch
observations. They cover:

- four matching GPUs with sufficient memory;
- wrong GPU count;
- wrong GPU model;
- one under-memory GPU;
- missing or malformed `nvidia-smi`;
- insufficient system RAM;
- CUDA unavailable or wrong visible-device count;
- operation-scoped path selection;
- disabled `recovery_finetune` despite passing hardware checks; and
- the existing RTX 5080 profile behavior.

The full offline test suite and `compileall` must continue to pass. Running the
new profile on the current Mac or an unconfigured server must return
`blocked`, which is the expected safe result.

## Definition of done

- The three L40 configuration files exist and parse.
- No field claims an unmeasured runtime version or deployed trainer.
- The smoke recipe fixes FSDP full sharding, four ranks, BF16, head-only,
  micro-batch one, precomputed conditioning, and one optimizer step.
- Preflight measures local GPU count/name/memory, Linux RAM, and PyTorch/CUDA
  visibility using fixed read-only mechanisms.
- Paths are filtered by requested operation.
- Real model-changing capabilities remain disabled.
- Tests prove both passing hardware observations and stable blocked outcomes.
- Harness4H3-v1.0 core contracts remain unchanged.
- The user's existing uncommitted M6 changes remain untouched.
