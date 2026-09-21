# External Model Operator Contract

`tools/h3_model_worker.py` is a safety/protocol adapter between Harness4H3 and
a trusted device-side trainer. It validates files, invokes a fixed trainer
argv, stages a returned child checkpoint, and normalizes the result. It does
not implement H3 training.

## Trust model

The Controller chooses only a registered operator and validated arguments.
The executable command is loaded from a trusted worker configuration such as
`configs/a1-worker.example.json`; it is never taken from Controller output.
The adapter uses `shell=False`, a fixed working directory, bounded JSON files,
captured stdout/stderr, and a timeout.

Fixtures under `tests/fixtures/` are protocol tests only. They are forbidden
for real experiment claims.

## Harness-to-worker request

The external operator supplies a JSON object containing at least:

```json
{
  "experiment_id": "exp_0001",
  "operator": "recovery_finetune",
  "operator_args": {},
  "parent": {
    "model_id": "M0000",
    "checkpoint_path": "/models/M0000.safetensors",
    "state": {}
  },
  "child_model_id": "M0001",
  "artifacts_dir": "/runs/exp_0001/artifacts"
}
```

The exact operator arguments are governed by the registered schema. The
parent checkpoint is read-only and `child_model_id` must identify a separate
artifact.

## Controller resource request

The Controller also declares the training resource policy. For elastic
distributed training, the preferred count and authorized range are explicit:

```json
{
  "gpu_count": 4,
  "min_gpu_count": 2,
  "max_gpu_count": 4,
  "elastic": true,
  "distributed": true,
  "exclusive": false,
  "on_unavailable": "wait"
}
```

The remote scheduler may queue the plan and retry it later. When at least the
minimum number of safe GPUs is available, it starts a new worker with the
actual allocation and matching `torchrun --nproc_per_node`. It does not
preempt unrelated jobs, and it does not resize a running worker. With
`elastic=false`, the preferred `gpu_count` is exact.

The remote Qwen/vLLM Controller has a separate queue launcher at
`tools/controller-wait-launch.sh`. It selects the smallest feasible 1/2/4-GPU
group, because the configured model has 16 attention heads and its tensor
parallel size must divide 16. For TP>1 it also scales vLLM's per-rank memory
reservation to the observed free VRAM. The launcher rechecks aggregate and
per-shard free VRAM before starting, returns failed vLLM startups to the queue,
and never preempts ComfyUI or unrelated GPU jobs. ComfyUI's GPU is not
permanently blacklisted: after the campaign calls `/free`, its queue must be
empty and only a small CUDA context may remain before the launcher can reuse
that card. A failed or ambiguous ComfyUI probe keeps the card blocked. The
Controller service itself is not a training worker: once it is available, it
emits the structured `ExperimentPlan` that the Harness validates and executes.

## Human next-plan directives

While a campaign is running, submit a bounded optimization objective to its
local output root:

```bash
python -m harness4h3 directive \
  --output-root var/remote-h3-controller-20260914 \
  --text "下一轮优先降低 peak_memory，质量下降不得超过 2%"
```

The command appends a `human_directive` observation to
`observations.jsonl`. The real LLM Controller sees it at the next planning
boundary and must consume its evidence ID in the validated `ExperimentPlan`.
It does not interrupt a running worker or directly select GPUs, commands,
checkpoints, evaluators, target-profile hard gates, or unregistered
operators. Repeating the same text (or reusing `--directive-id`) is
idempotent; use a new explicit ID when intentionally submitting a separate
objective.

## Controller reasoning budget

The vLLM planning path enables model reasoning with a bounded completion
budget: normal plans use 4096 tokens and plans triggered by a human directive
or recent failures use 6144. Short heartbeat reviews and connectivity probes
remain at 768 tokens with reasoning disabled. These limits increase useful
planning work without allowing one request to consume the full 16384-token
context window or hold the GPU indefinitely.

## Adapter-to-trainer request

The adapter preserves the validated request and adds:

```json
{
  "real_worker": true,
  "offline_simulation": false,
  "teacher_cache": {},
  "artifacts_dir": "/absolute/experiment/artifacts"
}
```

It invokes the configured trainer with:

```text
TRAINER_ARGV --request trainer_request.json --result trainer_result.json
```

## Trainer result

A successful trainer returns:

```json
{
  "status": "success",
  "output_state": {
    "model_id": "M0001",
    "parent_model_id": "M0000",
    "checkpoint_path": "/trainer/output/M0001.safetensors",
    "architecture_name": "MiniMax-H3",
    "provenance": {}
  },
  "cost": {
    "wall_time_s": 0.0,
    "gpu_hours": 0.0
  },
  "metrics": {}
}
```

For `recovery_finetune` and `step_distill`, `metrics` must include finite
training losses, a positive gradient norm, optimizer steps, trainable
parameter count, parent before/after SHA-256 digests, child hash, changed
tensor count, unchanged frozen-tensor count, and successful child reload.
Peak VRAM is included when measurable. These values must be measured by the
trainer, not synthesized by the adapter; the worker also checks the parent and
staged-child hashes independently.

`distill` uses the same distributed worker with a Controller-selected 2–4-rank
allocation, records a frozen-parent output target and its parent checkpoint
hash, and refuses a partial dataset fraction because the configured cache is
not a dataset loader. `prune_blocks` has no
gradient requirement: it must instead report removed/kept block indices,
reduced parameter and block counts, unchanged parent hash, changed child hash,
and a successful reload through the actual H3 loader. Zeroing weights is not
structural pruning.

`dmd2` uses the distributed H3 worker with a separately FSDP-sharded frozen
teacher, a trainable modality-preserving latent critic, alternating critic and
student updates, and explicit role-update counters. The child is still
benchmark-gated; the worker does not infer semantic quality from its training
loss. `quantize` is an explicitly configured trusted prebuilt-variant worker:
it verifies matching H3 metadata, quantization markers, source and parent
digests, then stages the immutable INT8/NVFP4 artifact. It does not claim to
have dynamically re-quantized arbitrary weights, and the benchmark must verify
that the staged child loads through the deployment path.

## Authenticity requirements

A real child is acceptable only when all of the following are evidenced:

- the parent hash is identical before and after execution;
- at least one intended trainable tensor differs from the parent;
- frozen tensors have no unexplained changes;
- loss is finite;
- at least one gradient norm is non-zero;
- at least one optimizer step completed;
- the child exists at a path distinct from the parent;
- the child reloads through the actual H3 loader; and
- the checkpoint is loadable by the benchmark deployment path.

The current adapter verifies path separation, file existence, H3 architecture
label, bounded cost fields, artifact staging, and parent-overwrite protection.
`tools/h3_real_train_worker.py` provides those weight-level and training-level
proofs for the remote MiniMax-H3 recovery/distillation path. The companion
`tools/h3_real_prune_worker.py` proves structural block removal, metadata
rewriting, parent immutability, child hashing, and reload through ComfyUI.

## TinyH3 reference validation

After installing `.[test,training]`, the complete two-process adapter chain
can be exercised with:

```bash
.venv/bin/python -m research.experiments.tiny_real_closed_loop \
  --output-root var/tiny-real-closed-loop
```

The fixed sequence performs recovery fine-tuning followed by one binary
progressive-distillation stage. Both children contain real PyTorch tensors and
are evaluated after reload. This is protocol and algorithm evidence only; it
is not MiniMax-H3 training, quality, compatibility, or hardware evidence.

## Stable adapter failures

| Failure type | Meaning |
|---|---|
| `worker_config_invalid` | No trusted worker configuration was supplied |
| `worker_contract` | Request/config/result path or JSON contract is invalid |
| `training_timeout` | Trainer exceeded the fixed timeout |
| `training_process` | Trainer exited nonzero |
| `missing_trainer_result` | Trainer produced no result JSON |
| `invalid_trainer_result` | Result lacks successful `output_state` |
| `invalid_training_evidence` | A training result lacks or contradicts required authenticity metrics |

Trainer-specific failures such as `training_oom`, non-finite loss, zero
gradient, immutable-parent violation, unchanged child, save failure, or reload
failure remain explicit when the trainer reports them. `training_process` is
used only when a nonzero trainer exit has no valid stable failure result.
Failures must not fall back to a fixture or copied checkpoint.

## Current implementation boundary

The checked-in remote workers satisfy the contract for real MiniMax-H3 BF16
training, structured block pruning, DMD2 reference updates, and configured
prebuilt quantized variants on the four-L40 host. The training path currently uses a
deterministic cache and trains the output heads; it therefore proves an
authentic optimization operation but not semantic quality improvement.
`prune_blocks` is structural and loader-compatible because it rewrites the H3
metadata and remaps complete block tensors. `prune_heads` and
`prune_channels` remain disabled until their coupled tensor shapes and loader
path are implemented.
