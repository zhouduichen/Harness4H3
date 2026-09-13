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
    "id": "M0000",
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

For a real model-changing experiment, `metrics` must include training loss,
gradient norm, optimizer steps, trainable parameter count, peak VRAM, parent
and child hashes, and changed tensor/parameter counts. These values must be
measured by the trainer, not synthesized by the adapter.

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
`tools/tiny_training_worker.py` now provides the weight-level and
training-level proofs above for the explicit `TinyH3` reference architecture.
The missing real MiniMax-H3 trainer must provide the same proofs for actual H3
weights.

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

Trainer-specific failures such as `training_oom`, non-finite loss, zero
gradient, immutable-parent violation, unchanged child, save failure, or reload
failure must remain explicit and must not fall back to a fixture or copied
checkpoint.

## Current implementation boundary

No checked-in trainer currently satisfies this contract for real MiniMax-H3 weights.
The RTX 5080 reconnaissance found that the official BF16 transformer alone is
about 61.73 GiB, beyond the measured 15.92 GiB VRAM and 31.45 GiB RAM. Real
`recovery_finetune`, pruning, and distillation therefore remain disabled in
the device profile.
