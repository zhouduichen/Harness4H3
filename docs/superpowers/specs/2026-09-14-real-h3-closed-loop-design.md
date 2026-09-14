# Real MiniMax-H3 Closed-Loop Design

**Date:** 2026-09-14

## Goal

Move Harness4H3 from a protocol-only loop to a real MiniMax-H3 execution path:
the controller produces a bounded plan, a real H3 operator produces a verified
child checkpoint, an independent ComfyUI benchmark measures the child, and the
evaluator records the result for the next controller decision.

This phase has two distinct execution surfaces:

- `RealMiniMaxH3Adapter` implements the generic `h3_training` adapter contract
  for a real ComfyUI MiniMax-H3 model. It is the single-device/model API used by
  recovery fine-tuning and future training methods.
- `tools/h3_real_train_worker.py` remains the production L40x4 operator. It
  owns distributed initialization, FSDP, bounded smoke-cache training, parent
  immutability checks, child publication, and process-independent reload.

The worker and adapter share the same H3 conventions: ComfyUI model loading,
BF16 FL2VA tensors, data-ward velocity, video shift 12, audio shift 3, and
`PackedLayout` construction. Neither surface may synthesize quality or device
metrics.

## State and evidence boundaries

The active loop searches `(ModelCandidate, SystemCandidate)` pairs. A model
operator creates a new model checkpoint and paired system; a runtime operator
creates only a system. Every benchmark result is an `EvaluationRecord` keyed by
model ID, system ID, device ID, task split, and benchmark provenance.

Training evidence and benchmark evidence remain separate:

- Training evidence contains finite initial/final loss, positive gradient norm,
  optimizer steps, trainable/frozen tensor checks, parent/child hashes, reload
  status, peak VRAM, and wall/GPU time.
- Benchmark evidence contains quality raw metrics, aggregate quality score,
  latency, peak memory, model size, optional energy, task split, evaluator
  version, and validity/hard-gate results.

`EvaluationRecord.feasible` is determined only by the benchmark and target hard
constraints. Controller acceptance text is retained as a suggestion and audit
field; it cannot promote an invalid or infeasible child.

## Real adapter contract

`RealMiniMaxH3Adapter(comfyui_root, device="cuda", dtype=torch.bfloat16)` will
provide:

```python
load_role(path: Path, trainable: bool = False) -> ModelRole
prepare_batch(raw: Mapping[str, Any], generator: torch.Generator) -> PreparedBatch
add_noise(clean, noise, timestep) -> ModalLatents
predict(role, noisy, timestep, conditioning) -> ModalPrediction
prediction_to_clean(noisy, prediction, timestep) -> ModalLatents
scheduler_step(role, latent, prediction, interval) -> ModalLatents
schedule(num_model_evaluations: int) -> ModalSchedule
resolve_trainable_parameters(role, policy: str) -> Iterable[str]
save_role(role, path: Path) -> Mapping[str, Any]
reload_role(path: Path) -> ModelRole
```

The adapter loads the transformer config from safetensors metadata, imports the
ComfyUI MiniMax-H3 implementation dynamically from the configured root, loads
weights without fabricating missing tensors, and refuses unavailable CUDA or
missing ComfyUI/H3 symbols with a stable failure. Batch preparation accepts the
verified H3 cache schema used by the real worker and converts packed video/audio
rows into native ComfyUI tensors. The default trainable policy is the four H3
output-head tensors already used by the worker; arbitrary policies are accepted
only when they resolve to real model parameters.

The adapter is deliberately not a CPU/TinyH3 fallback. Tests use a small fake
ComfyUI module only to exercise shape, sign, and save/reload contracts; no such
fixture is emitted as real evidence.

## Real A1 path

The real campaign will use the active `OptimizationLoop` data model or an
equivalent adapter with the same semantics. It must:

1. initialize `M0000 + S0000` from the inspected parent checkpoint;
2. ask the controller for `exp_0001` with operator `recovery_finetune`;
3. execute the external real worker in the experiment artifact directory;
4. register `M0001` only after the worker returns a verified child state;
5. benchmark `M0001 + S0000` with the same benchmark recipe used for baseline;
6. write the canonical evaluation and continuation decision;
7. expose the result and relevant history when producing `exp_0002`.

If a GPU, ComfyUI service, worker checkpoint, or real task set is unavailable,
the command must fail closed and write the missing prerequisite as evidence. A
CPU test run may verify contracts and rejection behavior, but cannot be called
an A1 real-H3 completion.

## Duplicate and evidence rules

Each experiment computes a deterministic fingerprint over parent model/system,
operator, normalized arguments, target, device, and benchmark recipe. Exact
duplicates are rejected unless the plan explicitly sets
`repeat_for_statistics=true`; repeats receive a distinct experiment ID but retain
the same fingerprint and statistical purpose.

The real path must preserve `offline_simulation=false` and a real-worker marker
through operator result, model provenance, benchmark provenance, evaluation, and
trajectory. TinyH3 and fake metrics remain available only in explicit protocol
tests and cannot satisfy the real gate.

## Verification

The implementation is verified in layers:

- adapter unit tests with a fake ComfyUI module and real safetensors round trip;
- worker contract tests for parent immutability, finite loss/gradient evidence,
  optimizer steps, child hash, frozen tensors, and reload;
- active-loop integration tests proving `M0000 -> M0001`, independent
  benchmark evaluation, reject/keep, and second controller planning;
- CPU test suite, compileall, and diff checks;
- an explicit GPU gate command that runs the configured L40x4 worker and records
  whether the full real acceptance criteria passed or which external prerequisite
  blocked it.

No source change can claim the GPU gate passed without its persisted evidence.
