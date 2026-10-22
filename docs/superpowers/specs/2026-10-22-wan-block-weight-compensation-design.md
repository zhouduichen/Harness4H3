# Wan Block Weight Compensation Experiment

## Goal

Test the missing part of the Block-only Wan work: after a fixed set of transformer Blocks is removed, can short distillation of retained Blocks recover quality without restoring the removed computation, and can the Controller choose useful recovery targets?

This experiment does not repeat the already-run Block candidate-search study. R0015–R0024 tested Block plan generation and forward probes; the October 17 design expressly prohibited weight fine-tuning. R150–R154 used mixed operators and do not answer the Block-weight recovery question.

## Hypotheses

- H1: Fine-tuning one retained Block after removing a fixed Block improves paired forward-probe similarity versus the same pruned checkpoint without training.
- H2: With the pruned set and training budget fixed, the Controller-selected recovery target improves held-out video quality over a fixed nearest-retained-Block target.
- H3: Recovered quality does not remove the inference or model-size benefit of pruning.

## Scope and invariants

- Only Wan2.1-T2V-1.3B transformer Blocks are removed or fine-tuned. No Head, Token, Attention, Cache, or timestep operators.
- Use one clean immutable parent checkpoint with no inherited runtime compression. Record and verify its SHA-256.
- In the pilot, prune the historical near-miss `block.17`; retrain retained `block.18` for the fixed-target arm. The Controller may choose one valid retained Block. If `block.18` is unavailable, use the nearest retained Block by original index and record the fallback.
- Keep optimizer, learning rate, steps, samples, timesteps, and trainable-Block count fixed across fixed-target and Controller-target arms. Only the recovery target differs.
- Remove and recovery targets are specified in original Block IDs. Map them to post-pruning indices explicitly; a removed Block can never receive gradients.
- Train only on a hash-verified real-Wan calibration manifest. The Controller receives calibration measurements and Block profile data, never held-out video results.
- Keep the existing 0.98 parent-relative quality floor and promotion gates unchanged. Forward probes screen candidates; they do not prove video quality.

## Experiment arms

1. **Parent:** unmodified immutable checkpoint, used as the paired reference.
2. **Prune only:** remove the fixed Block set; no training.
3. **Fixed recovery:** same pruned checkpoint; fine-tune one fixed nearest retained Block.
4. **Controller recovery:** same pruned checkpoint; Controller chooses one retained Block using the allowed Block profile and probe feedback; fine-tune with the identical budget.

Arms 2 vs 3 test whether weight compensation works. Arms 3 vs 4 test whether Controller target selection adds value. Since the pilot removes one Block, it is a mechanism test and cannot count as a successful efficiency promotion by itself. If compensation works, repeat the same matched arms on a predeclared two- or three-Block plan before claiming useful compression.

## Compensation method

Use teacher-student denoising-output MSE. Cache parent outputs for each calibration latent/conditioning pair at low, middle, and high noise timesteps. In the pruned student, enable gradients only for the selected retained Block and freeze every other parameter. Use a fixed short AdamW schedule (pilot default: 100 steps, learning rate 1e-5, batch size 1, fixed seed). Save ordinary updated Block weights into the checkpoint; do not add an inference-time module. Record per-step loss, gradient norm, changed tensor names, memory use, checkpoint hash, and post-reload output agreement.

The real Wan adapter's ordinary profile forward is under `torch.no_grad`; implementation must add a dedicated differentiable path. If the pilot has no finite loss, no gradients, no changed selected weights, an out-of-target weight change, a failed reload, or an OOM, classify it as an implementation/feasibility failure, not as a negative AI result.

## Data and evaluation

- Use the existing distinct eight-case/two-seed RSI dev and test manifests. Use dev for screening; keep test sealed until Controller targets and checkpoints are frozen.
- Reuse fixed Wan denoising calibration tensors only after verifying source hashes. If no complete real calibration manifest exists on the GPU host, stop before training and report the missing artifact.
- Run paired parent/candidate video generation with identical prompt and seed for each arm. Report semantic, temporal, motion, aggregate quality, latency, peak memory, model size, and per-case deltas. Bootstrap by prompt and report intervals; do not claim Controller superiority if the interval for the paired difference includes zero.
- Require every quality metric to be present. The current motion score is mean adjacent-frame pixel change, so blindly inspect finalists for flicker or collapse.
- Promotion requires the existing per-metric 0.98 parent-relative floor and at least one existing 5% latency, 5% peak-memory, or 10% model-size objective, plus existing regression/Pareto gates.

## Success and stopping rules

- **Engineering success:** gradients are finite and nonzero only in the selected retained Block; selected weights change; checkpoint reloads; inference has no additional module; the pruned structure remains intact.
- **Compensation effect:** fixed recovery improves over prune-only on held-out metrics at identical pruning. Controller target selection is useful only if it improves over fixed recovery with matched compute and its held-out paired interval excludes zero.
- **Compression result:** report a promoted optimization only if existing video and efficiency gates pass. Quality recovery without an efficiency objective is a mechanism result, not a compression win.
- Stop after the pilot if the training path is infeasible. Stop a Controller selection retry after two invalid/out-of-scope targets; do not silently substitute its choice. Do not run the larger multi-Block comparison unless the one-Block pilot shows a measurable recovery signal.

## Out of scope

- Repeating the exhausted R0015–R0024 Block-set search.
- Changing parent, quality floor, promotion thresholds, or video metric implementation.
- Expanding the operator family or using test results to adapt future Controller choices.
