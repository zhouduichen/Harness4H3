# Wan Block Shared-Weight Capacity Diagnostic

## Purpose

The completed three-seed 2×2 confirmation shows that neither calibration shape nor teacher/student trajectory mixture closes the Semantic gap: every arm remains about 0.068–0.073 below the parent. This diagnostic isolates effective trainable Block capacity while holding total active Block executions, initialization source, calibration data, optimizer, update count, and video evaluation fixed.

## Compared programs

**One independent weight set:** the existing intermediate-shape teacher-only arm. One copy of parent `block.15` is executed at positions 0–23; positions 24–29 are identities.

**Two independent weight sets:** two separate modules, each initialized as an exact copy of parent `block.15`. The first is executed at positions 0–11 and the second at positions 12–23; positions 24–29 are identities. Both modules are independently trainable. Every denoising pass therefore executes 24 Blocks in both programs. The two-weight arm has twice the trainable Block parameters; its file size and existing promotion-gate outcome are measured and reported rather than assumed.

## Fixed protocol

- Training data: the existing 48-record `416×240×41` intermediate-shape teacher-only manifest, verified from its source digests. This is not called full-shape matching.
- Initialization: both independent modules copy the same source `block.15` tensors; no second source Block is introduced.
- Training: three paired training seeds `20261008`, `20261009`, and `20261010`; 500 updates; AdamW; learning rate `1e-5`; existing MSE teacher-recovery objective; same record weighting and draw rules as the intermediate teacher arm.
- Development evaluation: the same four development prompts, generation seed `20261008`, 832×480×81 output, and 16 denoising steps used in the prior confirmation. No frozen set is used for selection.
- Comparisons: each two-weight checkpoint is paired to the already completed one-weight `intermediate_teacher` checkpoint with the same training seed. Report per-prompt Semantic/Temporal/Motion differences, seed-level variation, trainable and serialized parameter counts, logical file size, latency, peak memory, and current promotion flags.

## Runtime representation and persistence

Add a diagnostic-only ordered runtime operation `block_program_reuse_groups` carrying the full depth, source labels, and group position lists. The live transformer uses shared references within each group but keeps the two group modules distinct. Save only one representative state entry for each independently trained group; replay creates each group from its own saved representative before inference. Reloaded output must match the in-memory output exactly within the existing numeric tolerance. Runtime provenance must record both groups and all 24 active positions.

The recovery path accepts exactly two distinct target modules for this operation and trains both. The capacity-preparation step creates a two-module checkpoint by retaining parent `block.15` and a separate copy of `block.15`; it records source checkpoint hash, source Block ID, clone relationship, module object identity checks, and unique parameter count. Existing one-group behavior remains unchanged.

## Execution stages and stop conditions

1. Build the two-module initialization checkpoint and verify bitwise-equal initial tensor values with distinct parameter storage.
2. Run a one-update intermediate-shape memory and serialization smoke on a dynamically leased idle GPU. Verify two independent groups, 12 positions each, 24 total calls, and exact save/reload output.
3. Run three 500-update two-weight training seeds using dynamically leased GPUs. Stop and preserve a failure record on OOM or any group/aliasing invariant failure; do not silently reduce shape, steps, or sample budget.
4. Evaluate the three candidates on the paired development set and compare to the matching one-weight checkpoints.
5. Report whether capacity is supported, weakened, or unresolved. A two-weight arm that misses the existing volume/promotion gate is diagnostic only and cannot be promoted.

## Risks and limits

The new grouped-sharing serializer must not alias the two groups, omit one representative, or reload both groups from the same tensors. The extra parameters may raise training memory or artifact size. Three training seeds on one fixed cached calibration corpus estimate optimization variation, not data-set variation. Even a clear capacity effect would not prove that capacity is the only limit or rule out compensation-objective issues.

## Acceptance evidence

- Machine-readable source digests, preparation metadata, all three run records, output checkpoints, evaluation summaries, and commands are retained under the semantic diagnosis experiment directory.
- Runtime provenance shows 12 uses of each of two distinct trainable groups, with 24 total calls.
- Initial module tensors are equal and storage is independent; both groups change during training.
- Each candidate reloads and produces finite 81-frame videos on the shared development cases.
- The Chinese diagnosis report states the measured capacity contrast, uncertainty, promotion/size-gate outcome, and whether the evidence supports this as the main limiting factor.
