# Wan Block Weight Compensation Implementation Plan

> **For agentic workers:** Execute tasks inline in order. Each task ends with a focused verification checkpoint.

**Goal:** Add a bounded Block-local recovery training path and run matched prune-only, fixed-target recovery, and Controller-target recovery arms on a frozen Wan parent.

**Architecture:** A differentiable Wan adapter call and a small recovery trainer update only the selected retained Blocks against frozen parent denoising outputs from hash-verified calibration inputs. A standalone experiment runner uses the existing Controller provider and WAN video evaluator, saves immutable checkpoints and per-arm evidence, and leaves the main evolution loop unchanged.

**Tech Stack:** Python, PyTorch, existing Wan Diffusers adapter, `TICACalibrationManifest`, `WanVideoEvaluator`, and existing algorithm provider.

## Global Constraints

- Stay within transformer Block pruning and retained-Block weight updates.
- Keep the active parent checkpoint immutable and record its SHA-256.
- Use real Wan calibration tensors with verified hashes; fake inputs are only for unit tests.
- Keep the 0.98 quality floor and promotion policy unchanged.
- Never expose the held-out test manifest or results to the Controller.
- Do not change or discard unrelated uncommitted workspace edits.

---

### Task 1: Add a differentiable Wan denoising forward

**Files:**
- Modify: `harness4h3/adapters/wan/diffusers.py`
- Test: `tests/unit/test_wan_block_compensation.py`

**Interfaces:**
- Produces: `DiffusersWanAdapter.forward_denoising_grad(model, calibration_input, timestep) -> torch.Tensor`
- The method builds the same hidden-state, timestep, and encoder-conditioning inputs as `forward_denoising`, then calls the transformer without `torch.no_grad()`.

- [ ] Add a fake-transformer test asserting the returned output has a gradient path to its parameters and the existing `forward_denoising` output remains gradient-free.
- [ ] Run the focused test and confirm the new method is missing before implementation.
- [ ] Implement the method with the existing `_probe_output` normalization and existing Wan timestep conversion.
- [ ] Run the focused test and confirm both gradient assertions pass.

Run: `python -m pytest tests/unit/test_wan_block_compensation.py -q`

### Task 2: Implement targeted retained-Block recovery

**Files:**
- Create: `harness4h3/evolution/block_compensation.py`
- Modify: `harness4h3/adapters/wan/diffusers.py`
- Test: `tests/unit/test_wan_block_compensation.py`

**Interfaces:**
- Produces: `BlockRecoveryConfig(target_block_ids: tuple[str, ...], steps: int = 100, learning_rate: float = 1e-5, seed: int = 7)`.
- Produces: `BlockRecoveryResult(checkpoint: CheckpointArtifact, initial_loss: float, final_loss: float, gradient_norm: float, changed_tensor_names: tuple[str, ...], training_steps: int, metadata: Mapping[str, Any])`.
- Produces: `recover_pruned_blocks(adapter, teacher_artifact, pruned_artifact, removed_block_ids, calibration_manifest, config, output_dir) -> BlockRecoveryResult`.

- [ ] Add fake-Wan tests for target-ID validation, removed-ID rejection, original-ID-to-retained-index mapping, and no changes outside selected Blocks.
- [ ] Add a positive fake-Wan test proving finite loss, nonzero selected-Block gradients, a changed checkpoint hash, and reloadable output.
- [ ] Run focused tests and confirm the recovery module is missing.
- [ ] Implement teacher targets under `no_grad`, train only selected retained Blocks with fixed AdamW steps on real or fake manifest samples, and reject non-finite loss or absent gradients.
- [ ] Save through the adapter, reload, compare outputs within a fixed numerical tolerance, and emit hashes and tensor-level change metadata.
- [ ] Run focused tests and confirm the selected-only mutation and reload invariants pass.

### Task 3: Add a bounded Controller recovery-target response

**Files:**
- Modify: `harness4h3/evolution/agent.py`
- Test: `tests/unit/test_algorithm_program_search.py`

**Interfaces:**
- Produces: `AlgorithmAgent.choose_block_recovery_targets(*, parent_id, removed_block_ids, retained_block_ids, block_profile, probe_feedback, target_count=1) -> Mapping[str, Any]`.
- The provider task is `choose_block_compensation`; its only result fields are `target_block_ids` and `hypothesis`.

- [ ] Add tests that accept unique retained IDs of the required count and reject removed, unknown, duplicate, or out-of-scope IDs.
- [ ] Add a provider prompt requiring evidence-grounded retained-Block choices and JSON-only output.
- [ ] Run focused tests and verify invalid provider outputs fail closed without a fallback substitution.
- [ ] Keep fixed training hyperparameters outside the Controller response so arms use equal training budgets.

### Task 4: Add the paired experiment runner

**Files:**
- Create: `tools/wan_block_compensation_experiment.py`
- Create: `tests/integration/test_wan_block_compensation_experiment.py`
- Modify: `harness4h3/evolution/README.md`

**Interfaces:**
- Runner input: compression config, explicit prune Block IDs, hash-verified calibration manifest, dev/test evaluator configs, and output root.
- Runner arms: parent, prune-only, fixed nearest-retained-Block recovery, Controller-selected recovery.
- Runner output: `experiment.json` with parent/config/manifest/source hashes, proposal and validation traces, training summaries, checkpoint hashes, paired probe/video/hardware metrics, gate decisions, and failure classification.

- [ ] Add a fake-backend integration test asserting the four arms share one parent and one prune set, use equal training steps, and keep test results out of Controller payloads.
- [ ] Implement output-root isolation and refuse to overwrite an existing experiment.
- [ ] Create the pruned checkpoint once and branch both recovery arms from that exact checkpoint hash.
- [ ] Evaluate with paired prompts/seeds and enforce that every semantic/temporal/motion metric is present; report probe evidence separately from video evidence.
- [ ] Run the fake integration test before real-model execution.

### Task 5: Run the remote pilot and decide whether to continue

**Files:**
- Create: `configs/wan-block-compensation-pilot.yaml` (only after resolving actual remote paths)
- Create: `docs/superpowers/reports/2026-10-22-wan-block-compensation-pilot.md`

- [ ] Check the remote GPU queue and wait until the existing R155–R159 auto-resume batch is terminal; do not overlap latency measurements.
- [ ] Verify the calibration manifest's latent and conditioning files against their SHA-256 values and verify the chosen immutable parent hash.
- [ ] Run a bounded one-Block pilot for the historical `block.17` near miss. Stop on OOM, non-finite loss, zero gradients, out-of-target mutation, failed reload, or missing real calibration data.
- [ ] Compare prune-only, fixed-target, and Controller-target quality on dev; run the sealed eight-case/two-seed test only after target choices and checkpoints are frozen.
- [ ] Report paired effect sizes and prompt-bootstrap intervals. Continue to a two- or three-Block comparison only if fixed recovery shows a measurable held-out quality improvement while retaining an efficiency signal.
- [ ] Do not claim promotion unless the existing independent video and efficiency gates pass.
