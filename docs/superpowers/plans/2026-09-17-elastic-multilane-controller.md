# Elastic Multi-Lane Controller Implementation Plan

> **Goal:** Close the remote optimization loop without Codex long-running supervision, while using safe elastic GPU allocation and bounded model/context retention.

## Phase 1: bounded evaluation and controller lifecycle

- [ ] Add a configured H3 task deadline to `RemoteCampaignConfig` and pass it to `MiniMaxH3Adapter`.
- [ ] Add exact ComfyUI prompt cancellation on timeout and tests for the timeout/cancel path.
- [ ] Make `controller-wait-launch.sh` non-blocking for an exact D-state child; poll orphan completion before launching a replacement.
- [ ] Add launcher contract tests and preserve the no-unrelated-process invariant.

## Phase 2: independent GPU fill branch

- [ ] Add a bounded `planning_intent`/operator filter to Controller context generation.
- [ ] Generate a second plan only when the primary overlap plan is CPU-only and spare GPUs meet the worker waterline.
- [ ] Give the fill branch a separate durable state file, experiment id, result path, lease and checkpoint lifecycle.
- [ ] Join, preserve or discard fill branches at the evaluation decision boundary without changing active/Pareto state early.
- [ ] Add unit/integration tests for CPU-only primary, GPU fill selection, lineage invalidation and resume recovery.

## Phase 3: remote validation

- [ ] Run focused and full local tests plus `git diff --check`.
- [ ] Sync only the implementation/config/test files to the isolated SSH deployment.
- [ ] Inspect the existing campaign and leases before any remote action; do not kill the old unrelated ComfyUI daemon.
- [ ] Run a bounded real evaluation/training boundary and verify event order, exact releases, checkpoint retention, and per-GPU telemetry.
- [ ] Leave the remote supervisor autonomous after validation; Codex only reports the evidence.
