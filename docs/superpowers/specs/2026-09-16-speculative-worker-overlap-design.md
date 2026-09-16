# Speculative Worker Overlap Design

## Goal

Use GPUs that are free during the held-out benchmark to train the next LLM-selected experiment, while keeping the speculative result isolated until the current candidate has been evaluated and the Controller confirms that the lineage remains valid.

## Current gap

The campaign currently evaluates a child synchronously and only calls `_train_one()` after the benchmark and ComfyUI lease have finished. A prefetched plan can therefore be ready in `campaign_state.json` while the worker GPUs remain unused during the benchmark. The existing worker and checkpoint retention paths are synchronous and promote a child only through the normal evaluation decision.

## Design

### Speculative lifecycle

1. At the start of evaluating candidate `C`, reserve ComfyUI's GPU and decode the validated prefetched plan whose source child is `C`.
2. Rebase the plan's parent identity from the previous worker parent to `C`, preserving the operator, arguments, consumed observations, and experiment id.
3. Allocate only GPUs that are not reserved by ComfyUI, vLLM, or another worker. The request remains elastic: use the largest safe group in the plan's `[min_gpu_count, max_gpu_count]` range, with at least two GPUs for distributed H3 training.
4. Launch the trusted worker in a background thread. Its output checkpoint, result JSON, and lease remain isolated; it must not update the active model, Pareto archive, or evaluation records while speculative.
5. Complete the current benchmark. If it triggers a replan, critical regression, or lineage change, cancel the speculative process if still running and delete its unpromoted checkpoint under the existing retention policy.
6. If the current benchmark is accepted or Pareto-eligible and no replan is requested, join the speculative worker. Import its result as a normal experience, but leave it as an unevaluated candidate; the next cycle evaluates it and can promote it. If it completed before the benchmark ended, this removes the training idle gap.

### State and recovery

Persist a `speculative_worker` object in `campaign_state.json` with the experiment id, parent and source-child ids, child id, request/config/result paths, allocated GPUs, status, and start time. On resume:

- `running` or `launching` state is reconciled against the remote worker and result JSON;
- a completed result is imported once and remains an unevaluated candidate;
- a stale or cancelled state releases only this campaign's lease and removes only the exact speculative checkpoint;
- an active/promotion decision is never inferred from the presence of a speculative checkpoint alone.

### Safety rules

- Never use GPU0 while the ComfyUI benchmark lease is active.
- Never stop unrelated processes; a failed allocation waits or abandons the speculative attempt according to the plan's `on_unavailable` policy.
- Never promote a speculative child before the current candidate's benchmark decision is known.
- A replan, human directive, critical regression, or changed parent invalidates the speculative result.
- Checkpoint cleanup is restricted to the exact campaign child directory and uses `rejected_candidate_v1`.

### Events

Emit bounded events for `speculative_worker_started`, `speculative_worker_completed`, `speculative_worker_promoted`, `speculative_worker_discarded`, and `speculative_worker_recovered`. Existing `worker_*`, `resource_*`, and checkpoint-retention events remain the source of truth for the actual worker.

## Testing

- Unit-test rebasing and resource exclusion while ComfyUI owns GPU0.
- Integration-test that evaluation starts a speculative worker on remaining GPUs and that the current active model is unchanged until evaluation completes.
- Test discard on evaluation rejection/replan and exact checkpoint cleanup.
- Test resume from a completed speculative result without a duplicate worker or Controller call.
- Run the full repository test suite and a remote dry boundary check before starting a real overnight campaign.
