# Checkpoint Retention V1 Design

## Goal

Prevent rejected MiniMax-H3 A1 candidates from retaining large staged child
checkpoints indefinitely, while preserving the Harness4H3-v1.0 decision and
evaluation semantics.

## Constraints

- `research/experiments/a0_model_evolution.py` owns the lifecycle hook because
  A1 reuses `run_campaign`.
- `tools/h3_model_worker.py` continues to stage a child checkpoint under the
  experiment artifacts directory before evaluation.
- Cleanup runs only after the evaluator has returned and the campaign has
  classified the final outcome.
- `h3_training/engine/checkpoint.py`, the Evaluator, TargetProfile, reward,
  Controller decisions, and acceptance rules remain unchanged.
- M0000 and the active accepted model are never deleted by retention V1.
- No delta checkpoint, accepted-candidate quota, or Pareto-candidate deletion
  is included.

## Chosen approach

Add a focused retention helper in `a0_model_evolution.py`. It receives the
campaign output root, candidate/parent checkpoint paths, the finalized
outcome, and experiment artifacts. It resolves every local path before any
filesystem operation and only permits deletion when the resolved path is
strictly inside the resolved campaign output root. URI-like paths are refused.
The parent path and any path belonging to M0000 are protected before the root
check is used for deletion.

The helper returns a JSON-safe retention record with this stable shape:

```json
{
  "policy": "v1",
  "checkpoint_path": "...",
  "retained": false,
  "reason": "rejected_candidate",
  "deleted": true,
  "delete_error": null
}
```

For a rejected candidate, only the staged child checkpoint is a deletion
target. Its evidence sidecar is retained as reject evidence, while the
campaign record and trajectory retain the path, hash and evidence metadata.
Accepted candidates produce a retained record and are never deleted. Failed
experiments clean only checkpoint/optimizer-state files that were created under
this experiment's `output_root` subtree; experiment JSON, logs, metrics, and
failure evidence are retained. Missing files are already absent and produce a
successful no-op; unexpected deletion errors are recorded in `delete_error` and
never reported as successful deletion.

The campaign inserts the retention record into each iteration after evaluator
completion and final outcome classification, but before appending the record,
trajectory, and campaign persistence. The cleanup result is therefore
independent metadata and does not rewrite the ExperimentPlan or EvaluationResult
payloads.

## Safety behavior

1. Resolve `output_root`, candidate path, parent path, and every deletion target.
2. Refuse non-local/URI paths and paths outside `output_root` without unlinking.
3. Refuse the immutable parent path and M0000 path even if they are inside the
   output root.
4. Unlink files only; do not recursively delete directories or follow symlinks.
5. Record every requested target and its deletion result; preserve the main
   campaign result even if unlinking fails.

## Tests

Add focused campaign tests covering rejected local child deletion, accepted
child retention, M0000 protection, outside-root refusal, missing-file no-op,
and preservation of candidate metadata and trajectory after cleanup. Existing
campaign, worker, and checkpoint-resume tests remain unchanged and are run in
the full test suite.
