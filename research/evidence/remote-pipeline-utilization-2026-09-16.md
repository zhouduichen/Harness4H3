# Remote pipeline utilization evidence — 2026-09-16

## Scope

Host: `Jiayu-intern` (`intern@152.136.172.95:50904`)

Remote repository: `/home/intern/huangjiahao/Harness4H3-rsi`

Staged pipeline: `/home/intern/huangjiahao/Harness4H3-rsi/work/remote-pipeline-v2-20260916`

Legacy campaign output: `/home/intern/huangjiahao/Harness4H3-rsi/var/remote-h3-controller-20260914`

The staged pipeline is intended to run locally on the SSH server with
`run_overnight_controller.py --on-server`; Codex is not required to remain
attached. The Controller remains a structured remote vLLM client, while paths,
workers, benchmark settings, and resource limits remain trusted configuration.

## Offline and staging verification

- Relevant local regression suite: `74 passed`.
- Local Python compilation and shell syntax checks passed.
- Remote Python compilation passed for the staged campaign, provider,
  checkpoint helper, and training worker.
- The staged campaign files were hash-matched between the local checkout and
  the remote staging directory.
- Remote checkpoint-helper smoke passed: the first invocation reported
  `copied`, the second reported `cached`, and the cache contained one staged
  `.safetensors` entry. The temporary fixture was removed afterward.

## Implemented control-plane properties

- The remote campaign persists bounded observations, experiment recipes,
  evaluation summaries, power/utilization summaries, and retrieved prior
  experience. The LLM prompt receives a compact bounded view rather than raw
  media or full logs.
- A cloned Controller plans the next experiment during training/evaluation;
  the validated plan is persisted and reused only when parent, evidence, and
  training-call cursors still match.
- When the current trusted worker is CPU-only, the campaign now starts the
  filtered GPU-fill successor concurrently with the primary prefetch. It uses
  the bounded `parallel_prefetched_plan` cursor and the evaluation callback
  reuses that exact request instead of waiting for a second same-boundary LLM
  call. CPU-only work is never counted as GPU utilization.
- Evaluation reserves only the needed ComfyUI worker cards. During a
  benchmark lease, the Controller launcher caps its tensor parallel group so
  at least two cards remain available for distributed training.
- The scheduler allocates an elastic 2–4-card worker from a fresh memory
  waterline snapshot and publishes a campaign-owned worker lease.
- ComfyUI is released with `/free` using `unload_models=true` and
  `free_memory=true`; the reservation is released only after the memory
  waterline is verified. A campaign may stop only a ComfyUI launcher PID that
  it explicitly owns.
- Checkpoint staging uses a shared lock, atomic replacement, and a one-entry
  local cache. Retention keeps a bounded rollback/Pareto set while preserving
  recipes and evidence.
- The `300 W` value is recorded as a target/telemetry field. No power limit is
  forced, and 100% utilization is not claimed without measured evidence.

## Live handoff boundary

At the latest remote poll on 2026-09-16 22:33 CST:

- legacy supervisor PID `202320` was still alive;
- handoff watcher PID `321486` was alive and logging `lock=held
  exact_supervisor=present`;
- the legacy campaign lock was still held;
- v2 had not yet been activated, so no real v2 multi-round overlap claim is
  made.

Earlier legacy telemetry showed a 3-GPU worker on GPUs 0, 2, and 3 reaching
100% utilization while loading/training, and separate intervals with GPUs 2
and 3 idle during checkpoint staging. Those observations motivate the v2
prefetch/lease changes but are not v2 acceptance evidence.

The watcher is expected to activate the staged pipeline only after the legacy
supervisor and its lock have naturally released. An immediate migration would
require an explicit graceful-stop authorization; no such stop was issued in
this record.
