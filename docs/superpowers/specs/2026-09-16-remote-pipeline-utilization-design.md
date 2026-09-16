# Remote Pipeline GPU Utilization Design

## Goal

Make a remote SSH-hosted MiniMax-H3 campaign continue autonomously while
keeping the four-GPU machine work-conserving across planning, training, and
evaluation. The campaign must overlap independent pipeline stages when safe,
retain only useful checkpoints, and keep the local LLM and ComfyUI under the
same resource ownership protocol.

## Scope

In scope are the existing `RemoteCampaign`, its SSH resource scheduler,
ComfyUI leases, vLLM/Ollama Controller provider, speculative planning path,
checkpoint retention, and the remote overnight launcher. The default remote
workflow will support one evaluated candidate and one speculative successor in
flight. It will use independent benchmark tasks in parallel when they exist;
it will not duplicate a benchmark merely to inflate utilization unless a
statistics-repeat setting explicitly enables that behavior.

The Controller continues to emit only a structured `ExperimentPlan`. Trusted
worker entrypoints, benchmark workflows, paths, and safety thresholds remain
configuration-owned. The evaluator remains authoritative for quality and
hardware feasibility.

## Architecture

The campaign becomes a small durable pipeline with four resource classes:

```text
validated plan -> worker lease -> trusted training worker
                         |                 |
                         |                 +--> prefetch next validated plan
                         v
                 ComfyUI evaluation -> decision/archive/retention
```

At a boundary, the scheduler chooses a 2-, 3-, or 4-GPU training allocation
from the live waterline. If one GPU remains free, the Controller may run there
and prefetch the next plan. If training owns all four GPUs, the next plan is
generated before the worker lease is acquired and is persisted as a bounded
speculative plan. After training, evaluation obtains only the GPUs needed by
the available independent tasks; a completed successor may use the released
training GPUs while the current candidate is evaluated. A successor is always
an unevaluated branch until its own benchmark and fixed continuation policy
complete.

ComfyUI is an on-demand evaluator service. A campaign-owned process may be
started on a selected GPU immediately before evaluation, must unload models
with `/free` after its queue is empty, and is stopped after the configured idle
grace period. The lease marker and scheduler reservation are released only
after the memory waterline is verified. Foreign processes are never killed.

The Controller context remains bounded: current state, aggregate evaluation,
the latest eight relevant experience records, a short operator summary, and
observation IDs/hashes are sent to the local remote endpoint through SSH
forwarding. Raw media, model payloads, and full logs remain remote artifacts.

## Durability and failure handling

`campaign_state.json` stores the active cursor, pending/prefetched plan,
pipeline stage, worker result path, parent/child IDs, resource decision, and
lease token. Every state transition is atomic and mirrored in
`controller-events.jsonl`. A restart recovers a finished worker from its
result JSON, reclaims only expired campaign leases, and never repeats a
completed experiment. A failed worker becomes experience evidence; its
temporary weights and optimizer state are cleaned under the campaign root,
while logs and metadata remain.

## Checkpoint and utilization policy

The root, active accepted candidate, direct rollback parent, and necessary
Pareto candidates are protected. Rejected/failed staged payloads are removed
after classification; evidence sidecars and experiment recipes remain. The
default retained-weight cap is three non-root checkpoints. GPU power and
utilization are measured per device; 300 W and 100% are optimization targets,
not reasons to violate thermal, memory, power, or foreign-job isolation gates.

## Verification

Offline tests will cover stage transitions, 3+1 resource packing, full-4-GPU
pre-plan fallback, parallel independent evaluation, ComfyUI stop/unload
behavior, restart recovery, checkpoint protection, and bounded context. A
remote smoke run will confirm that the event stream records actual GPU
allocations, power samples, lease release, worker overlap, and the next plan
without requiring Codex to remain attached.
