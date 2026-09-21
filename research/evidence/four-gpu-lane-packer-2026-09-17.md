# Four-GPU lane-packing evidence — 2026-09-17

## Scope

Host: `Jiayu-intern` (`intern@152.136.172.95:50904`)

Remote campaign: `/home/intern/huangjiahao/Harness4H3-rsi/var/remote-h3-controller-20260914`

The campaign is detached on the SSH server and uses the server-local
Controller (`qwen3.5-controller`) through loopback port `8001`. Codex is not a
resident monitor or a required process in this workflow.

## Observed overlap

The remote event log contains successful overlap boundaries in which:

- one ComfyUI evaluator lane remained on its configured card;
- the Controller occupied one lane;
- a speculative distributed worker acquired the two remaining cards;
- a CPU-only primary plan triggered a parallel GPU-fill plan while the
  current boundary was still active.

Representative event records include `evaluation_started`,
`controller_plan_prefetch_ready`, `controller_plan_parallel_ready`, and
`speculative_worker_started`. The worker allocation was recorded as `[1, 2]`
in one boundary and as `[0, 2, 3]` in a later boundary.

The later three-card attempt also exposed a real race: a foreign process
entered GPU0 after the scheduler's first `nvidia-smi` snapshot, and the worker
failed with `training_oom`. This is not treated as utilization success.

## Safety and accounting changes

- The scheduler now performs a short second live snapshot before publishing a
  worker lease when the first snapshot is feasible. A newly busy card causes a
  resource wait instead of launching into a stale allocation.
- Evaluation keeps the primary configured worker for API compatibility but
  filters secondary workers against live compute processes, Controller leases,
  and the memory waterline. Unknown process-to-GPU mappings fail closed.
- `lane_allocation` events record the stage, ComfyUI workers, Controller lane,
  worker GPUs, and planned/effective resource requests.
- Per-GPU power telemetry exposes `power_mean_w`, `utilization_mean`,
  `sample_count`, and explicit `lane_unknown` fields. Missing lane metadata is
  not inferred as idle or fully utilized.
- Checkpoint retention and recipe/experience records remain bounded and
  separate: payload cleanup does not remove the evidence used by the next
  Controller plan.
- A persistent GPU wait is returned to the Controller after the configured
  three resource retries in the overnight profile. The next call uses the
  durable `resource_recovery_cpu` intent and filters candidates to registered
  CPU-only operators. If the parent is already at a configured quantization
  precision, that precision is removed from the filter before the LLM call.
- `controller_input` audit events now store observation IDs plus bounded scalar
  previews instead of embedding complete nested evaluation/power payloads. The
  append-only observation and experience stores remain the full source of
  truth for future planning.
- The campaign supervisor defaults to Controller ports `8001` and `8000`, so a
  detached restart preserves the live remote-LLM route.
- The Controller watcher also treats `8001` as the default external local-LLM
  endpoint, preventing a restart from launching a duplicate vLLM instance.

## Current boundary

At the latest live poll, campaign PID `1426178` was running with
`current_model_id=M0051`, `training_calls=57`, and `pipeline_stage=waiting`.
The real CPU recovery child `M0053` exists and is waiting for held-out
evaluation because the external Geneval workload still occupies GPU0--2 and
the server-local Controller occupies GPU3. No safe evaluator GPU is available;
the campaign therefore waits rather than stopping a foreign process or
claiming full utilization. The Controller preflight remains healthy on
`http://127.0.0.1:8001`.

## Real CPU-recovery validation

The latest remote boundary proved the new no-Codex recovery path:

- `resource_recovery_cpu` was requested after the bounded wait and the LLM's
  first `quantize(bits=8)` response was rejected because `M0051` was already
  8-bit.
- The next request dynamically filtered to `prune_blocks`; the LLM returned a
  validated plan for `exp_0060`.
- The trusted CPU worker completed successfully and produced `M0053`, reducing
  the checkpoint from 20 to 18 transformer blocks with real-worker evidence.
- While that worker ran, the campaign generated both a primary successor plan
  and an isolated `parallel_gpu_fill` plan. The parallel branch remained
  isolated until an evaluation boundary can safely launch it.
- A restart then consumed the stale recovery marker; the durable state now
  reports `resource_replan_intent=null`.

## Verification

- Focused remote-pipeline/controller regression suite after the lane,
  scheduler, bounded-replan, and audit-compaction changes: `118 passed`.
- Python compilation passed for the changed campaign, scheduler, and power
  telemetry modules.
- The 300 W value remains a target and telemetry field. No power limit is
  forced, and 100% utilization is claimed only when live per-GPU telemetry
  proves it.

## Dynamic full-card handoff — implementation boundary 2026-09-19

- Ordinary distributed plans retain the `3 worker + 1 Controller` overlap
  lane: the execution request is capped at three GPUs while preserving its
  validated minimum.
- A plan explicitly requesting `gpu_count` or `min_gpu_count` equal to the
  four-card scheduler capacity enters a separate full-card mode. The campaign
  generates and persists the successor plan first, writes the Controller
  handoff hold, releases only the campaign-owned Controller, and then acquires
  one disjoint four-GPU worker lease.
- `worker_resource_request_full_card`, `resource_scheduled`,
  `controller_release_requested`, and `worker_started` carry the
  `full_card_training` marker so later telemetry can distinguish the two
  modes. The remote campaign remains operator-paused; this boundary has local
  regression evidence only and does not claim live 300 W or 100% utilization.
