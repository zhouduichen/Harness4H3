# Remote Validation Window Design

## Goal

Provide a remote-side, bounded validation command that can start one explicitly
authorized campaign window, collect four-GPU evidence, and leave the campaign
paused when the window ends. Codex only needs to inspect the resulting report;
it does not need to remain attached to the process.

## Scope and safety

- The command never starts by default. Starting requires `--start` and an
  operator-resumed campaign; an existing `.operator-paused` marker is a hard
  refusal.
- It refuses to start when `nvidia-smi` reports any compute process. It never
  kills or changes a process it did not start.
- The campaign receives an explicit maximum iteration count. A wall-clock
  timeout requests the same graceful stop marker used by the service.
- Cleanup only pauses the configured campaign and waits for its supervisor to
  drain. The owned Controller launcher remains managed by the existing service.
- Model/checkpoint bytes are not copied into the evidence directory. Evidence
  contains bounded summaries, event offsets, telemetry CSV, and references to
  the campaign state and event log.

## Components and data flow

1. `remote-validation-window.sh` validates the operator marker, campaign
   status, and GPU process gate; records the pre-window GPU snapshot.
2. It starts a bounded sampler for per-GPU power, utilization, and memory,
   then starts the existing campaign service with
   `REMOTE_CAMPAIGN_MAX_ITERATIONS`.
3. It polls service status until the bounded campaign completes or the wall
   clock expires. On exit it requests a graceful pause, records post-window
   GPU state, and stops its own sampler.
4. `summarize_remote_validation.py` reads only the event-log suffix created
   during this window, the state JSON, and telemetry CSV. It produces
   `validation-report.json` and a short Markdown report containing per-GPU
   averages/peaks, observed lane labels, key lifecycle events, and explicit
   `evidence_status` values. Missing telemetry or missing events remain
   `unverified`; they are never inferred as 100% utilization or 300 W.

## Acceptance evidence

The report can prove only what the live window measures:

- all configured GPU indices were sampled;
- power/utilization/memory rows exist and are summarized per GPU;
- worker/controller/ComfyUI lane metadata was observed when lease markers
  existed;
- prefetch, evaluation overlap, round-policy activation, and ComfyUI release
  events are listed when they occurred;
- the campaign reached a terminal result or was gracefully stopped at the
  configured limit; and
- the final service state is paused/not running.

It must not claim that every GPU reached 300 W or 100% utilization unless the
sampled rows satisfy a future explicit threshold and the report says which
threshold was used.

## Compatibility

The existing service and direct `run_overnight_controller.py` entry points keep
their behavior. The service only gains optional environment overrides for
`max_iterations`; absent that variable it continues to use the long-run
default. Existing paused campaigns and state formats remain readable.
