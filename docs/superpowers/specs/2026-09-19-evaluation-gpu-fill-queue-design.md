# Evaluation-Phase GPU Fill Queue Design

## Problem

During a one-task held-out evaluation the safe layout is normally
`ComfyUI=1 GPU + Controller=1 GPU + trial=2 GPUs`. The current campaign can
still leave the two trial GPUs idle when the selected successor plan is
CPU-only and its separate GPU-fill plan is generated only after the benchmark
callback. Historical events show `resource_scheduled.actual_gpu_count=0` for
the primary branch while no parallel branch was ready.

## Design

When the current trusted worker is CPU-only (`prune_blocks` or `quantize`) and
the pipeline allows two in-flight branches, start the GPU-fill Controller
prefetch concurrently with the ordinary successor prefetch during that
worker. It uses the existing cloned local-LLM provider, the predicted child
lineage, the `parallel_prefetched_plan` state cursor, and the existing GPU
operator filter (`recovery_finetune`, `distill`, `step_distill`, `dmd2`). A
small in-memory job index lets the evaluation callback consume the same
handle while it is still running; it never opens a duplicate request.

The callback continues to use the scheduler for the final lease. It consumes
the persisted or ready GPU-fill plan only when parent, training-call cursor,
operator validation, memory waterline, and foreign-process checks match. The
primary CPU-only branch may continue for checkpoint/lineage progress, but it
does not count as GPU utilization. If no safe GPU plan or no two-card lease is
available, the campaign records the explicit idle/wait reason and does not
claim full utilization.

Once the successor plan is ready during a benchmark, the campaign-owned
Controller is handed off before the speculative worker acquires its lease.
With one evaluator on a four-card host this allows the worker to claim the
three remaining safe cards; the Controller overlap lane is retained only for
ordinary training boundaries where it is still needed to generate the next
plan.

The primary prefetch persistence boundary may additionally write exactly one
distributed GPU sibling from the same validated n-way response into the
isolated `parallel_prefetched_*` cursor. The complete candidate batch remains
transient; the sibling is bound to the same parent and training-call cursor,
and is still only a proposal until the normal worker, foreign-process, lease,
and evaluation gates accept it. This makes the overlap plan restart-safe and
prevents a second LLM request when the selected CPU plan has already yielded a
usable GPU alternative.

## Invariants

- At most one primary successor and one parallel GPU-fill successor exist for
  a boundary; both use distinct experiment IDs and bounded state cursors.
- The shared campaign worker lease has one logical owner at a time. A
  CPU-only primary completion cannot release a still-running GPU sibling, and
  a second speculative launcher records a bounded wait instead of overwriting
  the sibling's lease.
- A parallel plan is never executed without the normal schema, worker
  contract, capability, and scheduler lease checks.
- Evaluation tasks are not duplicated merely to inflate utilization. Multiple
  ComfyUI workers are used only for independent tasks.
- If the first evaluator selection is empty solely because the campaign-owned
  Controller currently reserves the candidate cards, the campaign releases
  that exact Controller lease and re-runs selection before declaring a resource
  wait. Foreign processes still fail closed and are never stopped.
- External processes are never killed, and an unknown GPU mapping fails
  closed.
- After all ComfyUI leases are successfully released, evaluator GPU indices
  are cleared from lane telemetry; a later training allocation cannot inherit
  a stale evaluator reservation hint.
- Power and utilization remain measured telemetry targets; no power limit is
  forced.
- The scheduler may attach one advisory `power.draw`/`utilization.gpu` sample
  per card to the next Controller context. Memory waterline, live compute
  process mapping, and leases remain the authoritative allocation gates.

## Verification

- Unit/integration tests prove eager parallel prefetch starts during a
  CPU-only worker, persists under `parallel_prefetched_plan`, and is consumed
  by evaluation without a second same-boundary request.
- Existing full regression remains green.
- Remote synchronization is performed with the operator pause marker and
  `REMOTE_START_CONTROLLER=0`; live 300 W/100% evidence remains a separate
  post-resume validation gate.
