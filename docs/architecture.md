# Architecture

Harness4H3 separates the research policy from the execution and measurement
mechanisms. The object being optimized is an H3 `ModelCandidate` combined with
a `SystemCandidate` runtime recipe; the Harness itself is not an experimental
variable.

## Component boundaries

| Component | Responsibility | May change during an experiment? |
|---|---|---|
| Controller | Proposes one structured `ExperimentPlan` from bounded context | Provider/model is fixed within a comparison |
| Validation pipeline | Enforces schema, policy, budget, operator, and cost constraints | No |
| Operator registry | Exposes the only executable interventions | Registration is fixed before a campaign |
| Device worker | Runs a trusted implementation selected by fixed configuration | Implementation is not supplied by the Controller |
| Benchmark adapter | Generates artifacts through H3/ComfyUI | No |
| Evaluator | Measures quality, validity, and hardware constraints | No |
| Model/System stores | Preserve immutable lineage and active pointers | Append-only candidates; pointer changes are atomic |
| Pareto archive | Retains feasible non-dominated candidates | Updated only from evaluator results |
| Observation stream | Carries hashes, bounded summaries, telemetry, artifacts, and feedback IDs | Append-only; large files remain URI references |
| Resource scheduler | Queues Controller-authorized resource requests and maps them to currently safe GPUs | Never kills or reassigns unrelated processes; elasticity is only applied at worker start |
| Trajectory | Records plans, execution, evidence, failures, and cost | Append-only |

## Data flow

```text
TargetProfile + DeviceProfile + current Model/System state
                              ↓
                 capability preflight (read-only)
                              ↓
 ControllerContext = goal + model/system state + budget + operators + prior evidence + Pareto
                              ↓
                        Controller
                              ↓
                     ExperimentPlan
                              ↓
 SchemaValidator → PolicyValidator → BudgetValidator → OperatorValidator
                              ↓
                   registered Operator only
                              ↓
          immutable child candidate or explicit failure
                              ↓
             benchmark → independent EvaluationRecord → Observation stream
                              ↓
 continuation policy → archive + trajectory + next Controller context
```

The device profile answers whether a required capability is present. The
TargetProfile defines success constraints. Keeping these separate prevents a
machine description from silently changing the scientific objective.

## A-Evolve-inspired round boundary

The remote campaign adopts the part of the A-Evolve-Training architecture
that is useful for expensive H3 trials without allowing an LLM to mutate the
execution substrate. The worker/evaluator configuration, benchmark recipe,
and parent checkpoint identity form the immutable substrate for a round. A
candidate gets its own lineage path and can be rejected or reclaimed without
rewriting the parent or the evidence stream.

The server-side Controller receives a strict `RoundPolicy` and a bounded
`DiscoveryDigest`. The policy limits operators, evaluation identity, GPU
overlap, and round budget; the digest summarizes recent recipes, failures,
Pareto evidence, observation IDs, and compact telemetry. Workers remain
memory-free with respect to the full history. Append-only experiment,
observation, evaluation, and controller-event records are the durable
research memory, while only their bounded view is placed in the next prompt.

This boundary also makes proxy drift observable: the fixed evaluator and
`RoundGate` remain authoritative, while the Controller may diagnose that a
proxy is no longer tracking the target and propose a bounded next-round
policy. It cannot lower a hard gate, change the benchmark identity, execute a
shell command, or promote a candidate by assertion.

The Controller is the only component allowed to decide optimization actions and
whether a training plan is elastic. A plan declares a preferred GPU count and,
when `elastic=true`, an explicit minimum/maximum range. The scheduler may only
queue, retry, map a validated request to currently available hardware, and run a
trusted command with the actual allocation. It never kills or reassigns
unrelated processes. Elasticity is applied between worker invocations, not by
hot-adding or removing GPUs from a running distributed job.

The resident vLLM Controller is normally launched with the smallest feasible
tensor-parallel group (usually TP=1), so one card is enough for the 35B FP8
planner while the remaining cards stay available to H3 workers. Its launcher
re-scans free memory on every service restart and may choose a different card
or TP=2/4 group. Before a worker starts, the scheduler publishes a short-lived
GPU lease; the launcher excludes that lease, eliminating a restart race. The
lease is released after the worker exits and expires automatically after a
bounded timeout.

Plan calls use one vLLM request with four batched structured candidates followed
by a short LLM selector request. This increases useful controller throughput
without creating four model replicas or four experimental branches. Known
context-local impossibilities are filtered before selection; the campaign's
full validator remains authoritative. ComfyUI's `idle_release` policy unloads
the model after evaluation and leaves only a small process CUDA context, which
does not reserve GPU0 from the dynamic scheduler. On-demand evaluator
launchers also enforce a finite lease age, so a crashed SSH campaign cannot
strand that process and its CUDA context indefinitely.

During a trusted training invocation, a cloned Controller provider asynchronously
prepares one validated speculative plan for the following training boundary.
The prefetch is kept in a separate audit trace and is reused only when the
active parent, training counter, and human-directive state still match. A
review-requested replan, lineage change, or new directive invalidates it, so
overlap reduces idle time without bypassing evidence or safety validation.

Checkpoint payloads use a bounded retention set in long campaigns. The active
model and one direct evaluated rollback parent are preferred, followed by the
current Pareto/recent candidates up to `checkpoint_retention.max_retained_checkpoints`.
Evaluation, experiment, observation, and experience metadata remain append-only
so the LLM retains reusable evidence even after an old multi-gigabyte weight
file is reclaimed. `keep_all` remains an explicit opt-out for archival runs.

## Trust boundaries

The Controller emits data, never shell commands. Local execution uses fixed
argument vectors with `shell=False`; worker/trainer commands come from trusted
operator configuration. Credentials are read from named environment variables
and are not placed in Controller context or trajectories.

The evaluator is authoritative. A Controller cannot modify quality thresholds,
promote a failed candidate, rewrite parent lineage, or substitute simulated
metrics for a real experiment. A real model-changing operator must return a
new checkpoint and authenticity evidence; copying the parent is not a valid
child.

## Research-preview core and research surface

`Harness4H3-v0.4` is a research preview. The Controller remains non-authoritative
for measurement, while the active pair-state, objective, evaluator, and
continuation interfaces may change only through predeclared research work.
Real MiniMax-H3 execution remains fail-closed per host capability gate. The
declared remote L40 host has passed the source-grounded adapter/worker gate;
other hosts still stop before model-changing execution if the same symbols,
checkpoint, or ComfyUI deployment contract are unavailable.

Research extensions belong in:

- `research/experiments/` for reproducible protocols;
- `harness4h3/operators/` or an external worker for a concrete bounded
  intervention;
- `configs/devices/` for machine facts and capabilities;
- `configs/targets/` for predeclared objectives;
- `research/evidence/` for measured results, including failures; and
- `research/history/` for completed design decisions.

## Capability boundary

The repository has a real H3 inference/benchmark path and trusted remote
workers. The Controller cannot provide executables; fixed worker configuration
maps each enabled operator to its launcher. The remote training worker proves
real forward, backward, non-zero-gradient optimizer updates, child save/reload,
and parent/child verification. The structural pruning worker proves reduced
H3 depth and reloadability. Quality improvement remains an evaluator result,
not a property inferred from the operation name.
