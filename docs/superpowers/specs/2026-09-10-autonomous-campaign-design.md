# Fixed-Harness Autonomous H3 Optimization Campaign

## Goal

Run a bounded, execution-grounded autonomous optimization campaign starting
from the validated NVFP4 H3 candidate under `Harness4H3-v1.0`, with the strict
RTX 5080 TargetProfile constraint `peak_vram_max_gb <= 16.0` preserved.

## Frozen boundary

The campaign must not modify the Controller protocol, ExperimentPlan schema,
Model/Runtime state schema, evaluator acceptance semantics, archive or
trajectory semantics, TargetProfile semantics, or Controller loop. Any code
change is limited to the campaign entry point's evidence bookkeeping or a
minimal execution wrapper for an already-defined registered operator.

The campaign's Harness version is always `Harness4H3-v1.0`, with status
`frozen` and change policy `bugfix_only`.

## Controller context and autonomy

For every experiment the Controller receives only:

- the fixed TargetProfile;
- current system/model state;
- registered operator schemas;
- previous experiment history and failure evidence;
- validated Design Genes and experiment summaries;
- current budget state.

The human does not select operators, operator arguments, or the next
experiment. The Controller diagnoses the remaining hard violation, proposes a
hypothesis, selects one registered operator, emits an ExperimentPlan, and
receives the execution result before choosing the next experiment.

The rejected `vae_tiling` result is read-only negative experience. The same
operator/configuration is not retried unless the Controller supplies a new
evidence-based hypothesis and materially different configuration.

## Failure, rejection, and parent selection

The campaign distinguishes a completed but infeasible candidate from an
execution failure:

- `rejected_candidate` means the operator ran, artifacts were valid, and one
  or more strict acceptance gates failed. This consumes experiment budget but
  does not consume `max_failed_experiments`.
- `failed_experiment` means the Controller emitted an invalid/duplicate plan,
  the operator could not execute, the backend crashed or timed out, an OOM or
  artifact-validity failure occurred, or another execution-level failure
  prevented a valid candidate evaluation. This consumes both experiment and
  failure budget.

Every experiment has a machine-checkable fingerprint:

```text
operator
+ normalized_operator_args
+ parent_state_digest
+ target_profile_id
```

An identical fingerprint is never retried. If the same operator is selected
with different arguments, the Controller hypothesis must cite a prior
experiment id for that operator and explain the materially different
mechanism expected from the new configuration. A plan that violates either
rule is an invalid Controller plan and is recorded as a failed experiment.

Rejected runtime candidates never replace the validated NVFP4 state as the
active baseline and are never used as an implicit parent for the next
experiment. The fixed Controller context exposes archive/Pareto summaries,
while the campaign supplies only an accepted active parent as
`current_system_state`; this respects the frozen `ExperimentPlan` parent-id
contract without adding a parent-selection field. Thus the effective search
shape is a set of branches from the validated active state, or from a later
strictly accepted state, rather than an unbounded chain through rejected
runtime candidates. Parent choice is therefore determined by persisted
accepted archive state, never by human intervention.

## Campaign budget and stop rules

- Maximum experiments: `8`.
- Maximum failed experiments: `5`.
- Acceptance splits: `dev` and `heldout`.
- Stop immediately on TargetProfile satisfaction.
- Stop when either budget is exhausted or the execution/controller failure
  limit is reached.

Each candidate is evaluated independently on both acceptance splits. The
active baseline remains the validated NVFP4 model/runtime state unless a
candidate passes the strict acceptance gate.

## Acceptance target

The campaign succeeds only when both splits satisfy:

```text
semantic_generation_valid = true
decode_success = true
black_frame_rate = 0
quality_drop <= 0.05
peak_vram_max <= 16GB
no critical latency/model-size regression
```

The validated NVFP4 model-size and latency gains are preserved as the parent
reference and are not traded away by changing model weights during this
campaign.

## Evidence record

Every experiment record and trajectory includes:

- state before experiment;
- Controller diagnosis;
- Controller hypothesis;
- selected operator and exact arguments;
- expected effect and risks;
- execution result;
- quality metrics;
- hardware metrics;
- accept/reject reason;
- state after experiment;
- next-decision context;
- `Harness4H3-v1.0`.

The campaign report additionally summarizes the complete decision sequence,
accepted/rejected candidates, final Pareto candidates, target satisfaction,
experiment and failure counts, wall time/GPU-hours when available, human
intervention count, repeated-failure avoidance, strategy changes after
negative evidence, the final recipe, and evidence that experience affected
later Controller decisions.

`human_intervention_count` is initialized to `0` and remains `0` for the
autonomous run. Correctness/security bugfixes and infrastructure recovery
that do not choose an operator, modify arguments, select a candidate, or
change acceptance thresholds are not optimization interventions. Any human
choice of operator, arguments, parent, candidate, or acceptance rule must be
counted as an intervention and ends the claim of a zero-intervention run.

Exhausting the campaign without satisfying the target is a valid research
outcome. The report must analyze whether the Controller diagnosed the memory
bottleneck, avoided repeated failures, changed strategy after negative
evidence, encountered an insufficient operator space, or failed for reasoning,
tool-capability, or target-feasibility reasons.

## Execution modes

The primary run uses the Ollama Controller `qwen3.5:9b-q8_0` and the remote
RTX 5080 ComfyUI backend. A backend or capability failure is retained as an
explicit failed experiment and consumes the bounded failure budget; it is not
reported as a target result. Offline mock execution is reserved for protocol
tests and cannot establish the 16GB target.

## Verification

Before the real run, execute the focused campaign tests and freeze metadata
tests. After the run, validate the JSON evidence and trajectory records, check
that all records carry the frozen Harness version and required fields, and
produce a research-oriented report from the persisted evidence.
