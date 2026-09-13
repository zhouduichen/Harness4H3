# Architecture

Harness4H3 separates the research policy from the execution and measurement
mechanisms. The object being optimized is an H3 model candidate or a runtime
configuration; the Harness itself is not an experimental variable.

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
| Trajectory | Records plans, execution, evidence, failures, and cost | Append-only |

## Data flow

```text
TargetProfile + DeviceProfile + current Model/System state
                              ↓
                 capability preflight (read-only)
                              ↓
 ControllerContext = state + budget + operators + prior evidence + Pareto
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
             benchmark → independent EvaluationResult
                              ↓
       keep/drop decision + archive + trajectory + next context
```

The device profile answers whether a required capability is present. The
TargetProfile defines success constraints. Keeping these separate prevents a
machine description from silently changing the scientific objective.

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

## Frozen core and research surface

`Harness4H3-v1.0` freezes Controller protocol, state and plan schemas,
validation order, evaluator authority, archive/trajectory semantics,
acceptance policy, and TargetProfile meaning. Only correctness or security
fixes may change these components.

Research extensions belong in:

- `research/experiments/` for reproducible protocols;
- `harness4h3/operators/` or an external worker for a concrete bounded
  intervention;
- `configs/devices/` for machine facts and capabilities;
- `configs/targets/` for predeclared objectives;
- `research/evidence/` for measured results, including failures; and
- `research/history/` for completed design decisions.

## Capability boundary

The repository has a real H3 inference and benchmark path and an external
model-worker adapter. The adapter is not a trainer. Until a memory-feasible H3
implementation performs real forward, backward, non-zero-gradient optimizer
updates, child save/reload, and parent/child verification, model-changing A1
remains blocked.
