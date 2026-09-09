# EvoGen-RSI Phase I Architecture

Harness4H3 treats the fixed controller as a planner, the Harness as a restricted execution environment, and H3/H3-derived students as the objects being optimized. A TargetProfile is fixed when an optimization session is created. Each controller call receives normalized ModelState, remaining BudgetState, registered operator schemas, recent experiments, failures and the current Pareto front.

The controller's only executable output is ExperimentPlan. Schema, policy, declared budget, operator registration/arguments and registered cost are validated before execution. Fake operators never mutate their parent. A successful modifying operator returns a new ModelState, which is registered as an immutable child ModelCandidate. QualityEvaluator and HardwareEvaluator are composed behind a ConstraintEvaluator; the controller cannot change any of them.

```text
TargetProfile + ModelState + Budget + Experience + Pareto
                         ↓
                Fixed Controller
                         ↓
                 ExperimentPlan
                         ↓
   schema → policy → budget → operator validation
                         ↓
                  registered Operator
                         ↓
             immutable child ModelCandidate
                         ↓
       Quality + Hardware + Constraint evaluation
                         ↓
        ExperimentRecord + Pareto + next state
```

The loop is bounded by iterations, failures, wall time, GPU hours and controller calls. An atomic session checkpoint is written after every experiment. The first delivery provides deterministic fake execution only; the same interfaces are the replacement points for a real provider, H3 inspector, process executor and real evaluators.
