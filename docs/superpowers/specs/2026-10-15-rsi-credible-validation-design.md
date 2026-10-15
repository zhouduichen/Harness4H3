# EvoCompress RSI Credible Validation Design

## Scope

This is an incremental correction to the existing EvoCompress loop. It keeps
`ModelCandidate`, `AlgorithmCandidate`, `AlgorithmStore`, `AlgorithmValidator`,
`SearchSpaceManager`, `SkillHub`, `AlgorithmSandbox`, `CandidateScreeningPipeline`,
`Evaluator`, and `PromotionPolicy`. Legacy configurations remain unchanged and
all new behavior is opt-in through the existing evolution flags.

## 1. Model-level algorithm evaluation

`AlgorithmValidator` will continue to own parent/child comparison, but its
evaluator callback will execute the generated plan against the same immutable
parent checkpoint for every fixed validation scenario. The evaluator will:

1. use a deterministic adapter probe input and seed;
2. execute the trusted compression registry and reload the child checkpoint;
3. measure parameter count, structural/estimated FLOPs, output MSE and cosine
   similarity against the uncompressed parent probe output;
4. record checkpoint hash, structural change, validity, wall time, invocation
   count, and budget consumption;
5. mark GPU latency, peak memory, video quality, and other unavailable metrics
   as `available: false` rather than estimating them.

Fake evidence is labeled `cpu_fake_algorithm_measurement`; it is never passed
to model promotion as real Wan evidence. Equal plans are evaluated by structure
and therefore receive equal metrics independent of operator naming.

## 2. Feedback-driven search-space evolution

Initialization only loads the current validated search-space version. It does
not propose or register an extension. After each terminal round,
`EvolutionLoop` derives a bounded bottleneck report from persisted round
evidence: consecutive Pareto stagnation, repeated failures, duplicate plans,
or an expressiveness counterexample. Only after a configurable evidence
threshold is met does `AlgorithmAgent.propose_search_space_change` receive the
report. The report, proposal, validation, rejection, and rollback remain
separate artifacts.

The deterministic provider-free path may produce a proposal only when the
threshold is met, and its proposal is explicitly marked as a deterministic
test fallback rather than Agent discovery. Provider-backed proposals must cite
observed round/algorithm evidence IDs.

## 3. Three-level change taxonomy

Every algorithm/search-space evidence record carries one of:

- `parameter_search`: same registered operators, different arguments or ratios;
- `algorithm_rule_evolution`: source-level scoring/ranking/budget logic changed
  while the reachable structural transform remains expressible by the active
  operator closure;
- `compression_space_extension`: the validated transform is outside the
  baseline operator closure and includes an explicit structural counterexample.

`prune_ffn_by_weight_threshold` and `GeneratedChannelRuleOperator` default to
`algorithm_rule_evolution` unless an expression checker proves a structural
counterexample. Merely registering a new name is never sufficient. Unsupported
new model structures remain rejected proposals with adapter limitations.

## 4. Recursive parent selection and evidence levels

The loop distinguishes latest generated, validated effective, historical best,
and pending branch algorithms. A rejected or unmeasured candidate cannot become
the effective parent solely because it is latest. New improvement requests use
the best compatible validated parent, while pending branches remain searchable
experiments. Source changes, measured improvement, and model promotion are
stored as independent facts.

Evidence levels are monotonic and explicit: `sandbox_only`,
`cpu_fake_structure`, `cpu_fake_quality_proxy`, `real_wan_structure`,
`real_wan_video`, and `real_wan_hardware`. Model promotion remains exclusively
controlled by the existing evaluator and `PromotionPolicy`.

## Verification

Focused tests will cover equal-plan invariance, different-plan quality/size
trade-offs, fixed inputs and seeds, measurement cost, feedback thresholds,
taxonomy classification, rejected-state preservation, and parent fallback.
The CPU Fake CLI will run multiple rounds with the new evaluator and record
auditable JSON evidence. No existing real Wan run will be changed or reused as
CPU evidence.
