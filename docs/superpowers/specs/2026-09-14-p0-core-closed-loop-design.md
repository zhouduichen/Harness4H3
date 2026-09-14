# P0 Core Closed-Loop Integration Design

## Goal

Make the existing Harness4H3 model/runtime lineage and evaluator primitives
authoritative in the optimization loop, so a search point is an explicit
`ModelCandidate + SystemCandidate` pair and continuation is decided by fixed
Harness rules rather than by Controller-provided acceptance text.

This phase does not implement the real MiniMax-H3 training adapter. The
existing fail-closed adapter contract and trusted remote workers remain the
boundary for the next phase.

## Scope

### Included

- Track both `current_model_id` and `current_system_id` in loop checkpoints.
- Initialize and persist a baseline `S0000` system referencing `M0000`.
- Make model operators produce model children; make runtime operators produce
  system children that continue to reference the same model.
- Evaluate a model/system pair without mutating model state.
- Add declarative `ObjectiveSpec` values to `TargetProfile`, while preserving
  existing hard constraints. Use the objective list for Pareto dominance and a
  separate scalar search score for ranking context.
- Add a fixed `ContinuationPolicy` with `reject`, `exploratory_keep`,
  `pareto_keep`, and `final_accept` outcomes. The Controller may describe an
  intended acceptance condition, but it cannot override evaluator feasibility,
  quality validity, or Pareto membership.
- Introduce canonical `EvaluationRecord` compatibility aliases so the core
  evaluator and archive share one evidence shape. The legacy subprocess
  evaluator remains readable through compatibility properties.
- Extend Controller context with the active system, objective summary, and a
  bounded campaign summary containing operator outcomes and failure counts.
- Add unit/integration coverage for model-only and runtime-only branches,
  hard-gate continuation, target-driven objectives, pair evaluation, and old
  checkpoint compatibility.
- Update active architecture/research/quickstart documentation and change the
  package version to a research-preview `0.4.x` line.

### Excluded

- Loading, forwarding, backward, optimizer, save, or reload implementation for
  a real MiniMax-H3 adapter.
- Changes to remote SSH worker semantics or the ComfyUI benchmark protocol.
- New pruning/distillation algorithms.
- Moving or deleting legacy files while existing replay paths depend on them.

## Architecture

The loop owns a `SearchState` made of two immutable lineage pointers:

```text
SearchState
  model_id  -> ModelCandidate -> checkpoint/training provenance
  system_id -> SystemCandidate -> model_ref + runtime recipe
```

Model-changing operators allocate `Mxxxx`, then the loop creates a paired
`Sxxxx` pointing at the new model and inheriting the parent runtime recipe.
Runtime-only operators allocate `Sxxxx` and keep `model_ref` unchanged. The
model archive never receives a runtime-only child.

The evaluator receives the model state and the system state. It creates a
benchmark view by overlaying system algorithm/runtime state onto the referenced
model state. Measured metrics are produced by the evaluator; operators may
return execution metadata but may not claim quality or hardware improvement.

The Pareto archive stores search-point IDs and the objective definitions used
for the comparison. New loop records use system IDs, while old model-only
entries remain readable and continue to use the legacy default objective set.

## Target and scoring

`TargetProfile` keeps hardware/quality limits as hard constraints. It gains an
`objectives` tuple of `ObjectiveSpec(name, direction, weight)`. Supported
metrics are `quality_score`, `latency_s`, `peak_memory_gb`, `model_size_gb`,
and `energy_j`; `feasibility` is a gate and is not a numeric objective.

If a target omits `objectives`, the existing `priority` tuple is translated to
the default objective set. `priority` remains readable for compatibility but
is no longer the only source of optimization semantics.

Pareto dominance uses all available configured objective values and never
allows an infeasible evaluation to dominate a feasible one. Scalar search
score is exposed separately as a weighted signed sum for Controller ranking;
it never replaces the Pareto archive or hard gates.

## Continuation policy

The fixed policy is evaluated after independent evaluation and Pareto update:

1. Invalid evidence or critical quality regression → `reject`.
2. Feasible evaluation → `final_accept`.
3. Valid non-feasible evaluation on the Pareto front → `pareto_keep`.
4. Valid non-feasible evaluation that improves a configured objective while
   remaining above the quality floor → `exploratory_keep` only when the loop's
   explicit exploration flag is enabled.
5. Otherwise → `reject`.

Only `pareto_keep` and `final_accept` advance the active system by default.
The plan's acceptance mapping is retained as audit input and may tighten the
quality floor, but cannot loosen evaluator hard constraints or promote a
non-Pareto candidate. Any rejected child remains archived with its evidence.

## Compatibility and migration

- `SessionState.from_dict` accepts old checkpoints without `current_system_id`
  and synthesizes a baseline system for the current model.
- `OptimizationResult.current_model_id` remains available; a new
  `current_system_id` is added.
- Existing `ModelCandidate` and `SystemCandidate` JSON files remain readable.
  New system IDs use `Sxxxx`; legacy `Cxxxx` IDs are accepted for replay.
- Existing evaluator callers using `.score`/`.metrics` continue to work via
  compatibility properties on the canonical record.
- The old `harness4h3.harness` workflow runner is not migrated in this phase;
  only the active `controller.OptimizationLoop` path uses pair evaluation.

## Error handling

- Missing model/system references, mismatched model references, duplicate
  child IDs, invalid objective definitions, and invalid continuation inputs
  fail closed with typed `ValueError`/store errors.
- Runtime operator failures do not create model children.
- A model operator failure does not create either child.
- Evaluation failures are recorded as failed experiment records and do not
  change active pointers.
- Checkpoint writes remain atomic and old checkpoint formats are migrated in
  memory only; no existing evidence is overwritten.

## Testing strategy

- Unit tests validate objective parsing/scoring, target defaults, Pareto
  directions, continuation outcomes, canonical evaluation compatibility, and
  `Sxxxx`/legacy `Cxxxx` store behavior.
- Integration tests run the fake closed loop through a model branch followed by
  a runtime branch and assert model lineage is unchanged by the runtime step.
- Resume tests assert old model-only checkpoints synthesize `S0000` and new
  checkpoints resume the exact model/system pair.
- The full existing CPU test suite and compile check remain required.

## Non-goals and claim boundary

This integration increases protocol correctness and auditability. It does not
constitute MiniMax-H3 training evidence, quality improvement evidence, or a
claim that the Harness outperforms an LLM-only baseline.
