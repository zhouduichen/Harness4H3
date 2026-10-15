# TICA-Prune: Time- and Interaction-Calibrated Wan Compression Design

## Status and scope

TICA-Prune is an explicitly human-designed algorithm baseline for the
Harness4H3 EvoCompress study. It is not evidence that AlgorithmAgent
discovered the method. Later Agent revisions will be represented as child
algorithm versions and will be accepted only after a controlled parent/child
comparison.

This change is additive and is developed on `codex/tica-prune`, based on
`codex/rsi-core-verification` at `cdd5a8b`. It does not change the four active
Wan experiments, their checkpoints, configurations, output roots, or GPU
processes. CPU Fake is the only execution target in this phase.

## Research contract

The primary object is a training-free structural compression algorithm for
Wan2.1-1.3B that reuses the existing block and global FFN channel operators.
The method must distinguish three claims:

1. A candidate is structurally executable and reloadable.
2. TICA's risk and search rules improve the quality/size Pareto trade-off over
   the human baselines under the same candidate budget.
3. A future Runtime Compression Operator can reduce attention execution cost.

Claim 3 is not a weight-compression claim and is not enabled by this change.
Estimated FLOPs are structural estimates only; GPU latency, peak memory,
video quality, and speedup remain unavailable until the independent Wan
evaluator measures them.

## Calibration and evidence

The canonical calibration object is a versioned manifest containing fixed
sample IDs, prompt/seed metadata, cached latent tensors, cached encoder hidden
states, and the timestep schedule. A real Wan provider may create this
manifest once using the full pipeline; all subsequent structural probes reuse
the cached transformer inputs and never silently regenerate them. A Fake
provider creates the same mapping deterministically on CPU and marks it
`offline_simulation=true`.

The adapter protocol gains a narrow calibration boundary:

```python
adapter.calibration_samples(model, manifest) -> tuple[CalibrationInput, ...]
adapter.forward_denoising(model, calibration_input, timestep) -> Any
```

The existing `profile_sample`/`forward_profile` path remains intact for legacy
tests and the AlgorithmModelEvaluator. TICA uses the new boundary when it is
available and otherwise fails closed with an explicit unavailable evidence
level; it does not treat random probes as real Wan calibration evidence.

For every candidate, TICA persists:

- model, algorithm, search-space, calibration-manifest, source, and seed
  digests;
- candidate recipe and actual structural change;
- per-timestep denoising output error and weighted risk;
- parameter count/reduction, structural FLOPs estimate, and real elapsed
  measurement cost;
- fast-screen versus full-ablation status and budget counters;
- parent/child interaction deltas and Pareto comparison.

## TICA-Prune algorithm

The human baseline recipe has four independently addressable stages:

1. **Time weighting.** For each calibration sample and timestep, compute the
   relative denoising-output error caused by a local block removal. The default
   weights are normalized inverse-noise weights over the manifest schedule,
   with the exact formula and values recorded in the recipe.
2. **Block risk.** Use local probes to rank block removals. Re-evaluate only
   the top `max_full_block_evals` candidates by executing the complete model
   path for every selected timestep. The risk is the weighted relative error,
   not a fixed score or operator label.
3. **FFN contribution.** Estimate each channel's activation contribution from
   cached calibration activations and paired FFN weights. Because the current
   real adapter requires a shared FFN width/index set across all remaining
   blocks, TICA emits one globally compatible retained-index set. It rejects
   per-layer unequal-width recipes unless a future adapter explicitly declares
   that capability.
4. **Interaction-aware search.** Measure a bounded set of block pairs and
   block-plus-FFN recipes by actually executing the composed structural
   mutation. The interaction penalty is the measured joint error minus the
   sum of the independently measured errors. Missing measurements are
   `unavailable`, never zero.

The candidate search has a shared interface for all baselines:

```python
search(profile, budget) -> SearchResult
```

The initial baselines are:

- `independent_block`: block risk without time reweighting;
- `tica_block`: time-weighted block risk;
- `interaction_block`: time-weighted block risk plus measured pair penalty;
- `tica_block_ffn`: interaction-aware block plus global FFN channel recipe.

The default search strategy is bounded Greedy followed by Beam Search. Each
candidate consumes an explicit fast-screen or full-ablation budget. Search
returns a Pareto set over quality proxy, parameter reduction, FLOPs estimate,
and measurement cost. Adding more candidates cannot by itself count as an
improvement; a child must beat its parent on the same budget and fixed
calibration/test conditions.

## Agent recursion

The TICA baseline source is stored as an `AlgorithmCandidate` with
`origin=human_baseline`, an explicit hypothesis, and a source hash. Its source
produces a `tica_prune` recipe from the supplied ProfileReport; it does not
hard-code a block ID or pretend to be autonomous discovery.

AlgorithmAgent may change only through a child source revision. The source can
change time-weight formulas, risk aggregation, activation/weight mixing,
interaction penalty, risk limits, beam width, and candidate budget. A change
that only changes block IDs or retention ratios is classified as parameter
search. Source changes that produce a different recipe are algorithm-rule
evolution. The existing AlgorithmValidator runs parent and child source under
the same manifest, model checkpoint, profile, and search budget, and the
TICA runner executes both recipes before validation status can become
`validated`.

## Runtime sparse-attention boundary

Add a disabled `RuntimeCompressionOperator` protocol and a Wan adapter
capability report. The initial concrete design is an untrusted proposal for a
SVOO-like operator with:

```python
prepare_runtime(model, runtime_config) -> RuntimeCompressionHandle
run(model, calibration_input, timestep, handle) -> Any
measure_runtime(handle) -> RuntimeCost
```

The proposal records that it changes execution routing, not checkpoint
structure. No CUDA kernel, Diffusers attention monkey-patch, or real Wan
runtime modification is added in this phase. CPU Fake tests cover capability
negotiation, disabled-by-default behavior, and refusal when the adapter lacks
runtime dispatch or hardware measurement.

## Failure and safety rules

- A missing calibration tensor, activation hook, or real hardware metric is
  explicit and blocks claims that require it.
- Structural mutations run through the existing trusted registry and adapter;
  AlgorithmSandbox only validates the recipe-producing source.
- A search result cannot bypass `PromotionPolicy`; algorithm validation and
  model promotion remain separate records.
- A malformed or unsafe Agent child remains pending/rejected with its source
  diff and failure evidence.
- CPU Fake output error is a quality proxy only and cannot be stored as real
  Wan video quality.

## Verification plan

CPU Fake tests will prove deterministic calibration, multiple timestep risk,
profile/formula sensitivity, global FFN alignment, measured interaction
penalties, bounded Greedy/Beam budgets, Pareto output, four baseline labels,
human-origin metadata, and a reproducible parent/child source revision.
Integration tests will run a complete TICA round without changing the legacy
four-arm experiment roots. Real Wan tests remain opt-in and initially verify
only cached-input denoising outputs, structural reload, and unavailable
hardware metrics before any video generation is attempted.
