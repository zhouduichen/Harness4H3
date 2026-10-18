# Evidence-Guided Heterogeneous WAN RSI Search

## Goal

Use measured Wan2.1-T2V-1.3B execution evidence and real candidate outcomes to guide Harness4H3's bounded search over existing structural, token, timestep, sparse-attention, and cross-step-cache methods. The search must test whether an offspring improves on its matched parent without misreporting runtime-only methods as parameter reduction.

## Current Evidence and Gaps

- The real 832×480×81 profile measured 61.866 s of inclusive Self-Attention time and 32.814 s of inclusive FFN time within 89.886 s of Transformer time. These module timings are nested and overlap; they are useful for bottleneck ranking but cannot be summed to estimate end-to-end savings.
- On the same one-prompt/one-seed recipe, Dense measured 105.996 s median, SVOO 65.129 s, MagCache 62.947 s, and SVOO+MagCache 39.895 s. The combined result's aggregate score was 0.00178 below Dense. This is promising single-sample evidence, not a quality-equivalence claim.
- Harness's `sparse_runtime` is its own Top-K K/V-selection policy. It is distinct from SparseVideo's SVOO implementation.
- SVOO and MagCache currently enter `WanVideoEvaluator` through fixed evaluator configuration. Candidate-specific external runtime policies are not yet represented in `EvolutionLoop` evaluation.
- The bounded search generator ranks candidates within operator families, then truncates and orders families round-robin. Measured candidate outcomes do not currently determine how much of a round's candidate budget each family receives.
- `ModelVariant` already separates `ordered_operations`, `runtime_policy`, and structural checkpoint identity. This is the intended carrier for heterogeneous runtime methods; runtime-only policies must keep parameter-reduction metrics at zero.

## Design

### 1. Candidate and runtime-policy representation

Keep the existing operator and `compose` interfaces. Register thin, versioned runtime-policy adapters for SparseVideo SVOO and Diffusers MagCache. The adapters store validated method/configuration metadata on the candidate variant; SparseVideo and Diffusers remain responsible for executing their published implementations. Do not alias either method to `sparse_runtime` or copy their kernels into Harness4H3.

Preserve method order in `ModelVariant.ordered_operations`. The evaluator resolves a candidate's runtime policy from that ordered list and installs attention and cache methods through the existing runtime hook paths. The same `WanVideoEvaluator` invocation must receive the candidate policy, so the policy used for measurement is part of the candidate identity and result provenance. A runtime-only candidate keeps the immutable parent weights and records no parameter reduction.

### 2. Bounded heterogeneous candidate generation

Retain single-method candidates as controls. Generate at most two-method composites from registered, compatible operators. Each proposed pair must satisfy both conditions:

1. The measured profile or prior same-parent evaluations support both methods' target bottlenecks.
2. The pair is not an order-insensitive duplicate of an already measured candidate; when order changes execution semantics, retain and record the order.

The first supported external-method interaction is SVOO+MagCache, because both have standalone and combined real Wan measurements. Subsequent pairs may combine methods already present in the active catalog, such as timestep-conditioned token selection with a compatible attention/cache method. Do not generate unconstrained Cartesian products.

### 3. Evidence-driven budget and score

Use the existing `max_candidates`, `algorithm_probe_samples`, search-space versioning, and memory records. For the initial round, rank target families by measured baseline component evidence and recorded standalone method results. Since module categories are inclusive, use component-specific values only for ordering; estimate candidate benefit from paired end-to-end latency deltas, never by summing module categories.

After each same-parent evaluation, update family utility from the paired quality delta, measured end-to-end latency delta, peak allocated memory delta, parameter/checkpoint-size changes, validity, and evaluation cost. A candidate that fails the configured quality gate cannot be scored as a successful compression even when it is faster. Preserve an exploration allocation for compatible families with little evidence so one early result does not monopolize the search budget. Record the budget allocation, source evidence, utility values, and omitted-candidate reasons in search diagnostics.

### 4. Evaluation and promotion evidence

Every candidate comparison uses the same immutable parent checkpoint, validation case/seed set, evaluator version, device, dtype, and generation recipe. The evaluator records the applied method names/configuration/order, method counters when available, latency samples and median, peak allocated/reserved memory, quality scores, visual-gate result, parameter count, and checkpoint size. SVOO dispatch counters and MagCache skipped-step counters are recorded only when the upstream hook exposes them; absent counters remain unavailable.

The real Wan evaluator is the acceptance boundary for RSI offspring. Structural or tensor probes may filter candidates but do not count as algorithm discoveries. An offspring is a successful discovery only if it is valid, passes the unchanged quality threshold, has same-parent real Wan measurements, and improves at least one declared objective without violating the others' configured limits. Runtime-only methods are reported separately from parameter reduction.

### 5. Ablation compatibility

Keep the six-arm protocol identities and matched data/evaluation budgets. Update the protocol fingerprint when the operator catalog or candidate behavior changes. Fixed and non-evolving arms retain their configured search behavior; the expanded frozen arm exposes only its declared frozen additions; online search-space evolution alone may propose and validate new capabilities. Include failed candidates and their GPU/evaluator cost in the arm totals.

## Implementation Boundaries

- Reuse `AlgorithmAgent`, `SearchSpaceManager`, `ModelVariant`, `CompositeCompressionOperator`, `WanVideoEvaluator`, `ExperienceStore`, and `EvolutionLoop`.
- Reuse official SparseVideo SVOO and Diffusers MagCache implementations. Do not implement replacement attention/cache kernels.
- Do not modify model weights to encode a runtime-only policy. Store candidate policy in the existing variant sidecar and pass it to the evaluator explicitly.
- Keep the legacy four-arm protocol validator compatible while fingerprinting the six-arm paper protocol's current source and profile assets.
- Remote files and experiment outputs may only be read or written under the server's `ai4mbse` directory. The absolute path must be confirmed before any remote operation.

## Acceptance Evidence

1. Local candidate records distinguish SVOO, MagCache, and Harness Top-K `sparse_runtime`; composed variants preserve policy order and correct structural/runtime classification.
2. Candidate-specific runtime policy is visible in evaluation provenance and can be reloaded for a repeated evaluation without silently falling back to Dense.
3. Search diagnostics show measured bottleneck evidence, source of each score, candidate budget per family, and the reason each supported method or pair was selected or omitted.
4. On the allowed server directory, run a same-parent Wan comparison for Dense, SVOO, MagCache, and SVOO+MagCache, then compare the RSI-selected offspring against its parent using the same prompts, seeds, quality gate, latency repeats, and hardware.
5. The six-arm report includes quality, real GPU latency, peak memory, parameter/checkpoint size, search/evaluation cost, valid offspring count, and algorithm-discovery evidence. No causal claim is made from one prompt/seed; paper results use the configured matched validation set and budget.

## Explicit Limitations

BWCache has no supported Wan2.1 path in the official repository checked for this campaign, so this design does not claim BWCache reproduction. The available combined SVOO+MagCache result is a console-derived single-seed summary because its raw JSON is not in the local worktree. Formal RSI rounds and the six-arm study remain unrun until they complete inside the allowed server directory.
