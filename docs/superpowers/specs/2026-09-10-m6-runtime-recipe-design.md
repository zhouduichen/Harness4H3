# M6 Runtime Recipe and System Candidate Design

## Scope

This increment records the rejected `vae_tiling` hypothesis and extends M6 from
one runtime intervention to a bounded, Controller-selected runtime recipe. It
does not change model weights, TargetProfile thresholds, evaluator gates,
Harness Evolution, or Controller training.

The recipe is an experiment campaign executed under the frozen
`Harness4H3-v1.0` environment. The campaign may select existing runtime tools
and retain their evidence, but it does not make the Harness itself a research
variable.

## Negative Design Gene

The real RTX 5080 Laptop run is stored as a `DesignGene` with status
`rejected`. It records the NVFP4/M5.5 state, `vae_tiling` intervention,
measured quality and latency preservation, peak-VRAM maxima above 16GB, and
the complete evidence paths. The gene is read-only context for later plans and
is not treated as a successful prior.

## Candidate model

`ModelCandidate` remains the immutable checkpoint/archive object with `M…`
identity. A new immutable `SystemCandidate` composes a model reference with
algorithm/runtime state, evaluation, parent system identity, and experiment
provenance. Runtime-only changes therefore create a `C…` lineage entry without
copying a checkpoint or pretending that weights changed. `SystemCandidateStore`
provides atomic JSON persistence, parent-generation checks, and lineage lookup;
existing model stores and APIs remain compatible.

## Runtime operators and recipes

The registry keeps the existing single-intervention operators and adds guarded
component-lifecycle controls:

* `component_lifecycle_optimize` — unload text encoder after encode, defer VAE
  residency until decode, and release cache before decode;
* `vae_decode_offload` — request explicit VAE CPU/offload placement;
* `cache_release` — request an explicit stage cache release.

Each operator validates its booleans/mode and the benchmark adapter applies
only controls exposed by the active workflow/backend. Missing capability raises
`runtime_policy_unsupported`; no operator may silently report success. A recipe
is an ordered list of these policies. Every step retains attribution and
produces a child `SystemCandidate` that references the same model when weights
are unchanged.

## Controller continuation

After a rejected runtime branch, the next Controller context includes the
failure type, gate values, rejected Design Gene, and current system/runtime
state. The bounded loop lets Qwen select one legal next operator or append a
legal recipe step, while preserving the same TargetProfile and budgets. It
stops only when all M6 gates pass, the budget is exhausted, or repeated
unsupported/critical failures reach the configured limit.

## Verification

Unit tests cover candidate identity/lineage, negative gene parsing, operator
capability failures, recipe attribution, and continuation stop conditions.
The existing test suite remains green, and real acceptance still requires
`peak_vram_max_gb <= 16.0` on both dev and held-out splits.
