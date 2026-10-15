# EvoCompress Search-Space Evolution Design

## Status

Approved for implementation in the isolated worktree
`codex/evo-search-space-evolution`. The implementation is additive and starts
from commit `b0a9710` of `codex/rsi-compression-loop`.

## Goal

Extend the existing Harness4H3 EvoCompress loop from selecting one of three
trusted compression strategies to an evidence-backed, restartable system that
can improve algorithm source code and evolve its executable compression search
space. All new paths are CPU/Fake-first and disabled by default, so existing
Wan2.1-1.3B runs keep their current code, output roots, evaluator, and GPU
allocation behavior.

## Non-goals

- No change to the currently running four-arm Wan experiment.
- No real Wan GPU evaluation, promotion, or claim of compression benefit in
  this development cycle.
- No unrestricted execution of provider-generated Python.
- No dynamic import or direct registry mutation from an Agent.
- No removal or incompatible rewrite of the existing three operators,
  `PromotionPolicy`, Sandbox boundary, checkpoint lineage, or resume protocol.

## Safety and isolation contract

1. Development uses a separate Git worktree and branch. The dirty checkout at
   `/Users/huangjiahao/MinMax-H3/Harness4H3` is never edited.
2. New configuration switches default to `False`/legacy behavior. Existing
   `profile_model`, `EvolutionLoop`, and CLI calls remain behavior-compatible
   unless the caller explicitly enables the new feature.
3. New artifacts are written below the caller's new output root only:
   `profiles/`, `algorithms/`, `search-space/`, `screening/`, and round-local
   JSON files. No code reads or writes the running experiment roots.
4. Agent proposals are data until a trusted validator accepts them. Sandbox
   source validation and process isolation remain mandatory for algorithm
   source. Search-space registration accepts only a built-in trusted operator
   factory result.
5. Every metric carries evidence status (`measured`, `estimated`, or
   `unavailable`) and a method. CPU/Fake evidence is marked
   `offline_simulation` and cannot satisfy real Wan `PromotionPolicy`.

## Existing boundaries retained

The existing architecture already provides the correct core boundaries:

```text
ProfileReport -> AlgorithmAgent/AlgorithmStore -> AlgorithmSandbox
       -> WanCompressionRegistry/WanModelAdapter -> EvaluationFn
       -> PromotionPolicy -> ExperienceStore/SkillHub
```

The extension adds four focused components around those boundaries rather than
creating a second harness:

```text
CalibrationCorpus -> ProfileReport v2
Profile + lineage + feedback -> AlgorithmImprover -> AlgorithmValidator
SearchSpaceProposal -> SearchSpaceValidator -> trusted Registry view
Candidate -> Static -> Diagnostic -> Full evaluator -> PromotionPolicy
```

## 1. Multi-granularity Profiling

### Calibration interface

Add `harness4h3/profiler/calibration.py` with immutable, serializable types:

- `CalibrationSampleSpec(prompt, seed, latent_shape, timesteps, sample_id)`;
- `CalibrationSample(spec, payload, provenance)`;
- `CalibrationProvider` protocol with `build(spec)` and `batch(specs)`;
- `FakeWanCalibrationProvider`, deterministic CPU-only, seeded from the spec.

The spec records the fixed Prompt, Seed, latent shape and timestep values. The
sample payload is adapter-specific and may be a tensor or mapping. The real Wan
adapter can later provide a valid transformer probe without changing the
profiler API. The CPU implementation is the acceptance path for this cycle.

`profile_model(..., calibration=None)` keeps the existing default sample path.
When a `CalibrationCorpus` is supplied, all block, channel and timestep probes
use the same deterministic sample set and record a corpus digest.

### Evidence and hierarchy

Extend `ProfileReport` with additive fields:

- `schema_version=2` for newly written reports, while `from_dict` reads v1;
- `calibration`: corpus specs, digest, and sample provenance;
- `relationships`: block-to-layer, block-to-channel and block-to-timestep
  references;
- `evidence`: report-level status and unavailable reasons.

Existing fields remain present. Each new or enriched measurement includes:

```json
{
  "value": 0.12,
  "status": "measured|estimated|unavailable",
  "method": "activation|output_perturbation|weight_proxy|hook_latency|flop_formula",
  "offline_simulation": true,
  "reason": null
}
```

`available` and `approximation` are retained as compatibility aliases, but a
missing value is never serialized as a measured zero.

### Full, chunked channel analysis

Add an extensible `ChannelImportanceMethod` protocol in
`profiler/sensitivity.py` (or a small adjacent module if needed):

- `WeightContributionMethod`: existing paired-weight approximation;
- `ActivationImportanceMethod`: accumulates activation magnitude per channel
  through a forward hook when the adapter exposes the required FFN boundary;
- `OutputPerturbationMethod`: masks channel chunks and measures output delta;
- `UnavailableImportanceMethod`: explicit fail-closed record.

`measure_channel_sensitivity` gains `method`, `chunk_size`, `sample_set` and
`full=True` options. Advanced mode evaluates every channel in chunks, not just
the first 32. Unsupported methods produce unavailable records with a reason.
All methods preserve Wan's all-block FFN alignment constraint; they only provide
evidence and never mutate the trusted model.

Layer records continue to contain parameter count, FLOP estimate, latency and
memory fields. New memory telemetry is measured only where the runtime exposes
it; CPU fake reports it as unavailable. Block, channel and timestep records are
joined by stable block IDs in `relationships` so an Agent can distinguish cost
from sensitivity.

## 2. Recursive algorithm improvement

Add `harness4h3/evolution/improvement.py` containing:

- `AlgorithmImprovementRequest`: parent algorithm ID/source hash, parent source,
  bounded profile, history, failure reasons, applicable conditions and search
  space version;
- `AlgorithmImprovementResult`: candidate, changed fields, rationale, evidence
  IDs and status;
- `AlgorithmValidator`: executes parent and child source in the same Sandbox,
  on the same profile/model state and diagnostic budget, then compares the
  supplied evaluation records;
- `AlgorithmValidationRecord`: immutable evidence with `valid`, `evidence_level`,
  parent/child source hashes, controlled-condition digest and failure reason.

Extend `AlgorithmAgent` with `improve(...)`. The deterministic provider path
must read the parent source and produce a source diff with a concrete strategy
change (for example, change from repeated block selection to a profile-guided
FFN threshold rule after an efficiency plateau). A structured provider gets the
same parent source, source hash, historical failures and applicability context
in its bounded prompt. It cannot change the trusted parent ID.

Extend `AlgorithmCandidate` and `AlgorithmStore` with immutable lineage fields:

- `parent_algorithm_id` and `parent_source_sha256`;
- `change_summary`, `change_rationale`, `evidence_ids`;
- `validation_status` and `validation_record_path`.

The source itself, source hash and parent identity remain immutable. Failed or
duplicate versions are retained as negative evidence. Validation never changes
the model candidate status: model lineage (`M...`) and algorithm lineage
(`A...`) are written separately and linked by the round decision.

`SkillHub` retrieves algorithm source summaries plus the bounded parent lineage
and failure evidence. The complete source remains in `AlgorithmStore`; prompts
receive only the bounded source required for the requested improvement.

## 3. Evolvable compression search space

Add `harness4h3/operators/wan/search_space.py` with:

- `OperatorSpec(name, version, argument_schema, granularity, constraints,
  limitations, trusted_factory_key)`;
- `SearchSpaceVersion(version_id, parent_version_id, operators, constraints,
  evidence_ids, status)`;
- `SearchSpaceProposal(proposal_id, parent_version_id, kind, spec, rationale,
  evidence_ids, status, rejection_reason)`;
- `SearchSpaceStore` using atomic JSON/JSONL persistence;
- `SearchSpaceManager` implementing propose, validate, reject and register.

The initial version describes the existing three operators and the current
Wan alignment rule. A proposal is not executable and does not change the
Sandbox allowlist. Validation performs, in order:

1. schema/interface and argument-bound checks;
2. trusted-factory lookup (no arbitrary source import);
3. CPU Fake execution and forward/finite checks;
4. save/reload and checkpoint hash verification;
5. independent diagnostic evaluation with `offline_simulation=true`.

Only a successful validation creates a new `SearchSpaceVersion` and a new
Registry view. The original Registry remains unchanged for callers that do not
request the new version.

The CPU demonstration registers a built-in
`prune_ffn_by_weight_threshold` operator only after validation. It derives
retained indices from paired FFN weight evidence and delegates the actual
mutation to the existing trusted channel-pruning operator. This is a search
space demonstration, not a claim of a new Wan compression technique. On a real
Wan adapter it must expand to all current positional blocks before execution;
otherwise validation rejects it.

Validated operator records are written to SkillHub as measured/partial design
genes with the SearchSpaceVersion, proposal ID, source and experiment evidence.
Later AlgorithmAgent requests receive the active search-space description and
can propose further modifications, while unvalidated operators remain
retrievable only as rejected proposals.

## 4. Low-cost candidate screening

Add `harness4h3/evolution/screening.py` with:

- `ScreeningStage`: `static`, `diagnostic`, `full`;
- `ScreeningResult`: stage, status, cost, evidence level, reasons and metrics;
- `CandidateScreeningPipeline`: restartable stage scheduler;
- `ScreeningContext`: parent artifact, profile, plan, registry/search-space
  version and optional CPU diagnostic evaluator.

Stage behavior:

1. Static: source policy, duplicate source, plan schema, operator availability,
   parameter ranges, model structure and composition constraints. No model
   mutation and no GPU allocation.
2. Diagnostic: bounded Fake/CPU forward, structural execution, finite parameter
   checks, save/reload hash checks and limited simulated metrics. Cost and
   evidence are recorded explicitly.
3. Full: invoke the caller's existing independent Wan evaluator only after the
   first two stages pass. The result is stored separately and passed to the
   unchanged `PromotionPolicy`.

The pipeline writes `screening.json` atomically and resumes completed stages.
Screening acceptance is never promotion acceptance; a Fake result cannot set
`valid` for a real parent or bypass missing real metrics.

`EvolutionLoop` receives optional `screening_pipeline` and feature switches.
When disabled, its current sequence is unchanged. When enabled, it records
screening paths in the round and skips full evaluation for statically or
diagnostically rejected candidates while still recording their negative
experience.

## End-to-end round

With all new flags enabled, a round follows:

```text
calibration/profile
  -> active SearchSpaceVersion
  -> retrieve parent algorithm/source/history/limitations
  -> improve or propose algorithm
  -> AlgorithmStore immutable child
  -> Sandbox validation
  -> static screening
  -> CPU/Fake diagnostic screening
  -> trusted operator execution
  -> optional algorithm parent/child comparison
  -> full independent evaluator
  -> unchanged PromotionPolicy
  -> model + algorithm + search-space evidence in Memory/SkillHub
  -> next round retrieves all three lineages
```

Generation failures, duplicate sources, invalid operators, incompatible model
layouts, and interrupted rounds become durable failure records. A resumed
round reuses `profile.json`, `algorithm.json`, `search-space.json`,
`screening.json`, Sandbox output and compression artifacts after validating
their hashes; it never silently restarts a GPU evaluation.

## Configuration and compatibility

Additive compression-config fields are all disabled by default:

```yaml
advanced_profiling: false
algorithm_improvement: false
search_space_evolution: false
candidate_screening: false
```

When enabled, each feature still defaults to `backend: fake` in tests and to
CPU diagnostic work. Existing real Wan configs do not inherit these flags.
The four running experiment configs and their output roots are not edited.

## Test plan

Use the existing `.venv` and CPU Fake/Mock components. Add focused unit tests
for calibration determinism, all-channel chunking, evidence status, source
lineage and hash immutability, improvement diffs, algorithm comparison,
search-space proposal/validation/rejection/registration, threshold operator
alignment, and each screening stage. Add an integration test covering:

```text
profile -> history failure -> algorithm improvement -> Sandbox/Validator
-> new search-space proposal -> Fake validation -> candidate screening
-> simulated evaluation -> evidence/SkillHub -> next-round retrieval
```

Also cover provider failure, duplicate algorithm, invalid operator, incompatible
model, interruption/resume, and bounded long Memory. Existing tests must pass
without changing their assertions. Tests must assert that offline results never
reach real promotion.

Acceptance evidence is:

- existing target tests and full CPU test suite pass;
- new integration path passes without CUDA or real Wan assets;
- static compilation passes;
- `git diff` shows no changes in the original dirty checkout or experiment
  output roots;
- documentation clearly separates CPU-tested behavior from future real Wan
  validation.
