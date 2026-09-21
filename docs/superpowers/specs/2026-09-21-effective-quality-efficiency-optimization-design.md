# Effective Student Quality-Efficiency Optimization Design

**Date:** 2026-09-21  
**Status:** Approved direction; implementation pending spec review  
**Scope:** The remote H3-to-Student campaign only

## Objective

Turn the Student campaign from a validity loop into an effective quality-efficiency optimization loop.

The goal is not for a 1B–2B Student to exceed the full H3 teacher in absolute visual quality. The goal is to retain an acceptable fraction of H3 quality while improving edge-relevant cost metrics. A candidate is useful only when it is valid, satisfies the H3-relative quality floor, and improves the current quality-efficiency frontier.

The initial hard quality floor is:

```text
Q_student / Q_H3 >= 0.90
```

All training, evaluation, metric aggregation, checkpoint retention, and LLM feedback remain server-side. Codex only deploys code and starts the detached campaign.

## Current Problem

The existing campaign treats a decodable video and complete continuous metric evidence as sufficient for promotion. That is too weak for optimization:

- the current quality score is explicitly a structural brightness/black-frame proxy, not semantic or temporal video quality;
- the reward uses raw latency in milliseconds, so a roughly 10-second generation dominates a quality score near one;
- the campaign stops after a configured number of valid rounds rather than after a measured Pareto improvement;
- the controller receives prior metrics but has no formal incumbent/frontier decision to optimize against;
- the evaluator samples the first available H3 cache item rather than a fixed held-out evaluation manifest;
- a later Student with the same parameter count and worse quality can still be accepted.

The four-round remote run demonstrated this failure mode: all videos decoded, but the quality proxy moved from approximately `0.852` to `0.837` without a meaningful hardware improvement. That run is evidence of execution, not effective optimization.

## Design Alternatives

### A. Constrained Pareto optimization — recommended

Calibrate a fixed H3 teacher baseline, require every promotable Student to stay above 90% relative quality, normalize efficiency improvements against the current incumbent, and maintain a Pareto frontier. Stop only after a configurable patience window with no frontier improvement.

This directly matches the edge-model objective and reuses the existing Metric Verifier Bank, campaign memory, compiler, remote worker, and retention boundaries.

### B. Absolute quality maximization

Use a semantic quality score as the main objective and require each round to improve it. This is easy to explain but selects larger or slower models and does not represent the 1B–2B edge target.

### C. Full VBench-only gate

Require VBench for every remote evaluation and reject campaigns when VBench is unavailable. This gives a strong quality signal but couples the core campaign to a heavyweight dependency and makes hardware optimization unavailable during evaluator bring-up.

The implementation uses A. VBench can be a quality backend when installed, while the first fallback is a server-local image-text CLIP model applied to sampled frames plus cosine temporal consistency over adjacent-frame embeddings. If neither VBench nor the configured CLIP checkpoint is available, the campaign fails closed. The structural proxy remains diagnostic only.

## System Architecture

```text
fixed evaluation manifest
        |
        +--> H3 teacher calibration --> teacher-baseline.json
        |
        +--> Student round generation --> quality/hardware metric bank
                                             |
                         teacher ratio + incumbent delta + validity
                                             |
                                  normalized reward + Pareto gate
                                             |
                      keep incumbent/frontier or record rejection
                                             |
                                  bounded LLM next-round context
```

### 1. Fixed evaluation manifest

The campaign receives a server-local manifest containing a small deterministic set of held-out evaluation cases. Each case has a stable ID, prompt or conditioning-cache item, and seed. The same cases are used for H3 calibration and every Student evaluation.

The manifest is immutable for one campaign. It is copied into the campaign root and its digest is recorded in every evaluation record. A candidate cannot change prompts, seeds, evaluator code, or metric definitions through the LLM proposal.

The first production manifest uses four cases and two seeds per case. A smoke configuration may use one case and one seed, but a smoke result cannot be promoted as an optimization result.

### 2. H3 teacher calibration

Before the first Student proposal, a trusted server-side baseline worker evaluates H3 on the fixed manifest and writes:

```text
teacher-baseline.json
  manifest_digest
  quality_metrics
  aggregate_quality
  hardware_metrics
  metric_backend_versions
```

Teacher quality is the reference for the 0.90 floor. Teacher hardware metrics are reported for context but do not make the Student compete against the full H3 latency or memory footprint as if they were the same runtime.

Calibration failure is terminal for the campaign. The campaign must not substitute the structural proxy or an invented teacher score.

### 3. Student evaluation

The trusted evaluator generates one video per manifest case and seed, then aggregates metrics across the complete set. Required quality signals are:

- semantic alignment, using VBench when configured and available, otherwise a server-local image-text CLIP model applied to the fixed frames;
- temporal consistency and frame-to-frame stability;
- video decode validity and finite-pixel checks;
- motion/degeneracy checks that reject frozen, black, or invalid videos.

Required hardware signals are median and p95 generation latency, peak CUDA memory, checkpoint size, and device identity. Energy is optional and is included only when measured by the server power sampler; it is never inferred from latency or TDP.

If no configured real semantic/temporal backend is available, the evaluator returns `quality_evaluator_unavailable` and the candidate is not promotable. The structural proxy is recorded under `diagnostic_metrics` only.

### 4. Normalized reward

Raw values are not summed directly. The bank computes a teacher-relative quality ratio and bounded efficiency deltas:

```text
q_ratio     = aggregate_student_quality / aggregate_h3_quality
lat_delta   = clip(log(incumbent_latency / student_latency), -1, 1)
memory_delta= clip(log(incumbent_memory / student_memory), -1, 1)
size_delta  = clip(log(incumbent_size / student_size), -1, 1)
energy_delta= clip(log(incumbent_energy / student_energy), -1, 1)
```

The default scalar ranking score is:

```text
R = 0.60 * q_ratio
  + 0.20 * lat_delta
  + 0.10 * memory_delta
  + 0.05 * size_delta
  + 0.05 * energy_delta
```

Missing optional energy removes its term and renormalizes the remaining weights. Missing required metrics produce no reward and no promotion. Each metric observation records its raw value, reference, normalization, direction, source, and backend version so the LLM can distinguish real evidence from unavailable evidence.

The incumbent reference is the currently best accepted Student for the active device and evaluation manifest. The first feasible Student establishes the incumbent but is not counted as an optimization improvement.

### 5. Promotion and Pareto policy

Hard promotion gates, in order:

1. The proposal passes the existing schema, parameter, graph, and memory constraints.
2. Training produces changed, loadable, quantized Student weights.
3. Every manifest case produces a decodable finite video.
4. Real semantic and temporal metrics are present.
5. `q_ratio >= 0.90`.
6. The candidate either improves normalized reward by at least `0.02` or enters the Pareto frontier with a material efficiency gain.

For the frontier check, quality is maximized and latency, memory, size, and energy are minimized. A candidate enters the frontier when it is not dominated by an existing accepted candidate and improves at least one efficiency metric by 5% while no required metric worsens by more than 2%, or when its teacher-relative quality improves by at least 1% at no material hardware regression.

A candidate that fails the quality floor or frontier gate is a measured rejection, not a campaign crash. Its full evidence and reason are appended to experience memory; its large checkpoints are deleted according to the existing retention policy.

### 6. Campaign stopping

The campaign no longer treats `min_rounds_before_success` as proof of optimization. The default effective-optimization policy is:

- `max_rounds: 8`;
- `no_improvement_patience: 3`;
- success requires at least one feasible Student and one post-baseline Pareto improvement;
- stop with `no_pareto_improvement` when patience is exhausted;
- stop with `campaign_budget_exhausted` at the round limit;
- stop with `campaign_success` only when an accepted frontier candidate exists and the configured stopping condition is met.

The LLM receives the incumbent, frontier summary, quality-floor gap, reward delta, metric deltas, and the last rejection reason. It may propose a new legal architecture or training plan, but it cannot edit evaluator thresholds, metric code, prompts, seeds, or trusted worker commands.

### 7. Checkpoint retention

The campaign retains:

- the current incumbent Student;
- one checkpoint for each non-dominated frontier point;
- the teacher baseline metadata and all compact metric evidence;
- the last failed round metadata for diagnosis.

Dominated full-precision and int8 checkpoints are removed only under the campaign root. The manifest, proposal, compiler digest, metric evidence, and failure record remain durable.

## Code Boundaries

The implementation stays inside the existing Student flow:

- `harness4h3/student/metrics.py`: teacher-relative normalization, bounded reward terms, metric backend evidence, and Pareto comparison primitives;
- `harness4h3/student/evaluator.py`: aggregate per-case quality and hardware evidence without changing hard video validity semantics;
- `harness4h3/student/campaign.py`: baseline bootstrap, incumbent/frontier state, promotion decisions, rejection feedback, patience stopping, and durable resume state;
- `harness4h3/student/config.py`: quality backend, evaluation manifest, quality floor, Pareto margins, reward weights, patience, and round budget;
- `harness4h3/student/inference.py`: deterministic evaluation-case conditioning instead of silently selecting the first cache item;
- `tools/student_evaluate_worker.py`: trusted multi-case evaluator and machine-readable failure codes;
- `tools/student_teacher_baseline_worker.py`: trusted H3 calibration entrypoint using the existing remote H3/ComfyUI path;
- `tests/unit/test_student_metrics.py`: normalization, missing evidence, quality floor, reward, and Pareto tests;
- `tests/unit/test_student_campaign.py`: baseline, rejection, frontier promotion, patience, and resume tests;
- `tests/integration/test_student_full_contract.py`: end-to-end fake metric backend and strict campaign contract;
- `docs/metric-verifier-bank.md` and `docs/student-campaign.md`: operator contract and server-only execution instructions.

The campaign remains server-side and detached. No UI, experience graph, broad operator expansion, or always-on Codex monitoring is part of this change.

## Failure Handling

Every failure is classified separately:

- `teacher_baseline_failed`: H3 calibration did not produce valid reference evidence;
- `quality_evaluator_unavailable`: no configured real quality backend is available;
- `quality_floor_failed`: Student quality ratio is below 0.90;
- `pareto_rejected`: valid Student does not improve the frontier;
- `metric_evidence_missing` or `metric_evidence_invalid`: required evidence is absent or malformed;
- existing compile, training, resource, checkpoint, and decode failure codes remain unchanged.

Only infrastructure and worker failures consume the failure budget. Quality-floor and Pareto rejections are preserved as experience and still advance the bounded LLM iteration, subject to the round budget.

## Verification and Acceptance

### Offline tests

- A normalized reward test proves that a 10-second latency value cannot overwhelm the bounded quality term.
- A quality-floor test rejects `q_ratio=0.89` and accepts `q_ratio=0.90` when all other gates pass.
- Pareto tests cover domination, 5% material efficiency gains, 2% regression tolerance, and quality-only improvement.
- Campaign tests prove that a valid-but-dominated candidate is rejected and its failure context reaches the next proposal.
- Stopping tests prove that four merely valid rounds do not terminate an effective campaign when no frontier improvement exists.
- Resume tests preserve the baseline digest, incumbent, frontier, and patience counter.

### Remote acceptance

A real server run may claim effective optimization only when its campaign root contains:

1. one immutable H3 teacher baseline;
2. the fixed evaluation manifest and digest;
3. at least four Student proposal→compile→train→evaluate rounds;
4. at least one candidate that meets the 0.90 H3 quality floor;
5. at least one post-baseline Pareto improvement with real semantic/temporal evidence;
6. a final incumbent/frontier record and retained checkpoint hashes;
7. no acceptance based solely on the structural proxy.

If the run produces only valid videos but no Pareto improvement, the correct result is `no_pareto_improvement`, not success.
