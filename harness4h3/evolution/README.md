# Harness Evolution (EvoCompress)

This package contains the additive algorithm-evolution path for Wan-shaped
DiT compression. It is separate from the existing Wan student training
campaign: student training owns distillation policy search, while this package
owns structural compression, generated algorithm code, and compression
experience.

The current runtime pieces are:

- `AlgorithmAgent`: deterministic offline generation or a bounded structured
  provider response;
- `AlgorithmSandbox`: AST-checked, no-shell execution of `propose(profile,
  history)`;
- `AlgorithmStore`: immutable source lineage and experiment metadata;
- `SkillHub`: architecture/operator-aware retrieval from append-only memory;
- `EvolutionLoop`: profile → generate → sandbox → compress → evaluate → record.

`evolve-compression` can use `controller.provider: deterministic` for offline
tests or `ollama`, `vllm`, `openai_compatible`, and `openai` for a structured
JSON LLM request. The provider receives profile/history context and can only
return `source`, `hypothesis`, and the already-bound `parent_id`; the sandbox
remains the execution gate.

CPU fake verification:

```bash
.venv/bin/python -m harness4h3 profile-wan \
  --config configs/evolutionary-compression.fake.yaml --json
.venv/bin/python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.fake.yaml --rounds 2 --json
```

All fake results are marked `offline_simulation=true`. Real Wan2.1-T2V-1.3B
execution uses the optional Diffusers adapter and must be run on a host with
the checkpoint, CUDA, and the matching Diffusers dependency set.

## Optional VBench promotion gates

The real Wan evaluator can run the pinned VBench custom-input API on the same
prompt/seed videos it already generates for the parent and candidate. Set
`evaluator.vbench.enabled: true` and provide the isolated `python`, pinned
`source_dir`, `source_lock`, local `cache_dir`, complete `downloads_manifest`,
and one physical `cuda_visible_devices` index. The formal test example shows
the complete YAML shape. Controller startup verifies the pinned Git revision
(when the source is a checkout), hashes for every VBench runtime `.py`, `.json`,
and YAML file, and every cached asset required by the selected dimensions before
a round can begin. The weight manifest must include the dimension checkpoints,
the DINO cache code, and hashes for its Python files. A failed or incomplete
prefetch stops evaluation; scoring uses offline Hugging Face/Transformers
settings and does not fetch missing weights.

The source lock is JSON with a pinned `commit` and a `files` object mapping
source-relative paths to SHA256 strings. It must cover every VBench runtime
`.py`, `.json`, `.yaml`, and `.yml` file under `vbench/`. The downloads manifest
is JSON with an empty `failed` list and a nonempty `files` list; each file record
has `file` (or `path`) and `sha256`. It must cover all required model checkpoints
and local DINO Python files. The scorer runs after the Wan generation pipeline
is released, one video and one dimension per subprocess. Results and worker logs
are saved below each checkpoint's content-addressed evaluation directory so a
resumed round reuses completed scores.

All seven dimensions remain separate. The existing `quality_floor_ratio`
(0.98 by default) adds a parent-relative gate for `subject_consistency`,
`background_consistency`, `motion_smoothness`, `aesthetic_quality`,
`imaging_quality`, and `overall_consistency`. `dynamic_degree` is recorded as
motion presence and does not gate promotion. Parent/candidate deltas are paired
by prompt and seed, with equal prompt weighting and prompt-cluster bootstrap
intervals. VBench values are not collapsed into `quality_score` or a new
composite, and this project policy is not a threshold prescribed by VBench.
`overall_consistency` is VBench's prompt/video embedding cosine score, so a
finite negative score is retained as a real measurement and compared with the
same parent-relative floor.

These are VBench custom-input measurements, not a full standard VBench
leaderboard run or a human preference study. Transformer forward-probe output
similarity remains a separate fidelity diagnostic and does not substitute for
video scoring. VBench is opt-in; disabled configurations retain the current
evaluation path.

## Cost-aware multi-branch search

`evolve-compression --strategy pareto` selects up to three distinct verified
parents from `pareto-front.json` using the HarnessPro normalized squared-
distance rule over quality and deployment inference latency. Search wall time
is accounted separately as the campaign budget. It requires a fixed quality floor and a campaign wall-time
budget. `--strategy single` can use the same ledger by passing the same two
budget options, allowing the comparison tool to check the immutable protocol
fingerprint before summarizing results:

```bash
.venv/bin/python -m harness4h3 evolve-compression --config <single.yaml> \
  --strategy single --rounds 20 --budget-seconds 72000 --baseline-quality 0.98 --json
.venv/bin/python -m harness4h3 evolve-compression --config <pareto.yaml> \
  --strategy pareto --rounds 20 --budget-seconds 72000 --baseline-quality 0.98 --json
.venv/bin/python tools/compare_evolution_campaigns.py \
  --single <single-output-root> --pareto <pareto-output-root> --output comparison.json
```

Both runs must use the same Controller, checkpoint, device, evaluator, and
quality settings. The ledger records parent loading, profiling, algorithm
generation, sandbox validation, compression/training, checkpoint verification,
evaluation, and retry wall time. Candidate representatives use measured
`inference_wall_time_s_mean` from their evaluation records. These are measured
process-stage seconds; they do not establish GPU-active seconds or provider-
billed token cost. Failed and rejected rounds remain in `branch-events.jsonl`,
including the work already spent on them. Offline/fake runs only validate the
mechanics and are excluded from research conclusions. Freeze selected
candidates before running the held-out evaluation split.
