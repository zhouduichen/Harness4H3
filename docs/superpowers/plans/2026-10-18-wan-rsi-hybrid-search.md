# Evidence-Guided Heterogeneous Wan RSI Search Implementation Plan

> **For agentic workers:** Implement inline in the existing Harness4H3 worktree. Keep each milestone independently reviewable and do not start remote work before the allowed server root is confirmed.

**Goal:** Make AlgorithmAgent use measured Wan execution and matched candidate outcomes to allocate search budget and test bounded combinations of existing compression methods, including SparseVideo SVOO and Diffusers MagCache.

**Architecture:** Reuse `ModelVariant` ordered operations and runtime-policy metadata as the candidate identity. Add thin runtime-library adapters that persist SVOO/MagCache policy metadata in a child checkpoint and install the existing upstream implementations during candidate-specific Wan evaluation. Keep candidate generation, budget ranking, and evidence collection in the existing AlgorithmAgent, SearchSpaceManager, WanVideoEvaluator, and EvolutionLoop.

**Tech Stack:** Python, Harness4H3, PyTorch, Diffusers, SparseVideo, existing pytest suite, real Wan2.1-T2V-1.3B evaluation on CUDA.

## Global Constraints

- Follow the approved design in `docs/superpowers/specs/2026-10-18-evidence-guided-wan-rsi-design.md`.
- Keep SVOO, MagCache, and Harness `sparse_runtime` distinct; reuse official SVOO and Diffusers MagCache execution.
- Preserve existing quality limits and same-parent paired comparison rules.
- Never add inclusive module timings to predict total savings; use them for bottleneck ordering only.
- Report runtime-only methods with zero parameter reduction.
- Do not touch any server path outside the user's `ai4mbse` directory. The remote absolute root must be supplied before SSH reads, writes, or runs.
- Keep the existing local branch and uncommitted work intact; commit only files created by a plan task.

---

### Task 1: Persist external runtime policies through the existing Wan variant contract

**Files:**
- Modify: `harness4h3/operators/wan/runtime_policy.py`
- Modify: `harness4h3/operators/wan/compression.py`
- Modify: `harness4h3/adapters/wan/contracts.py`
- Modify: `harness4h3/adapters/wan/diffusers.py`
- Modify: `harness4h3/operators/wan/search_space.py`
- Test: `tests/unit/test_model_variant.py`
- Test: `tests/unit/test_wan_compression.py`

**Interface:** Add validated `SVOOPolicy` and `MagCachePolicy` parsers; register operators named `svoo_attention` and `magcache`; persist their JSON-ready policy under the transformer's Harness runtime-policy config; classify both as runtime operations in `ModelVariant.from_compression`; leave structural checkpoint identity unset when the plan contains only runtime operations.

- [x] Add policy parsers that accept `svoo_attention: {method: "svoo", model_key: "wan21-t2v-1.3b", config: {...}}` and `magcache: {config: {...}}`, reject missing method names and non-mapping config, and preserve unknown official upstream config fields.
- [x] Add one shared runtime-library policy executor in `compression.py` that loads the parent, calls the adapter's runtime-policy persistence method, saves and reloads the child, verifies the persisted policy round-trips, and sets `parameter_reduction_ratio` to `0.0`.
- [x] Add `apply_external_runtime_policy(model, policies)` to the Diffusers Wan adapter; store policy metadata with the transformer config's existing Harness metadata and include it in the adapter's checkpoint digest.
- [x] Register `svoo_attention` and `magcache` in the default search space and compression registry; keep `sparse_runtime` registered under its existing Top-K name and behavior.
- [x] Add both names to `ModelVariant.from_compression`'s runtime operation set so `ordered_operations` preserves order while `runtime_policy` reports the methods separately from structural checkpoint changes.
- [ ] Run the full `pytest tests/unit/test_model_variant.py tests/unit/test_wan_compression.py -q` checkpoint round-trip tests (local environment has no PyTorch; tests were collected and skipped).

### Task 2: Apply candidate-specific runtime methods in the real Wan evaluator

**Files:**
- Modify: `harness4h3/evolution/evaluator.py`
- Modify: `harness4h3/evolution/loop.py`
- Test: `tests/unit/test_wan_video_evaluator.py`
- Test: `tests/integration/test_evolutionary_compression.py`

**Interface:** `WanVideoEvaluator.__call__(child, profile)` reads the persisted Harness policy from the candidate checkpoint, installs the ordered runtime methods through the existing Diffusers pipeline hook path, and records method/config/order plus available runtime counters in the candidate result.

- [x] Read the candidate's persisted runtime policy before constructing the evaluation cache key; the policy is included in the transformer config covered by checkpoint SHA and recipe SHA, so Dense, SVOO, MagCache, and their composition cannot share cached results.
- [x] Install SVOO first through `sparsevideo.apply` and then MagCache through `transformer.enable_cache(MagCacheConfig(...))`, matching the existing successful combination order.
- [x] Keep the evaluator-level baseline Dense unless the configured baseline explicitly names a method; candidate-persisted methods take precedence over same-method global defaults.
- [x] Include normalized candidate runtime policy, installation order, upstream versions, post-generation SVOO `handle.summary()`, and MagCache config in per-case and aggregate evaluation JSON.
- [x] Add focused unit coverage asserting policy-specific installation order and runtime provenance.
- [ ] Run the complete evaluator unit suite and runtime-policy integration case (local environment lacks PyTorch, so only the focused import-free evaluator test ran).

### Task 3: Generate bounded mixed-granularity candidates and allocate measured search budget

**Files:**
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/evolution/loop.py`
- Test: `tests/unit/test_algorithm_program_search.py`
- Test: `tests/unit/test_algorithm_improvement.py`
- Test: `tests/integration/test_standard_search_selection.py`

**Interface:** The generated search program returns at most `state.budget.max_candidates` unique plans, a deterministic rank, explicit per-family allocated counts, and an evidence record per candidate. It retains single-method controls and creates only compatible, at-most-two-method `compose` plans.

- [x] Keep profile component evidence separated by semantic category and attach `module_timing_scope`; use measured categories for target ordering without adding overlapping inclusive values.
- [x] Add standalone SVOO and MagCache plans when registered; add only the SVOO→MagCache two-method composite as the currently supported external interaction. Same-parent duplicates are filtered by plan digest.
- [x] Reject nested, oversized, and unsupported composite proposals; keep the generator's external interaction bounded to the measured SVOO/MagCache pair.
- [x] Replace family-order truncation with deterministic weighted allocation, reserving first probes and using same-parent real-Wan quality-gated latency gain per GPU second for later family priority.
- [x] Persist `allocation_reason`, `evidence_refs`, family sample counts, family utilities, and candidate profile evidence in search diagnostics; mark missing timings unavailable.
- [x] Add unit coverage for ordered external composition, deterministic family weighting, and external policies remaining distinct from Top-K sparse runtime.
- [ ] Run the complete targeted integration suite on a PyTorch-enabled environment.

### Task 4: Feed actual candidate outcomes into Memory and EvolutionLoop decisions

**Files:**
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/evolution/evaluator.py`
- Test: `tests/integration/test_standard_search_selection.py`
- Test: `tests/integration/test_evolutionary_compression.py`

- [x] Preserve same-parent quality/latency and hardware metrics, allocated/reserved memory, parameter/checkpoint size, and evaluator cost in the existing real evaluation and ExperienceStore records.
- [x] Update family GPU utility only from valid, feasible `measured_video` results for the same parent; structural probes and CPU/Fake results remain screening evidence.
- [x] Score candidate utility with the unchanged quality floor as a feasibility condition; family GPU utility now requires valid, feasible measured-video evidence from the same parent and an explicit full-evaluation GPU cost.
- [x] Preserve failures and invalid candidates in search-cost totals while preventing them from counting as discoveries; elapsed evaluator cost includes failed attempts, and GPU totals expose observation coverage instead of inferring missing cost.
- [x] Add integration coverage showing invalid/Fake/missing-cost/other-parent evidence cannot update real-GPU family utility, and failed-candidate time remains in the search-cost summary. Existing quality-floor selection tests confirm near misses cannot displace feasible candidates.

### Task 5: Version the paper protocol and add reproducible local experiment recipes

**Files:**
- Modify: `harness4h3/evolution/protocol.py`
- Modify: `tools/build_evolution_protocol_configs.py`
- Modify: `tools/validate_evolution_protocol.py`
- Modify: `configs/wan-rsi-paper-20261018/`
- Create: `configs/wan-rsi-hybrid-20261018/`
- Test: `tests/unit/test_evolution_protocol.py`
- Test: `tests/unit/test_build_evolution_protocol_configs.py`
- Test: `tests/unit/test_validate_evolution_protocol.py`

- [x] Add the default and expanded operator catalogs, runtime-policy adapter versions, and search-ranking version to the stable protocol fingerprint.
- [x] Keep the six arms matched on checkpoint, prompt/case/seed set, device, dtype, quality gate, evaluator revision, and candidate evaluation budget; keep output roots unique per arm.
- [x] Add local portable recipes for Dense, SVOO, MagCache, SVOO+MagCache, and RSI evaluation at 832×480×81 and 16 steps, with outputs rooted under the prescribed experiment subdirectory.
- [x] Validate six-arm recipes against the current source snapshot; the local structural preflight passes, while absent model/profile assets still require verification under `ai4mbse`.
- [x] Run the protocol unit, config builder, and validator suites (12 passed).

### Task 6: Run the same-parent real Wan experiment inside `ai4mbse`

**Files:**
- Read/write only under the user-supplied server `ai4mbse` absolute root.
- Use local recipes from `configs/wan-rsi-hybrid-20261018/`.
- Write all outputs under the relative directory `experiments/wan-rsi-hybrid-20261018/` inside that root.

- [ ] Confirm the supplied root basename is exactly `ai4mbse`; create the experiment directory only under that root.
- [ ] Before any CUDA run, record the allowed device's model, free memory, and active process list; start no experiment if the target device is occupied by an unrelated process.
- [ ] Copy the current source snapshot and experiment configs under the allowed root; verify source revision and checkpoint digest against the approved profile before running.
- [ ] Measure Dense, SVOO, MagCache, and SVOO+MagCache on the same parent, six validation case-seeds, and three timed repeats; save raw evaluator JSON and console output under the experiment directory.
- [ ] Run at least two RSI rounds using the corrected checkpoint-matched execution profile; compare each selected offspring to the same active parent with the same six case-seeds and evaluator settings.
- [ ] Run the six-arm protocol sequentially on the same allowed GPU, preserving independent arm output roots and recording all candidate failures and total GPU time.
- [ ] Summarize quality, latency, peak allocated/reserved memory, parameters, checkpoint size, search cost, and valid/nondominated discoveries. Do not claim improvement unless matched real Wan results support it.

## Completion Check

- [ ] Candidate-specific SVOO and MagCache policies survive save/reload and are visible in evaluation provenance.
- [ ] SVOO, MagCache, and Harness Top-K sparse runtime remain distinct in the search space and reports.
- [ ] Agent candidate selection uses measured evidence without summing overlapping profile categories and allocates the configured budget reproducibly.
- [ ] Combined runtime methods and RSI offspring have same-parent real Wan quality/latency/memory evidence under the unchanged gate.
- [ ] The six-arm protocol report includes all requested quality, runtime, memory, parameter, search-cost, and discovery outcomes, and every remote artifact is under `ai4mbse`.

## Local Implementation Checkpoint

- The SVOO/MagCache policy parsers, checkpoint persistence/replay, candidate evaluator installation, bounded external composition, weighted family allocation, and protocol fingerprint changes are implemented in the current local worktree.
- Local validation: 34 earlier targeted tests passed; this turn added 21 passing non-PyTorch search/program tests. A broader integration attempt produced 39 passes and 8 failures because the environment lacks optional PyTorch for Fake Wan adapter construction. `git diff --check` and Python 3.12 compileall passed.
- The six-arm current source fingerprint is `1290ae7+source-tree-v2:ac9be420e65ec3708b555db3957674748e4d20f12a8b5714c9d15e895e9b4580`; local preflight fingerprint is `3241bbc8706670815f9a9ac667dd2ccb3109c495c522ba64cf535c4cd2c273ff`. The preflight explicitly allowed missing checkpoint/profile assets and is not evidence that server assets are present.
- All five local baseline/RSI recipes resolve their input and output paths under the project root. The six-arm protocol also passes local structural validation with missing assets explicitly allowed; that check does not establish server-side asset availability.
- The checkpoint round-trip suites `test_model_variant.py` and `test_wan_compression.py`, plus the full evaluator/integration suites, could not execute because the local environment has no PyTorch. They were not treated as passing.
- No remote command was issued in this implementation turn. The user restricted server work to `ai4mbse`; only the folder basename is known, so no safe absolute root has been identified for the required GPU experiments.
