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

- [ ] Add policy parsers that accept `svoo_attention: {method: "svoo", model_key: "wan21-t2v-1.3b", config: {...}}` and `magcache: {config: {...}}`, reject missing method names and non-mapping config, and preserve unknown official upstream config fields.
- [ ] Add one shared runtime-library policy executor in `compression.py` that loads the parent, calls the adapter's runtime-policy persistence method, saves and reloads the child, verifies the persisted policy round-trips, and sets `parameter_reduction_ratio` to `0.0`.
- [ ] Add `apply_external_runtime_policy(model, policies)` to the Diffusers Wan adapter; store policy metadata with the transformer config's existing Harness metadata and include it in the adapter's checkpoint digest.
- [ ] Register `svoo_attention` and `magcache` in the default search space and compression registry; keep `sparse_runtime` registered under its existing Top-K name and behavior.
- [ ] Add both names to `ModelVariant.from_compression`'s runtime operation set so `ordered_operations` preserves order while `runtime_policy` reports the methods separately from structural checkpoint changes.
- [ ] Run `pytest tests/unit/test_model_variant.py tests/unit/test_wan_compression.py -q`; confirm runtime-only variants retain base checkpoint identity and parameter reduction remains zero.

### Task 2: Apply candidate-specific runtime methods in the real Wan evaluator

**Files:**
- Modify: `harness4h3/evolution/evaluator.py`
- Modify: `harness4h3/evolution/loop.py`
- Test: `tests/unit/test_wan_video_evaluator.py`
- Test: `tests/integration/test_evolutionary_compression.py`

**Interface:** `WanVideoEvaluator.__call__(child, profile)` reads the persisted Harness policy from the candidate checkpoint, installs the ordered runtime methods through the existing Diffusers pipeline hook path, and records method/config/order plus available runtime counters in the candidate result.

- [ ] Read the candidate's persisted runtime policy before constructing the evaluation cache key; hash the normalized policy together with checkpoint SHA and recipe SHA so Dense, SVOO, MagCache, and their composition cannot share cached results.
- [ ] Install SVOO first through `sparsevideo.apply` and then MagCache through `transformer.enable_cache(MagCacheConfig(...))`, matching the existing successful combination order.
- [ ] Keep the evaluator-level baseline Dense unless the configured baseline explicitly names a method; apply candidate policy only to the candidate side of `__call__`.
- [ ] Include normalized candidate runtime policy, installation order, upstream versions, post-generation SVOO `handle.summary()`, and MagCache config in both per-case and aggregate evaluation JSON.
- [ ] Add unit coverage with a fake pipeline loader that asserts policy-specific installation order and verifies that two policies produce distinct cache identities.
- [ ] Run `pytest tests/unit/test_wan_video_evaluator.py -q` and the targeted runtime-policy integration case in `tests/integration/test_evolutionary_compression.py`.

### Task 3: Generate bounded mixed-granularity candidates and allocate measured search budget

**Files:**
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/evolution/loop.py`
- Test: `tests/unit/test_algorithm_program_search.py`
- Test: `tests/unit/test_algorithm_improvement.py`
- Test: `tests/integration/test_standard_search_selection.py`

**Interface:** The generated search program returns at most `state.budget.max_candidates` unique plans, a deterministic rank, explicit per-family allocated counts, and an evidence record per candidate. It retains single-method controls and creates only compatible, at-most-two-method `compose` plans.

- [ ] Keep profile component evidence separated by semantic category and attach the `module_timing_scope`; use the measured self-attention and FFN values for target ordering without adding overlapping inclusive values.
- [ ] Add standalone SVOO and MagCache plans when their operators exist in the active search space; add the SVOO+MagCache ordered composite as the first supported external-method interaction.
- [ ] Add other two-method combinations only when both component sensitivities exist and both operator specs are active; reject nested composites, repeated families, and pairs already measured on the same parent.
- [ ] Replace family-order round robin with deterministic weighted allocation: seed weights from profile bottleneck order and standalone evidence; reserve one probe for each available family until one sample exists; distribute remaining slots by paired quality-gated latency gain per measured GPU second, with memory and parameter constraints applied before ranking.
- [ ] Persist `allocation_reason`, `evidence_refs`, family sample counts, and candidate utility inputs in the search trace; label missing timing evidence as unavailable instead of substituting FLOP estimates.
- [ ] Add tests proving (a) inclusive categories are never summed, (b) SVOO+MagCache preserves order, (c) duplicate same-parent pairs are omitted, and (d) family budget and rank are deterministic under equal inputs.
- [ ] Run the targeted unit and integration commands listed above.

### Task 4: Feed actual candidate outcomes into Memory and EvolutionLoop decisions

**Files:**
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/agent.py`
- Modify: `harness4h3/evolution/evaluator.py`
- Test: `tests/integration/test_standard_search_selection.py`
- Test: `tests/integration/test_evolutionary_compression.py`

- [ ] Store each valid candidate's same-parent quality delta, end-to-end latency delta, peak allocated/reserved memory delta, parameter/checkpoint-size changes, method order, evaluator cost, and validity in the existing experiment/ExperienceStore record.
- [ ] Update learned family scores only from completed real-Wan candidate evaluations; keep structural probes and CPU/Fake results marked as screening evidence and exclude them from promotion utility.
- [ ] Score candidate utility with the unchanged quality floor as a feasibility condition; then rank feasible candidates by latency improvement, memory cost, parameter change, and GPU evaluation cost according to the existing promotion policy fields.
- [ ] Preserve failures and invalid candidates in search-cost totals while preventing them from counting as discoveries.
- [ ] Add integration coverage showing a quality-gate failure cannot win through a large latency improvement and an invalid/Fake candidate cannot update real-GPU family utility.

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

- [ ] Add candidate operator catalog and runtime-policy versions to the stable protocol fingerprint so experiment artifacts identify the exact SVOO/MagCache adapter and search semantics.
- [ ] Keep the six arms matched on checkpoint, prompt/case/seed set, device, dtype, quality gate, evaluator revision, and candidate evaluation budget; keep output roots unique per arm.
- [ ] Add local recipes for Dense, SVOO, MagCache, SVOO+MagCache, and RSI-selected candidate evaluation with the fixed 832×480×81, 16-step recipe.
- [ ] Validate the six-arm protocol and write manifests without opening or mutating remote assets.
- [ ] Run `pytest tests/unit/test_evolution_protocol.py tests/unit/test_build_evolution_protocol_configs.py tests/unit/test_validate_evolution_protocol.py -q`.

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
