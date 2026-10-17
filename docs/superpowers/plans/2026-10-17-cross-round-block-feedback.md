# Cross-Round Block Search Feedback Implementation Plan

**Goal:** Make the approved Block-only RSI use prior measured candidate results, prefer evidence-backed algorithm parents, and avoid repeated evaluations without relaxing the 0.98 quality floor.

**Architecture:** Read persisted `algorithm-program-trace.json` files as the source of candidate-level feedback. Pass a compact bounded history to the existing Algorithm Agent, use trace measurements for algorithm-lineage ranking, and seed same-parent plan deduplication from prior rounds. Preserve the current model promotion gate and operator registry.

**Verification location:** All Python tests and Wan smoke/experiments run on the remote server. Local checks are limited to static inspection and `git diff --check`.

## Task 1: Persisted candidate feedback for the Agent

**Files:** `harness4h3/evolution/loop.py`, `harness4h3/evolution/agent.py`; unit tests in `tests/unit/test_algorithm_evolution.py`.

- [ ] Add a bounded history reader that extracts recent same-parent Block candidate plans, digests, quality/floor results, efficiency proxy, validity, recovery action, and parent checkpoint identity from saved traces.
- [ ] Include all tested Block sets and the four most informative results from up to four recent rounds ahead of selection-only history; preserve complete traces for internal scoring/deduplication.
- [ ] Update the standard improvement/search instruction to use the evidence, never claim an under-floor result is feasible, and avoid previously measured same-parent plans.
- [ ] Test trace extraction, context compaction, and prompt payload with representative measured candidates.

## Task 2: Evidence-based algorithm parent and duplicate-source retry

**Files:** `harness4h3/evolution/loop.py`; tests in `tests/unit/test_algorithm_evolution.py` or a focused loop test module.

- [ ] Rank algorithms with measured trace evidence when no validated algorithm exists: feasibility first, then quality, search score, efficiency, and recency.
- [ ] Keep this ranking separate from model promotion; under-floor search evidence can select an algorithm parent but cannot select a model candidate.
- [ ] On duplicate source/logic, retry improvement once from the existing duplicate with an explicit `duplicate_algorithm_source` reason and the same Block feedback.
- [ ] Preserve `AlgorithmStore` duplicate checks; test successful retry and bounded failure on a second duplicate.

## Task 3: Cross-round same-parent candidate deduplication

**Files:** `harness4h3/evolution/loop.py`; tests in `tests/integration/test_standard_search_selection.py`.

- [ ] Seed the per-search seen-plan set with prior measured digests only when the immutable parent checkpoint hash matches.
- [ ] Record cross-round duplicate proposals in `not_evaluated` with an explicit reason and do not call the evaluator for them.
- [ ] Verify a previously measured plan remains eligible when the parent checkpoint differs.
- [ ] Verify no candidate below 0.98 can be selected even when its search score is the best available near miss.

## Task 4: Remote verification and full experiment

- [ ] Run syntax/diff checks locally; sync only source/tests/config to an isolated remote directory, never model weights.
- [ ] Run focused unit and integration tests remotely.
- [ ] Seed isolated smoke/full output roots with only prior algorithm candidates and R0001-R0010 decision/trace evidence; keep the model baseline clean and do not copy checkpoints/videos.
- [ ] Run a short real-Wan remote smoke in its seeded output directory and verify the structured feedback, same-parent dedupe, quality floor, and terminal trace.
- [ ] If smoke passes, launch ten new RSI rounds in a separate seeded output directory (R0011-R0020 IDs are acceptable). Do not continuously poll; capture PID/log/start status and expected duration, then report completion status based on the saved run summary.

## Non-goals

- No operator or search-space expansion, no lowering of the quality floor, no local model download, no training/fine-tuning, and no claim of speedup from structural estimates alone.
