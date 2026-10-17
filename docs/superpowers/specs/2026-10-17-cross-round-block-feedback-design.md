# Cross-Round Block Search Feedback

## Goal

Make Block-only Algorithm Program evolution adapt to measured candidate outcomes across RSI rounds, instead of regenerating near-identical source and repeating the same Block subsets. Preserve the existing hard quality floor and keep failed compressed candidates ineligible for promotion.

## Evidence from the completed remote run

- Eight rounds measured eight Block plans each; every measurement was a valid transformer-output probe, but none reached the configured 0.98 quality floor. The best result was 0.9742358 for removing `block.17`.
- Two additional rounds stopped at `duplicate_algorithm_source` before candidate evaluation.
- The per-round trace contains plan, quality, feasibility, compute, and parent-checkpoint evidence. The history passed to the next algorithm improvement contains only round decisions and therefore exposes the generic `algorithm_search_failed` reason, not those candidate measurements.
- Parent selection scores only round-level experience evaluations. Since a failed search does not create one, later improvements repeatedly fall back to A0001.
- Repeated candidate plans are deduplicated only inside one round, so prior measured plans can consume a later round's evaluation budget again.

## Design

1. Build a compact, recent `algorithm_search_feedback` history item from each saved measured-search trace. Include the immutable parent checkpoint hash, quality floor, and each Block plan's digest, validity, measured quality, floor feasibility, efficiency proxy, and failure/recovery information.
2. Supply that evidence to the existing Agent improvement and search entrypoints. Keep the Agent responsible for proposing Block subsets and restore/replace actions. The prompt must explicitly require using measured outcomes and avoiding plans already tested against the same parent.
3. Select the algorithm parent using measured search evidence when no algorithm has a validated result. Prefer a quality-feasible candidate, then quality, measured score, efficiency, and the more recent algorithm on ties. This ranks algorithm search behavior only; it does not promote an under-floor model.
4. Seed each round's local duplicate set with plan digests measured against the same parent checkpoint. Report skipped plans as previously measured; do not spend full-evaluation budget on them. Measurements against a different parent remain eligible.
5. On an exact/logical duplicate algorithm source, make one bounded improvement retry using the duplicate source and explicit duplicate feedback. If the retry is still a duplicate, fail explicitly rather than disguising identical logic as a new algorithm.

## Boundaries

- Keep the active operator scope to `prune_blocks`; do not add or enable other compression operators.
- Keep the quality floor at 0.98; do not weaken feasibility, ranking, or promotion rules.
- Do not train or fine-tune weights. Recovery remains plan revision from the same immutable parent checkpoint.
- Preserve AlgorithmStore identity and duplicate checks. Retry duplicate generation instead of bypassing them.
- All model/GPU/video experiments remain remote. Local work is limited to code and tests.

## Verification

- Unit-test compact feedback extraction and parent selection using persisted trace fixtures.
- Test that previously measured plans for the same parent are skipped without invoking the evaluator, while the same plan remains eligible for a different parent.
- Test the bounded duplicate-source retry and its explicit feedback.
- Run focused fake-Wan unit/integration tests remotely, then a short remote smoke using the real Wan adapter with the existing bounded probe configuration.
- Only after the smoke is clean, launch a fresh full ten-round RSI in a new remote output directory. Report it as complete only when the remote summary says all rounds are terminal.
