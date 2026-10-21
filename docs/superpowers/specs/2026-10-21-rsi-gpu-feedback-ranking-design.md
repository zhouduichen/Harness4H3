# RSI GPU Feedback Ranking and Candidate Novelty

## Context

The real Wan rounds R71–R82 repeatedly selected `sparse_runtime` keep ratios and three rounds failed before GPU measurement because their candidates were duplicates. R113–R115 later produced valid videos, but all three were rejected for semantic quality and/or peak-memory regression. The active family utility only considered feasible video results and ranked families using latency saved per GPU second, so those failures did not discourage another attention-only search.

## Design

Keep the current measured structural probe, candidate budget, evaluation pipeline, and promotion thresholds. For each operator family measured on the exact same parent checkpoint, use every valid independent video result, including rejected candidates. Derive one auditable family utility from parent-relative semantic retention, peak-memory improvement, and latency improvement; expose its components and evidence count in the search trace. Use that utility only to allocate measured candidate slots between families. Missing GPU evidence remains unknown and does not become a fabricated score.

Before storing an improved algorithm, validate its proposal and ranked search plans against the active operator schemas and the same-parent plan blocklist. If a response is schema-invalid or contains only duplicate plans, feed the concrete validation failure back and allow at most two additional provider attempts. Prefer unseen registered cache-plus-attention compositions, particularly BWCache with SVOO or another supported self-attention policy; use them only when both operators and their argument schemas are active. Keep the existing promotion gates unchanged.

Raise the improvement provider output budget to 8192 tokens and its request timeout to 900 seconds. Record the source hashes, budget, retry count, selected plan, independent video metrics, and unchanged promotion policy in the next recovery attempt. Verify with a bounded real Wan GPU round before starting the following six-round batch.

## Alternatives

1. Prompt-only guidance is small but leaves family allocation blind to failed video quality and memory outcomes.
2. Replacing probe scores with independent GPU metrics would require a full GPU video evaluation for every candidate and exceed the current search budget.
3. **Recommended:** retain probe ranking within a round, add parent-matched GPU family utility for allocation across families, and retry only schema-invalid or duplicate candidates. This reuses current telemetry and keeps the promotion decision authoritative.

## Scope and success criteria

This change touches only the existing RSI Agent, measured-search ranking, recovery config, and run manifest. It does not add compression operators or alter promotion thresholds. Success means a new candidate is schema-valid and unseen on the active parent, the trace shows how semantic quality and peak memory influenced family order, the real GPU evaluation completes, and promotion remains governed by the existing gates.
