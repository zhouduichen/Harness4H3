# Independent Candidate Selection Design

## Problem

`tools/run_wan_independent_evaluation.py` currently loads `models/active.json` as
the candidate. A rejected Child remains `evaluated` rather than `active`, so an
independent run can validly measure the baseline twice while appearing to have
measured a candidate. This is especially misleading when the active model is
`M0000` and the run has real rejected Children.

## Design

Keep the existing default behavior for compatibility, and add an explicit
per-run candidate override:

```text
--candidate-model fixed=M0003 \
--candidate-model memory=M0003 \
--candidate-model rsi=M0003 \
--candidate-model coding=M0002
```

The CLI maps each run name to one candidate record under
`<run-root>/models/candidates/<MODEL_ID>.json`. When an override is present,
the tool must fail if the record does not exist; it must never fall back to
`active.json`. Without an override, `active.json` remains the selected
candidate and is reported as the legacy/default selection.

Every per-run summary records both `active_model_id` and
`candidate_model_id`, together with the candidate checkpoint identity. This
makes a baseline-as-candidate result explicit (`candidate_model_id == M0000`)
instead of silently inferred.

## Data flow and failure behavior

1. Parse repeated `NAME=MODEL_ID` overrides and reject malformed or duplicate
   run names before starting worker processes.
2. Resolve the baseline from `M0000.json`.
3. Resolve the explicit candidate record, or the active record when no override
   is supplied.
4. Run the existing independent evaluator unchanged.
5. Write the selected candidate ID and selection mode into the summary.
6. Missing candidate records raise a worker failure and produce a failed
   summary; no evaluation result is accepted as evidence.

## Verification

Unit tests cover default active selection, explicit rejected-Child selection,
missing candidate failure, malformed/duplicate CLI mappings, and summary
identity fields. The existing CPU suite remains the regression gate. After the
change, real held-out evaluations use the latest rejected Child from each run:
Fixed/Memory/RSI `M0003` and Coding `M0002`.
