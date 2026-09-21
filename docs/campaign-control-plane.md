# Campaign Control Plane

Harness4H3 now has a shared verifier-grounded campaign boundary for both the
Student path and the legacy H3 operator loop.

The control-plane path is deliberately narrower than repository editing:
controllers emit JSON candidate envelopes, while compiler, worker, evaluator,
review, gate, archive, and retention code remain trusted implementation
boundaries. A candidate cannot introduce Python, shell, evaluator rules, or an
arbitrary repository path into the action space.

## Immutable campaign identity

One campaign persists a `CampaignBase` containing the target profile, verifier
bank, dataset and evaluation recipe hashes, controller/Critical/evaluator
identities, prompt version, and capability snapshot. Every `DecisionEvent`
stores the base digest and its own payload digest. Changing the base requires a
new campaign; it cannot append to the old decision trace.

The append-only trace is written to `decision-trace.jsonl`. It records proposal
generation and validation, review, training, evaluation, gate, parent,
replanning, and stop events. The legacy H3 adapter writes the same envelope to
`campaign-<session>-decision-trace.jsonl` without changing operator execution.

## Student control mode

Programmatic callers enable the shared path by passing `campaign_base`, a
fail-closed `CapabilitySnapshot`, and an independent `ReviewPipeline` to
`StudentCampaign`. Providers should implement `propose_batch()` and return
three to five declarative proposals. The compatibility `propose()` API remains
available for the older single-proposal campaign mode.

Each batch is checked before review or training. A feasible candidate is
`promotable`, but the campaign only reports `target_satisfied` after the target
gate and minimum-round policy are satisfied. A failed batch retains its parent;
only a verified child can become the next parent. `experience.jsonl` keeps the
candidate, parent, verifier evidence, and predicted-versus-actual metric delta.

The focused offline contract test is:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_campaign_*.py \
  tests/unit/test_campaign_adapters.py \
  tests/integration/test_campaign_control_plane.py
```

The real Student worker/evaluator still requires the configured remote H3,
CUDA, and target-device services. Offline scripted tests exercise contracts,
lineage, failure attribution, and trace integrity only; they do not constitute
edge-device quality evidence.
