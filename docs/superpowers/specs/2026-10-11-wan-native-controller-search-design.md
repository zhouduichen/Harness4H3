# Wan-Native 5s Controller Search Design

## Goal

Restore bounded Controller recipe search only after the verified native Wan 32→16 result, while keeping the search on the authenticated native 5s H3 supervision path.

## Scope and non-goals

The Controller may search only progressive-distillation loss and optimization fields that are meaningful for the native 32→16 trajectory:

- learning rate and train steps;
- same-state endpoint, flow, and clean-anchor weights;
- rollout steps, rollout frequency, rollout endpoint weight, and rollout clean weight;
- teacher reuse and the existing trainable scope/precision fields.

Every trial uses the native 5s cache, one immutable real Wan 32-step trajectory record, target 16 steps, the fixed four-prompt/three-seed manifest, the visual information-content veto, failure memory, and the frozen legacy R2 promotion rule. The Controller cannot select 30s continuation, R2, DMD2, architecture changes, code edits, or a different Teacher/cache.

## Architecture

The existing `wan_vllm_campaign.py` remains the control plane. Its trusted proposal schema gains `endpoint_weight` and its command builder passes all five loss weights plus the native rollout-state binding to `wan_student_train_worker.py`. A new native campaign config binds the initial parent checkpoint to the successful F3 candidate and binds every round to the immutable seed-20261010 native trajectory record.

The Controller still proposes JSON only. The trusted validator checks the parent digest, binary 32→16 curriculum, native search-space membership, and the fixed native data/trajectory contract before execution. F1/F2/F3 artifacts remain independent under the campaign root; no candidate is accepted unless F3 has 12/12 visual passes, quality is at least Base Wan32 and frozen R2, scheduler parity is true, resource metrics are present, and the checkpoint digest is valid.

## Data flow

1. Evaluate the current parent with the native 5s four-prompt/three-seed manifest.
2. Ask vLLM for one bounded recipe.
3. Validate the recipe and bind it to the current parent digest.
4. Run the real worker in `native_rollout_state` against the fixed seed-20261010 trajectory.
5. Run F1→F2→F3 with the same manifest and visual veto.
6. Append proposal, training, evaluation, decision, and failure-memory evidence; advance the parent only for an accepted candidate.

## Acceptance and safety

The existing campaign gate remains fail-closed. `controller_search_enabled` is recorded as true only in this new native campaign; the old R2 checkpoint remains ineligible. Any missing trajectory, wrong digest, non-native mode, unsupported weight, visual collapse, or quality/resource regression rejects the round without mutating the parent.

## Verification

- Unit tests cover the new schema field, search-space validation, and native command arguments.
- Existing native trajectory, adapter, correctness-gate, and campaign tests remain green.
- The first real campaign is bounded to one Controller round and retains all F1/F2/F3 evidence for paper reporting.
