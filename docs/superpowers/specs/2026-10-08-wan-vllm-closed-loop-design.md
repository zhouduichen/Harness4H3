# Wan vLLM Closed-Loop Experiment Design

## Goal

Connect the real Wan Student training worker to a resumable vLLM-controlled
experiment loop. The loop must evaluate a finished checkpoint on the fixed
30-second multi-prompt/multi-seed manifest, ask vLLM for a constrained training
proposal, validate that proposal locally, train from the selected parent, and
evaluate the child before selecting the next parent.

vLLM may propose training-policy values only. It must not emit Python, shell,
paths, device assignments, model architecture, or code edits. The trusted
worker remains the only component that chooses files, devices, Teacher
checkpoint, and execution commands.

## Architecture and data flow

`wan_vllm_campaign.py` is the remote supervisor and owns one campaign root.
For each round it:

1. evaluates the current parent checkpoint with the immutable evaluation
   manifest and the 30-second continuation contract;
2. sends the prior evaluation, lineage, and round history to the configured
   OpenAI-compatible vLLM endpoint;
3. parses and validates one `WanTrainingProposal` locally;
4. launches the existing trusted `wan_student_train_worker.py` with only the
   validated numeric overrides and the trusted parent checkpoint;
5. evaluates the resulting checkpoint and records the complete round;
6. selects the child only when it is valid, contract-compliant, and improves
   semantic quality without violating the configured aggregate-quality floor.

The loop writes `proposal.json`, `training-result.json`,
`evaluation-result.json`, `decision.json`, and append-only `events.jsonl` per
round, plus an atomic `campaign-result.json`. A failed proposal, training
failure, invalid video, missing semantic evidence, or lineage mismatch stops
the campaign without silently falling back to a rule-based proposal.

## Proposal contract

The vLLM response is a JSON object with exactly these fields:

```json
{
  "schema_version": 1,
  "proposal_id": "wan_0001",
  "parent_checkpoint_sha256": "sha256:...",
  "steps": 8,
  "teacher_reuse_steps": 2,
  "learning_rate": 0.0000005,
  "focus": "semantic",
  "hypothesis": "..."
}
```

The validator restricts steps, reuse, learning rate, focus, identifier format,
and parent digest. The campaign overwrites executable paths and parent path
from trusted state; those values never come from vLLM.

## Evaluation and selection

Every candidate uses the same manifest, prompts, seeds, Wan continuation
window geometry, 24 FPS output, 708–732 frame contract, and
`offline_simulation=false`. The evaluator records semantic, temporal, motion,
aggregate quality, latency, memory, and validity evidence. A candidate is
eligible only if all cases decode, continuation is verified, the semantic
score is present, and the Student remains above the configured Teacher-relative
semantic floor. Selection is deterministic: maximize semantic score first,
then aggregate quality, then lower latency.

## Operational safeguards

- Existing jobs are not killed or reused as campaign output roots.
- The campaign root lives on `/data`, not the quota-limited home filesystem.
- A parent checkpoint must exist and its SHA-256 must match the proposal and
  training result.
- Each round starts only after the previous round's result is durable.
- A campaign is resumable only from complete round artifacts; partial rounds
  are marked failed rather than guessed through.

## Verification

Unit tests cover proposal parsing, range validation, parent digest binding,
deterministic selection, and failure-stop behavior. A remote smoke run uses
the existing completed Wan checkpoint as parent and one round before enabling
the longer experiment budget.
