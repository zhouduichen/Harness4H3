# Human Controller Directive Implementation Plan

> **For agentic workers:** Execute this plan task-by-task in the current worktree. The implementation will be performed inline because this side conversation does not permit sub-agents.

**Goal:** Let a user submit a bounded, auditable optimization objective while a remote H3 campaign is running, so the real LLM Controller sees it at the next planning boundary and must account for it in its validated `ExperimentPlan`.

**Architecture:** Reuse the existing append-only `ObservationStore` as the human-to-controller bridge. A small directive module validates and canonicalizes text, creates an idempotent `human_directive` observation, and a new `harness4h3 directive` CLI command appends it to the campaign output root. The existing campaign context loader automatically exposes the observation; the LLM prompt will explicitly classify it as advisory control input and preserve its ID through plan evidence. A directive never interrupts a worker or changes hard constraints, evaluators, operators, resource safety, or executable commands.

**Tech Stack:** Python 3, dataclasses, existing JSONL observation store, argparse, pytest.

## Global Constraints

- Preserve all unrelated dirty worktree changes.
- Do not touch or restart the live remote campaign while implementing or testing this feature.
- Keep directives bounded and plain text; never accept shell, SSH, CUDA, evaluator, target-profile, or checkpoint commands from the user bridge.
- Apply directives only at the next Controller plan; an already-running training/evaluation worker is not modified.
- Keep the append-only evidence and idempotency semantics of `ObservationStore`.

## Task 1: Add the directive contract and idempotent observation writer

**Files:**
- Create `harness4h3/controller/directive.py`.
- Add `tests/unit/test_controller_directive.py`.

**Implementation:**

1. Define a frozen `HumanDirective` value object with `directive_id`, bounded non-empty `instruction`, `apply_at`, `created_at`, `source_uri`, and canonical `source_sha256` fields.
2. Normalize surrounding whitespace, reject blank text and text beyond the documented limit, and use `apply_at="next_controller_plan"` as the only supported timing.
3. Generate a stable ID when the caller does not provide one; reject malformed/empty explicit IDs. Derive a canonical sorted JSON payload and SHA-256 from the directive identity, instruction, and apply boundary.
4. Convert the value object into a normal `ObservationRecord` with `kind="human_directive"`, a stable observation ID, `source_uri=directive://...`, and a summary containing only `directive_id`, `instruction`, and `apply_at`.
5. Implement `submit_directive(store, ...)` through `ObservationStore.append`, returning whether the observation was newly written. Reusing the same ID and payload is idempotent; reusing an ID for different content raises the store’s conflict error.
6. Add unit tests for normalization, bounds, deterministic identity, idempotent retry, conflicting identity, persisted JSONL shape, and the absence of executable-control fields.

**Verification:**

```bash
.venv/bin/pytest -q tests/unit/test_controller_directive.py
```

## Task 2: Expose a user-facing CLI injection command

**Files:**
- Modify `harness4h3/cli.py`.
- Add `tests/integration/test_directive_cli.py`.

**Implementation:**

1. Import `ObservationStore` and `submit_directive`.
2. Add `cmd_directive(args)` that resolves `--output-root`, writes to `observations.jsonl`, and emits the directive ID, observation ID, status, source URI/hash, and output root.
3. Register a `directive` subcommand with required `--output-root` and `--text`, optional `--directive-id`, and the standard `--json` output flag.
4. Test parser dispatch and both first submission and idempotent retry against a temporary output root. Verify the command does not create campaign state, event commands, or worker requests.

**Verification:**

```bash
.venv/bin/python -m harness4h3 directive --help
.venv/bin/pytest -q tests/integration/test_directive_cli.py
```

## Task 3: Make the real LLM prompt visibly consume human directives

**Files:**
- Modify `harness4h3/controller/provider.py`.
- Add focused assertions to `tests/unit/test_controller_providers.py`.

**Implementation:**

1. Extend bounded observation compaction so `human_directive` observations always retain their instruction, ID, and apply boundary even if the general summary allowlist changes.
2. Add explicit prompt rules: human directives are advisory next-plan objectives, all unconsumed directive IDs must be consumed, and they cannot override TargetProfile hard gates, registered operators, evidence requirements, resource safety, or trusted worker commands.
3. Keep the prompt’s existing bounded-context behavior and structured `ExperimentPlan` schema unchanged.
4. Test that a directive survives `_prompt_context`, appears in the generated prompt, and remains paired with `unconsumed_observation_ids`.

**Verification:**

```bash
.venv/bin/pytest -q tests/unit/test_controller_providers.py
```

## Task 4: Prove campaign-boundary consumption and safe re-planning

**Files:**
- Modify `tests/integration/test_remote_h3_closed_loop.py`.

**Implementation:**

1. Append a directive to a temporary campaign’s observation store and assert `_controller_context` exposes it as an unconsumed `human_directive`.
2. Use the existing fake Controller to return a plan consuming all current observations; assert the directive is recorded in validated plan evidence and campaign state.
3. Seed a valid pending resource plan, append a new directive, and assert the next `_train_one` boundary invalidates the pending plan and asks the Controller again rather than changing the running worker.
4. Assert the directive path does not alter the target profile, scheduler reservation, evaluator configuration, or worker command construction.

**Verification:**

```bash
.venv/bin/pytest -q tests/integration/test_remote_h3_closed_loop.py
```

## Task 5: Document operation and run regression checks

**Files:**
- Modify `docs/operator-contract.md` with the CLI command, timing semantics, and safety boundary.

**Implementation:**

1. Add a concise operator section showing how to submit the next-round objective to an existing campaign output root.
2. Run the focused directive/provider/campaign suites, then the complete test suite.
3. Review the diff to ensure only the new directive feature and its documentation/tests are included in the feature commit; leave the live campaign untouched.

**Verification:**

```bash
.venv/bin/pytest -q tests/unit/test_controller_directive.py tests/integration/test_directive_cli.py tests/unit/test_controller_providers.py tests/integration/test_remote_h3_closed_loop.py
.venv/bin/pytest -q
git diff --check
```

## Execution Handoff

Execute the tasks in order in this worktree. No user confirmation is needed between tasks: the design has already been approved and the command will be provided after verification. Do not inject a directive into a live campaign until the user supplies the concrete optimization text.
