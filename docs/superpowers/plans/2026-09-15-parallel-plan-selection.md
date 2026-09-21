# Parallel Controller Plan Selection Implementation Plan

**Goal:** Batch-generate four structured ExperimentPlan candidates through one vLLM instance and select one candidate safely before the existing campaign validation/execution path.

**Architecture:** Extend the vLLM provider with a batched candidate request and a short structured selector request. Keep the campaign contract unchanged: it still receives one `ExperimentPlan`, and all existing validation, scheduling, and evaluator gates remain authoritative. Increase the launcher concurrency ceiling to four sequences without creating extra model replicas.

**Tech Stack:** Python 3.9+, urllib, PyYAML, vLLM OpenAI-compatible Chat Completions, pytest, Bash.

## Global Constraints

- One vLLM process and one model replica only.
- Candidate generation may be batched; only the selected plan may be executed.
- Existing schema, evidence, resource, worker, and evaluator validation remains authoritative.
- Candidate generation defaults to 4 sequences and bounded completion lengths.

### Task 1: Extend the vLLM provider

**Files:**
- Modify: `Harness4H3/harness4h3/controller/provider.py`
- Test: `Harness4H3/tests/unit/test_controller_providers.py`

**Interfaces:**
- Add provider configuration fields for candidate count/temperature and selector token budget.
- Keep `plan(context) -> ExperimentPlan` unchanged for all callers.
- Expose diagnostic fields for candidate count, request ids, selected index, and fallback reason.

- [x] Add request parameters for `temperature` and `n` while preserving existing review/probe defaults.
- [x] Parse multiple choices and discard only malformed choices.
- [x] Add a strict selector schema and fallback behavior.
- [x] Add tests for multi-choice parsing, selector choice, selector fallback, and legacy single-choice response.

### Task 2: Update runtime configuration

**Files:**
- Modify: `Harness4H3/configs/controller.yaml`
- Modify: `Harness4H3/tools/controller-wait-launch.sh`

**Interfaces:**
- Configuration defaults expose four candidate sequences and bounded plan/selector budgets.
- `CONTROLLER_MAX_NUM_SEQS` remains an environment override.

- [x] Set candidate generation defaults to 4, temperature 0.25, candidate max tokens 2048, and selector max tokens 384.
- [x] Set launcher default `MAX_NUM_SEQS` to 4 and retain the existing CLI flag.

### Task 3: Verify without starting a real campaign

**Files:**
- No new runtime files.

**Interfaces:**
- Run provider unit tests and syntax checks only.

- [x] Run targeted controller provider tests.
- [x] Run Python compilation checks for modified Python files.
- [x] Run shell syntax check for the launcher.
- [x] Tests do not start a remote process; the live campaign was separately verified with batched candidate generation.
