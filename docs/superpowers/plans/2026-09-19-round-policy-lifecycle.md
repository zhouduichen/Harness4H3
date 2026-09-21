# RoundPolicy Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the bounded LLM-produced `RoundPolicy` durable and enforceable without allowing speculative prefetch to change the search policy.

**Architecture:** Carry an optional strict policy beside each `ExperimentPlan`; validate it against runtime substrate/evaluation facts in `RemoteCampaign`; persist the active policy atomically in a sidecar and campaign state. Existing plans without a policy remain valid and keep current behavior.

**Tech Stack:** Python 3, frozen dataclasses, JSON schema, atomic JSON state, pytest.

## Global Constraints

- The remote campaign and idle watcher remain paused during implementation and sync.
- Only primary Controller plans may activate a policy; prefetch and parallel plans are read-only.
- Foreign GPU processes are never killed and no worker command/path is LLM-controlled.
- Existing legacy plan/state formats remain readable.

---

### Task 1: Carry a strict optional policy in plan output

**Files:**
- Modify: `harness4h3/controller/schemas.py`
- Modify: `harness4h3/controller/provider.py`
- Test: `tests/unit/test_controller_providers.py`

- [x] **Step 1: Add failing schema assertions**

Add a valid optional `round_policy` to the provider fixture and assert it survives
`ExperimentPlan.from_dict(...).to_dict()`. Assert a malformed policy value is rejected.

- [x] **Step 2: Implement the optional field and JSON schema**

Add `round_policy: Optional[Mapping[str, Any]] = None` at the end of
`ExperimentPlan`, parse it only when it is an object, and add a strict optional
`round_policy` property to `experiment_plan_json_schema` using the `RoundPolicy`
fields and current operator names.

- [x] **Step 3: Update the Controller prompt**

Tell the local LLM that a policy may be included only for a primary plan, must use
the supplied operator set, and is omitted for `parallel_gpu_fill`; state that the
campaign will independently verify digests, budgets and leases.

- [x] **Step 4: Run focused provider tests**

Run `./.venv/bin/python -m pytest -q tests/unit/test_controller_providers.py`.

### Task 2: Validate and persist the active policy at the campaign boundary

**Files:**
- Modify: `research/experiments/remote_h3_closed_loop.py`
- Modify: `harness4h3/controller/round_policy.py`
- Test: `tests/integration/test_remote_h3_closed_loop.py`

- [x] **Step 1: Add failing campaign tests**

Cover primary policy activation, rejection of a policy on a prefetch plan, rejection
of a disallowed operator/GPU lower bound/budget, and restart loading from the sidecar.

- [x] **Step 2: Add runtime digest and policy validation helpers**

Derive the substrate digest from target/workflow/worker contract/configured operator
facts and use `_evaluation_signature("heldout")` for the fixed evaluator digest.
Call `validate_round_policy` against the live scheduler GPU count and registry.

- [x] **Step 3: Persist only primary policies atomically**

Write `active-round-policy.json` and mirror the normalized value to
`campaign_state.json` only after the plan passes normal schema, evidence, operator and
resource validation. Emit `round_policy_activated` or `round_policy_unavailable`.

- [x] **Step 4: Enforce the active policy on later plans**

Reject a plan whose operator is not allowed, whose distributed minimum is below the
policy minimum, or whose declared GPU-hour budget exceeds the policy budget. Leave
legacy campaigns with no active policy unchanged.

- [x] **Step 5: Run focused integration tests**

Run `./.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k 'round_policy or controller_plan'`.

### Task 3: Full verification and paused remote synchronization

**Files:**
- Modify: `docs/superpowers/specs/2026-09-18-a-evolve-round-policy-design.md`
- Modify: `docs/superpowers/plans/2026-09-18-a-evolve-round-policy.md`

- [x] **Step 1: Record policy lifecycle and legacy compatibility**

Update the A-Evolve design and mark the completed policy lifecycle tasks.

- [x] **Step 2: Run verification**

Run `python3 -m py_compile`, `bash -n tools/*.sh`, `git diff --check`, and
`./.venv/bin/python -m pytest -q`.

- [x] **Step 3: Sync without starting services**

Run `REMOTE_START_CONTROLLER=0 tools/sync-remote-pipeline.sh`, activate the staged
files over SSH, and verify campaign `state=paused`, campaign `pid=none`, watcher
`pid=none`, and `.operator-paused` present.
