# Persisted GPU Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax.

**Goal:** Persist one validated GPU sibling from a primary n-way Controller response so evaluation and restart recovery can launch it without a duplicate LLM request.

**Architecture:** Extend the existing primary prefetch persistence boundary in \`RemoteCampaign\`. When the selected plan is CPU-only, derive one distributed GPU candidate with the existing \`_parallel_candidate_from_prefetch_handle\` guard and write it to the isolated \`parallel_prefetched_*\` cursor. Evaluation and resume already prefer that cursor; stale parent/cursor data is rejected by the existing lineage and scheduler gates. No full candidate batch or checkpoint bytes are persisted.

**Tech Stack:** Python 3, JSON campaign state, existing \`RemoteCampaign\` validation/scheduler, pytest.

## Global Constraints

- The remote campaign remains paused; no experiment, controller service, idle watcher, or worker is started during implementation.
- Persist at most one GPU sibling candidate; do not enlarge Controller context with the complete n-way batch.
- Preserve the primary selected plan and its \`prefetched_*\` cursor unchanged.
- A persisted sibling is only a proposal; execution still requires schema, operator, capability, lineage, checkpoint, foreign-process, and GPU-lease gates.
- Never kill or release a foreign process or lease.
- Checkpoint retention and append-only experience/evaluation metadata remain unchanged.

---

### Task 1: Reproduce the restart-safe loss of the n-way GPU candidate

**Files:**
- Modify: \`tests/integration/test_remote_h3_closed_loop.py\`
- Inspect: \`research/experiments/remote_h3_closed_loop.py:4838-5008\`

**Interfaces:**
- Use \`_start_controller_prefetch(...)\` and \`_finish_controller_prefetch(...)\` as the existing persistence boundary.
- Assert the state keys \`prefetched_plan\` and \`parallel_prefetched_plan\` independently.

- [ ] **Step 1: Write the failing test**

Add a test with a fake Controller that returns a CPU-only selected plan and a legal distributed GPU alternative through \`last_eligible_candidates\`. Start and finish a primary prefetch, then assert the primary plan is persisted and the parallel cursor contains the GPU alternative. The sibling request must use \`elastic=True\`, \`distributed=True\`, \`exclusive=False\`, \`evaluation_workers=1\`, \`min_gpu_count=2\`, \`max_gpu_count=2\`, and \`gpu_count=2\`.

Expected assertions:

\`\`\`python
state = campaign._load_campaign_state()
assert state["prefetched_plan"]["plan"]["experiment_id"] == "exp_0002"
assert state["parallel_prefetched_plan"]["plan"]["operator"] == "recovery_finetune"
assert state["parallel_prefetched_training_calls"] == 1
\`\`\`

- [ ] **Step 2: Run the focused test to verify it fails**

\`\`\`bash
.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k persisted_batch
\`\`\`

Expected failure: the parallel cursor is absent.

---

### Task 2: Persist the single guarded GPU sibling

**Files:**
- Modify: \`research/experiments/remote_h3_closed_loop.py:5510-5570\`
- Test: \`tests/integration/test_remote_h3_closed_loop.py\`

**Interfaces:**
- Reuse \`_parallel_candidate_from_prefetch_handle(handle, plan, int(handle["training_calls"]))\`.
- Reuse \`_persist_batched_parallel_candidate(handle, candidate, plan.experiment_id)\`.

- [ ] **Step 1: Add the minimal persistence hook**

After the selected primary state is written inside \`_persist_controller_prefetch\`, and only when \`state_key == "primary"\` and \`plan.operator\` is \`prune_blocks\` or \`quantize\`, derive one sibling with the existing helper. If it returns a candidate, call \`_persist_batched_parallel_candidate\` and append \`controller_plan_parallel_prefetch_armed\` with reason \`primary_prefetch_persisted_gpu_candidate\`. Do not persist alternatives for GPU primary plans or unfiltered candidates.

The hook must be equivalent to:

\`\`\`python
sibling = self._parallel_candidate_from_prefetch_handle(
    handle, plan, int(handle.get("training_calls", 0))
)
if sibling is not None:
    self._persist_batched_parallel_candidate(handle, sibling, plan.experiment_id)
\`\`\`

- [ ] **Step 2: Run the focused test to verify it passes**

\`\`\`bash
.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k persisted_batch
\`\`\`

Expected output: the new test passes.

- [ ] **Step 3: Add stale-state coverage**

Add assertions that changing \`parallel_prefetched_training_calls\` or \`parallel_prefetched_parent_model_id\` makes the persisted sibling unusable at the evaluation boundary and routes control to the existing filtered fallback path. This proves persistence does not weaken lineage safety.

---

### Task 3: Verify no duplicate request and preserve existing behavior

**Files:**
- Modify: \`tests/integration/test_remote_h3_closed_loop.py\`
- Modify: \`docs/superpowers/specs/2026-09-19-evaluation-gpu-fill-queue-design.md\`

**Interfaces:**
- The evaluation callback continues to check durable \`parallel_prefetched_plan\` before opening a filtered \`parallel_gpu_fill\` request.
- Existing event names remain compatible; add only the explicit persistence reason.

- [ ] **Step 1: Add the no-duplicate assertion**

Seed a persisted primary plan plus its \`parallel_prefetched_plan\`, invoke the evaluation overlap selection path, and assert the Controller call count does not increase and \`controller_plan_parallel_reused\` appears before \`speculative_worker_started\`.

- [ ] **Step 2: Update the design invariant**

Document that the primary prefetch may persist exactly one guarded GPU sibling, while the full n-way response remains transient and the persisted sibling remains subject to normal launch gates.

- [ ] **Step 3: Run focused and full checks**

\`\`\`bash
.venv/bin/python -m pytest -q tests/integration/test_remote_h3_closed_loop.py -k 'persisted_batch or parallel_gpu_fill or prefetch'
.venv/bin/python -m pytest -q
python3 -m py_compile research/experiments/remote_h3_closed_loop.py
git diff --check
\`\`\`

Expected output: focused tests pass, the full suite remains green, and \`git diff --check\` emits no output.

- [ ] **Step 4: Verify the paused remote state without starting anything**

\`\`\`bash
ssh Jiayu-intern 'cd /home/intern/huangjiahao/Harness4H3-rsi && tools/remote-campaign-service.sh status && tools/remote-idle-autostart.sh status && test -f var/remote-h3-controller-20260914/.operator-paused && echo operator_pause=present'
\`\`\`

Do not start or kill any remote process. When synchronization is later requested, use \`REMOTE_START_CONTROLLER=0\` and recheck \`pid=none\`, \`state=paused\`, and the pause marker.
