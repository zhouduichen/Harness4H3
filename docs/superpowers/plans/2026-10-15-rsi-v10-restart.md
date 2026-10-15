# RSI v10 Experiment Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute the steps inline with a test checkpoint after each code change.

**Goal:** Remove the known algorithm-provider context failure and run one clean, comparable four-arm Wan compression experiment.

**Architecture:** Keep the existing evaluator and remote runner unchanged. Bound the provider request by deduplicating repeated memory records, requesting a concise source body, and reserving enough output space below the controller context limit. Generate a fresh protocol from the resulting commit, then run Fixed, Coding Agent, Agent + Memory, and RSI sequentially in a new output root.

**Tech Stack:** Python, pytest, Git bundles, SSH, vLLM-compatible JSON API, existing Harness4H3 evolution CLI.

## Global Constraints

- Preserve unrelated Student/H3 working-tree changes.
- Do not modify historical experiment decisions.
- Do not mix v9 partial results with the clean v10 comparison.
- Stop and diagnose any new infrastructure failure before continuing to the next arm.
- Use the existing 16-case, three-latency-measurement evaluation protocol.

### Task 1: Bound provider context and add regression coverage

**Files:**
- Modify: `harness4h3/evolution/agent.py`
- Modify: `tests/unit/test_algorithm_evolution.py`

- [x] Deduplicate records present in both history and skills, constrain generated source length, and set the request output budget to 1536 tokens.
- [ ] Update the expected request-budget assertion and run the focused evolution tests.
- [ ] Run the full CPU suite, compile checks, and `git diff --check`.

### Task 2: Commit and synchronize the experiment checkout

**Files:**
- Commit only the two code/test files and this plan.

- [ ] Create a task-scoped commit on `codex/rsi-compression-loop`.
- [ ] Push the commit to `origin`.
- [ ] Bundle the branch and checkout the exact commit in `/home/intern/huangjiahao/Harness4H3-rsi-codex`.

### Task 3: Run and verify the clean four-arm experiment

**Files:**
- Create remote protocol/output data under `/data/models/harness4h3-evolutionary-compression-20261015-v10`.

- [ ] Build a fresh protocol whose source revision is the new commit.
- [ ] Run all four arms sequentially with the existing remote controller and three rounds per arm.
- [ ] Inspect each `round.json` and stop on any `failed` status rather than treating it as a result.
- [ ] Compare only the completed v10 arms with `tools/compare_evolution_runs.py --expected-cases 16`.

### Task 4: Report the evidence boundary

**Files:**
- Update: `docs/superpowers/reports/2026-10-14-evolution-evaluation-audit.md`

- [ ] Record the actual v10 protocol fingerprint, source revision, test count, and arm status.
- [ ] Separate measured effects from hypotheses about RSI versus Memory.
- [ ] State whether the minimum evidence for continual self-improvement has been met.

