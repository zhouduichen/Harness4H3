# Deterministic Algorithm History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent deterministic compression baselines from re-proposing the same selection logic across rounds while keeping evaluator Memory disabled for the fixed baseline.

**Architecture:** `EvolutionLoop` will build a local selection history from prior round decisions in the current output root. That history contains only round/algorithm/status/operator selection data, not evaluation results. It is passed to `AlgorithmAgent` together with SkillHub results only when `memory_enabled=True`; the existing `AlgorithmStore` duplicate-source rejection remains unchanged.

**Tech Stack:** Python, pytest, JSON round artifacts, existing `EvolutionLoop`, `AlgorithmAgent`, `AlgorithmStore`, and Fake Wan adapter.

## Global Constraints

- Preserve existing H3, Wan, Student, distillation, scheduler, and evaluator interfaces.
- Do not weaken duplicate algorithm detection or sandbox restrictions.
- `memory_enabled=false` must not expose ExperienceStore evaluation history to the agent.
- Preserve failed formal-run artifacts; never overwrite prior evidence.
- Verify changes with focused tests and the full available CPU test suite before push.

---

### Task 1: Add regression coverage for three deterministic rounds

**Files:**
- Modify: `tests/integration/test_evolutionary_compression.py`
- Test behavior: Fake Wan, `memory_enabled=False`, `adapt_search_policy=False`, `exploit_ratio=1.0`, three consecutive rounds.

**Interfaces:**
- Consumes: existing `EvolutionLoop`, `FakeWanDiTAdapter`, and round `decision.json` artifacts.
- Produces: a regression test proving three non-duplicate deterministic selections can run without `duplicate_algorithm` failures.

- [ ] **Step 1: Write the failing test**

```python
def test_deterministic_baseline_avoids_duplicate_selection_across_three_rounds(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=8, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        memory_enabled=False,
        adapt_search_policy=False,
        search_policy=SearchPolicy(exploit_ratio=1.0, seed=0),
    )

    outcomes = [loop.run_round() for _ in range(3)]

    assert all(outcome.status in {"promoted", "rejected"} for outcome in outcomes)
    assert loop.search_policy.duplicate_rounds == 0
    selected = []
    for outcome in outcomes:
        decision = json.loads(
            (tmp_path / "run" / "rounds" / outcome.round_id / "decision.json").read_text()
        )
        selected.append(decision["operator_args"]["block_ids"][0])
    assert len(set(selected)) == 3
```

- [ ] **Step 2: Run the focused test and verify the current implementation exposes the bug**

Run:

```bash
pytest -q tests/integration/test_evolutionary_compression.py::test_deterministic_baseline_avoids_duplicate_selection_across_three_rounds
```

Expected before the fix: failure on round two with `duplicate_algorithm_source` or a failed round status.

### Task 2: Pass local selection history without opening Memory

**Files:**
- Modify: `harness4h3/evolution/loop.py` near `run_round()` and helper methods.
- Test: `tests/integration/test_evolutionary_compression.py` from Task 1.

**Interfaces:**
- Consumes: `rounds/Rxxxx/decision.json` files in the current `output_root`.
- Produces: private `EvolutionLoop._selection_history() -> list[Mapping[str, Any]]`; each item has `round_id`, `algorithm_id`, `status`, and a nested `decision` with `operator` and `operator_args` only.

- [ ] **Step 1: Implement the local history reader**

```python
def _selection_history(self) -> list[Mapping[str, Any]]:
    history: list[Mapping[str, Any]] = []
    for decision_path in sorted(self.rounds_root.glob("R*/decision.json")):
        try:
            decision = self._read_json(decision_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        history.append(
            {
                "round_id": decision_path.parent.name,
                "algorithm_id": decision.get("algorithm_id"),
                "status": decision.get("status"),
                "decision": {
                    "operator": decision.get("operator"),
                    "operator_args": dict(decision.get("operator_args") or {}),
                },
            }
        )
    return history
```

- [ ] **Step 2: Combine local history with SkillHub history at generation time**

```python
skills = self.skill_hub.search(architecture=architecture, limit=8) if self.memory_enabled else []
selection_history = self._selection_history()
agent_history = [*selection_history, *skills]
```

Pass `agent_history` as `history` to `self.agent.generate(...)` and keep `skills=skills`. This ensures a fixed baseline receives only local selection history, while Memory-enabled runs additionally receive retrieved evaluation evidence.

- [ ] **Step 3: Run the focused regression test**

Run:

```bash
pytest -q tests/integration/test_evolutionary_compression.py::test_deterministic_baseline_avoids_duplicate_selection_across_three_rounds
```

Expected: PASS; all three rounds have distinct `block_ids`, and `duplicate_rounds == 0`.

### Task 3: Verify existing duplicate and restart semantics

**Files:**
- No new production files.
- Test: `tests/integration/test_evolutionary_compression.py`, `tests/unit/test_algorithm_evolution.py`.

- [ ] **Step 1: Run all evolutionary-compression tests**

```bash
pytest -q tests/integration/test_evolutionary_compression.py tests/unit/test_algorithm_evolution.py tests/unit/test_algorithm_sandbox.py
```

Expected: all pass, including the existing test that identical provider source is rejected as `duplicate_algorithm`.

- [ ] **Step 2: Inspect the diff and verify Memory isolation**

```bash
git diff -- harness4h3/evolution/loop.py tests/integration/test_evolutionary_compression.py
```

Confirm the local history contains no `evaluation`, `reward`, or `hardware_metrics` fields and `skills` remains empty when `memory_enabled=False`.

### Task 4: Commit and push the repair

**Files:**
- Commit only: `harness4h3/evolution/loop.py`, `tests/integration/test_evolutionary_compression.py`, and this plan document.

- [ ] **Step 1: Run repository checks available in the local environment**

```bash
git diff --check HEAD -- harness4h3/evolution/loop.py tests/integration/test_evolutionary_compression.py docs/superpowers/plans/2026-10-13-deterministic-algorithm-history.md
```

- [ ] **Step 2: Commit without staging unrelated Student/training changes**

```bash
git add harness4h3/evolution/loop.py tests/integration/test_evolutionary_compression.py docs/superpowers/plans/2026-10-13-deterministic-algorithm-history.md
git commit -m "fix: preserve deterministic algorithm selection history"
git push origin codex/rsi-compression-loop
```

### Task 5: Re-run formal evidence with a consistent revision

**Files:**
- Remote-only configs/output roots; do not overwrite the failed `formal-v10-*` artifacts.

- [ ] **Step 1: Deploy the pushed revision to a new remote source directory**

Use a new source directory and set `HARNESS4H3_SOURCE_REVISION` to the new commit. Keep the existing failed Fixed run as evidence.

- [ ] **Step 2: Start four independent runs with identical manifest, seeds, evaluator recipe, and 16-case budget**

Use new output roots for fixed algorithm, coding agent, agent+memory, and algorithm evolution. Do not reuse the old GPU processes or output roots as comparison evidence.

- [ ] **Step 3: Verify every run reaches three terminal round decisions**

For each root, require `rounds/R0001/decision.json`, `R0002/decision.json`, and `R0003/decision.json` with status in `promoted`, `rejected`, or `failed`; require measured-video evaluation and complete 16-case artifacts for every evaluated child.

- [ ] **Step 4: Run the read-only four-way comparison**

```bash
python tools/compare_evolution_runs.py --json \
  --expected-cases 16 \
  --run fixed-algorithm=<new-fixed-root> \
  --run coding-agent=<new-coding-root> \
  --run agent-memory=<new-memory-root> \
  --run algorithm-evolution=<new-rsi-root>
```

Record whether RSI improves the stated objectives; do not assume it wins.
