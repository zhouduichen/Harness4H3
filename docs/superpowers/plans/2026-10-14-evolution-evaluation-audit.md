# Evolution Evaluation Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复进化压缩实验统计和后续模型晋升协议，明确 Wan 指标口径，并用回归测试证明被拒绝候选不会污染最终结果。

**Architecture:** 保持现有 `EvolutionLoop`、`ModelStore` 和 Wan evaluator 边界不变。只让只读汇总器建立显式的 Active/候选状态视图；让 `PromotionPolicy` 在现有 `decide` 接口内执行硬约束、任一目标达标和 Pareto 检查；让 evaluator 在现有硬件指标记录中增加明确的 scope/availability 字段。

**Tech Stack:** Python 3.12、pytest、标准库 `pathlib/json/statistics`、现有 Torch/Diffusers 可选适配器。

## Global Constraints

- 不修改已有历史 `decision.json` 或用新协议追认旧实验。
- 不修改 Student、H3 training、蒸馏和 remote campaign 代码。
- 缺失指标使用 `null`/明确 unavailable 语义，不能默认填零或当作改善。
- 只提交本任务涉及的文件，保留工作区既有未提交修改。
- 不执行四组完整 GPU 实验；如需 GPU，只做低成本 smoke 检查。

---

### Task 1: Make run summaries active-model authoritative

**Files:**
- Modify: `tools/compare_evolution_runs.py`
- Test: `tests/unit/test_compare_evolution_runs.py`

**Interfaces:**
- Consumes: `models/active.json`, `models/candidates/M*.json`, `rounds/R*/decision.json` and `evaluation.json`.
- Produces: summary keys `active_model_id`, `last_candidate_model_id`, `last_candidate_status`, `historical_best_candidate_model_id`, `rejected_model_ids`, `final_source`, and null-valued `final` metrics when no reliable Active evidence exists.

- [ ] **Step 1: Write failing tests**

Add fixtures for an `active.json` containing only `{"id": "M0001"}`, a promoted Active candidate, a root Active with only provenance, a rejected final Child, multiple rejected rounds, and a reliable `baseline_*` record tied to `parent_model_id`.

```python
def test_summary_reads_id_and_never_uses_rejected_last_child_as_final(tmp_path):
    root = _make_run_with_active_id(tmp_path, active_id="M0001", active_metrics={"quality_score": 0.98})
    _write_round(root, "R0001", parent="M0001", child="M0002", status="rejected", quality=0.70)
    result = summarize_run("run", root)
    item = result
    assert item["active_model_id"] == "M0001"
    assert item["final_source"] == "active_model"
    assert item["final"]["quality_score"] == 0.98
    assert item["last_candidate_model_id"] == "M0002"
    assert item["rejected_model_ids"] == ["M0002"]


def test_summary_marks_missing_active_metrics_instead_of_falling_back_to_child(tmp_path):
    root = _make_run_with_active_id(tmp_path, active_id="M0000", active_metrics={})
    _write_round(root, "R0001", parent="M0000", child="M0001", status="rejected", quality=0.91)
    result = summarize_run("run", root)
    assert result["final_source"] == "missing"
    assert result["final"]["quality_score"] is None
    assert result["last_candidate_model_id"] == "M0001"


def test_summary_recovers_only_parent_baseline_for_active_model(tmp_path):
    root = _make_run_with_active_id(tmp_path, active_id="M0000", active_metrics={})
    _write_round(root, "R0001", parent="M0000", child="M0001", status="rejected", quality=0.91,
                 baseline_quality=1.0, baseline_hardware={"inference_wall_time_s_mean": 10.0})
    result = summarize_run("run", root)
    assert result["final_source"] == "associated_parent_baseline"
    assert result["final"]["quality_score"] == 1.0
    assert result["final"]["latency_s"] == 10.0
```

- [ ] **Step 2: Run the focused tests and verify the old fallback fails**

Run: `.venv/bin/pytest -q tests/unit/test_compare_evolution_runs.py`

Expected: the new tests fail because `id` is not read and the current implementation uses `evaluations[-1]` as `final`.

- [ ] **Step 3: Implement an explicit run-state resolver**

Add small helpers that read `id` first, validate the corresponding candidate, extract measured metrics, recover only `baseline_*` metrics from a round whose `parent_model_id` equals the Active ID, and collect candidate metadata independently from final metrics. Never assign a rejected or merely last evaluation to `final`.

Use a deterministic historical-best rule (`search_score`, then `quality_score`, then lower latency) and expose its ID/status without using it for `final`.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_compare_evolution_runs.py`

Expected: PASS, with missing Active metrics remaining `None` and the final source identifying the provenance.

- [ ] **Step 5: Commit the isolated statistics change**

```bash
git add tools/compare_evolution_runs.py tests/unit/test_compare_evolution_runs.py
git commit -m "fix: keep active and rejected evolution results separate"
```

### Task 2: Replace AND promotion thresholds with configurable multi-objective gates

**Files:**
- Modify: `harness4h3/evolution/evaluator.py`
- Test: `tests/unit/test_evolution_promotion.py`

**Interfaces:**
- Consumes: `PromotionPolicy.from_mapping()` and `PromotionPolicy.decide()` inputs already used by `EvolutionLoop`.
- Produces: auditable decision fields for per-objective thresholds, improvement map, missing metrics, quality metric failures, hardware regressions, and Pareto rejection.

- [ ] **Step 1: Write failing tests**

Update the old AND expectation and add tests for each independent objective and hard gate.

```python
def test_latency_only_gain_can_promote():
    decision = PromotionPolicy().decide(parent={}, evaluation=_evidence(latency=94.0, memory=10.0, size=9.5), child_integrity=True)
    assert decision["accepted"] is True


def test_memory_only_gain_can_promote():
    decision = PromotionPolicy().decide(parent={}, evaluation=_evidence(latency=100.0, memory=9.0, size=9.5), child_integrity=True)
    assert decision["accepted"] is True


def test_model_size_only_gain_can_promote():
    decision = PromotionPolicy().decide(parent={}, evaluation=_evidence(latency=100.0, memory=10.0, size=8.9), child_integrity=True)
    assert decision["accepted"] is True


def test_missing_objective_does_not_count_as_improvement_but_other_gain_remains_valid():
    evidence = _evidence(latency=90.0, memory=9.0, size=9.5)
    del evidence["hardware_metrics"]["model_size_gb"]
    decision = PromotionPolicy().decide(parent={}, evaluation=evidence, child_integrity=True)
    assert decision["accepted"] is True
    assert "model_size_gb" in decision["missing_metrics"]


def test_zero_peak_memory_is_unavailable_not_a_hundred_percent_gain():
    decision = PromotionPolicy().decide(parent={}, evaluation=_evidence(memory=0.0, latency=100.0, size=9.5), child_integrity=True)
    assert decision["accepted"] is False
    assert "peak_memory_gb" in decision["missing_metrics"]
```

Also cover semantic/temporal quality failure, non-target hardware regression, no valid target improvement, Pareto dominance, multiple simultaneous gains, and `from_mapping` with separate latency/memory thresholds.

- [ ] **Step 2: Run focused tests and verify old policy failures**

Run: `.venv/bin/pytest -q tests/unit/test_evolution_promotion.py`

Expected: the new single-objective tests fail under the current AND policy.

- [ ] **Step 3: Implement the policy**

Add `min_latency_improvement` and `min_memory_improvement` fields with defaults `0.05`; retain `min_efficiency_improvement` as a compatibility input in `from_mapping`. Use positive-only hardware measurements, check available quality submetrics against the same floor, reject regressions beyond tolerance, and accept the objective gate when any one of the three configured improvements meets its threshold. Keep `_dominates` and all reason strings auditable.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_evolution_promotion.py`

Expected: PASS.

- [ ] **Step 5: Commit the policy change**

```bash
git add harness4h3/evolution/evaluator.py tests/unit/test_evolution_promotion.py
git commit -m "fix: make compression promotion multi-objective"
```

### Task 3: Make Wan metric scopes and unavailable telemetry explicit

**Files:**
- Modify: `harness4h3/evolution/evaluator.py`
- Test: `tests/unit/test_wan_video_evaluator.py`

**Interfaces:**
- Consumes: existing checkpoint artifacts and evaluator config/device placement.
- Produces: `transformer_parameter_count`, `parameter_count_scope`, `transformer_model_size_gb`, `pipeline_model_size_gb`, `model_size_gb`, `peak_memory_available`, and nullable peak memory values.

- [ ] **Step 1: Write failing tests**

Add a file-size test with hard links and symlinks, an evaluator result test asserting the new scope fields, and a fake CUDA test where memory telemetry is unavailable and the result is `None`, not `0.0`.

```python
def test_checkpoint_size_deduplicates_hardlinks_and_symlinks(tmp_path):
    root = tmp_path / "pipeline"
    root.mkdir()
    source = root / "weights.bin"
    source.write_bytes(b"1234")
    (root / "hardlink.bin").hardlink_to(source)
    (root / "symlink.bin").symlink_to(source)
    assert WanVideoEvaluator._checkpoint_size_gb(str(root)) == pytest.approx(4 / 1024**3)


def test_unavailable_peak_memory_is_none():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False
    evaluator = object.__new__(WanVideoEvaluator)
    evaluator.config = SimpleNamespace(device="cuda:7")
    assert evaluator._peak_memory_gb(Torch) is None
```

- [ ] **Step 2: Run focused evaluator tests and verify failure**

Run: `.venv/bin/pytest -q tests/unit/test_wan_video_evaluator.py`

Expected: the deduplication and nullable telemetry tests fail under the current implementation.

- [ ] **Step 3: Implement metric accounting**

Count unique `(st_dev, st_ino)` entries while walking checkpoint files. Compute target Transformer size from `transformer/` when present and full pipeline size from the artifact root. Preserve `model_size_gb` as the complete-pipeline compatibility field, add explicit scope fields, return `None` for invalid/unavailable CUDA telemetry, and aggregate peak memory as `None` if any required case has no valid measurement.

- [ ] **Step 4: Run evaluator tests**

Run: `.venv/bin/pytest -q tests/unit/test_wan_video_evaluator.py`

Expected: PASS.

- [ ] **Step 5: Commit metric changes**

```bash
git add harness4h3/evolution/evaluator.py tests/unit/test_wan_video_evaluator.py
git commit -m "fix: make Wan evaluation metric scopes explicit"
```

### Task 4: Update the evaluation recipe and audit record

**Files:**
- Modify: `docs/evolutionary-compression.md`
- Modify: `docs/experiment-schema.md`
- Create: `docs/superpowers/reports/2026-10-14-evolution-evaluation-audit.md`

**Interfaces:**
- Consumes: the implemented summary fields, policy reasons, metric scopes, and repository evidence inventory.
- Produces: a reproducible recipe and a Chinese audit report that distinguishes code-confirmed facts from unavailable experiment evidence.

- [ ] **Step 1: Document the new protocol**

Document the Active-only final rule, associated baseline recovery, candidate status fields, 5%/5%/10% OR gate, 98% quality floor, quality submetric checks, regression tolerance, zero/unavailable telemetry semantics, deduplicated size scopes, actual CUDA device, and the fact that old decisions are not recomputed.

- [ ] **Step 2: Write the audit report**

Record the discovered code defects, the available local evidence, the absence of complete four-group Wan RSI logs in this checkout, the limited conclusion about Memory versus RSI, and the smallest follow-up experiment needed to test experience accumulation.

- [ ] **Step 3: Check documentation consistency**

Run: `rg -n "last_candidate|historical_best|model_size_gb|peak_memory|PromotionPolicy|quality_floor|OR|active" docs/evolutionary-compression.md docs/experiment-schema.md docs/superpowers/reports/2026-10-14-evolution-evaluation-audit.md`

Expected: every new machine-readable field and gate definition has one clear description and no claim that old experiments were promoted retroactively.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/evolutionary-compression.md docs/experiment-schema.md docs/superpowers/reports/2026-10-14-evolution-evaluation-audit.md
git commit -m "docs: define trustworthy evolution evaluation protocol"
```

### Task 5: Run repository verification and publish the branch

**Files:**
- No source changes expected; inspect only the files staged by Tasks 1–4.

- [ ] **Step 1: Run focused regression tests**

Run: `.venv/bin/pytest -q tests/unit/test_compare_evolution_runs.py tests/unit/test_evolution_promotion.py tests/unit/test_wan_video_evaluator.py`

Expected: PASS.

- [ ] **Step 2: Run CPU full suite**

Run: `.venv/bin/pytest -q`

Expected: all collected CPU tests pass, with any skips reported from this execution rather than copied from an old report.

- [ ] **Step 3: Run compilation and CI-contract checks**

Run: `.venv/bin/python -m compileall -q harness4h3 tools tests` and inspect `.github/workflows/cpu-ci.yml` for the actual test command and dependency setup.

Expected: compileall exits zero and the documented CI command remains compatible.

- [ ] **Step 4: Review the final diff and preserve unrelated work**

Run: `git status --short`, `git diff --check`, and `git diff --cached --name-only` after staging only task files. Confirm no Student/H3-training changes were staged.

- [ ] **Step 5: Commit and push only task files**

If the requested branch is not checked out, create or switch to `codex/rsi-compression-loop` without resetting the worktree. Push with:

```bash
git push -u origin codex/rsi-compression-loop
```

Report the actual test counts, commit IDs, push result, and any unrelated dirty files left untouched.
