# 受限组合压缩算法与经验反馈 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让现有 Wan 演化闭环生成、执行、评测并记忆由 profile 和历史反馈驱动的受限组合压缩算法，同时保持旧单操作算法兼容。

**Architecture:** 在 Sandbox 的单操作计划上增加一个不允许嵌套的 `compose` 计划；Wan registry 顺序执行基础 operator 并只返回最后 checkpoint。新增设计记录模块将 profile、历史反馈、计划 fingerprint 和实际决策写入每轮 artifact、ExperienceStore 和 DesignGeneStore；AlgorithmAgent 的确定性路径据此改变算法结构。

**Tech Stack:** Python 3.12、dataclasses、JSON/JSONL、现有 Wan adapter/operator registry、pytest；不新增运行时依赖，不执行真实 GPU 全量实验。

## Global Constraints

- 只修改压缩演化、Sandbox、Wan operator、经验记录、相关测试和文档；不修改 H3、Student、训练或蒸馏路径。
- `compose` 最多包含 3 个基础步骤，禁止嵌套 compose。
- 任何计划都必须经过现有 Sandbox 和完整性校验；真实晋升仍由 `PromotionPolicy` 决定。
- 缺失/无效评测指标仍保持缺失，不补零；fake 结果不得作为真实 Wan 结论。
- 保留旧 `{"operator": ..., "args": ...}` 算法文件和 CLI 行为。

---

### Task 1: 建立设计记录与反馈摘要

**Files:**
- Create: `harness4h3/evolution/design.py`
- Test: `tests/unit/test_algorithm_design.py`

**Interfaces:**
- `build_design_record(*, algorithm_id, parent_algorithm_id, search_choice, profile, history, plan, hypothesis) -> dict[str, Any]`
- `design_fingerprint(record_without_fingerprint) -> str`
- `feedback_summary(history) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing tests**

Add tests asserting that the same inputs produce the same fingerprint, profile evidence fields are listed, quality failures and efficiency failures are summarized separately, and a changed plan changes the fingerprint.

- [ ] **Step 2: Run the focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_algorithm_design.py`

Expected: collection or assertion failure because the design module does not exist.

- [ ] **Step 3: Implement the design record**

Use canonical JSON and `stable_digest`. Return a JSON-safe mapping with `schema_version`, `algorithm_id`, `parent_algorithm_id`, `search_choice`, `bottleneck`, `evidence_refs`, `feedback`, `hypothesis`, `plan`, and `fingerprint`. Derive bottleneck from available layer latency/FLOPs and record only evidence keys that are actually present.

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_algorithm_design.py`

Expected: all design-record tests pass.

### Task 2: Add bounded composite Sandbox plans

**Files:**
- Modify: `harness4h3/sandbox/runner.py`
- Test: `tests/unit/test_algorithm_sandbox.py`

**Interfaces:**
- Extend `CompressionPlan.from_mapping()` to accept `{"operator": "compose", "args": {"steps": [...]}}`.
- Preserve `.operator`, `.args`, `.to_dict()` for existing single-operation callers.

- [ ] **Step 1: Add failing tests**

Cover a valid two-step plan, rejection of nested compose, rejection of four steps, rejection of unsafe step keys, and unchanged acceptance of a legacy single operation.

- [ ] **Step 2: Run the focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_algorithm_sandbox.py`

Expected: new composite tests fail while legacy tests pass.

- [ ] **Step 3: Implement validation**

Factor base-step validation into a private helper. Validate every step using the same exact argument rules as the existing operators, allow 2–3 steps, reject nested compose and unknown keys, and normalize the persisted representation to JSON lists/dicts.

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_algorithm_sandbox.py`

Expected: all Sandbox tests pass.

### Task 3: Execute composite Wan compression atomically

**Files:**
- Modify: `harness4h3/operators/wan/compression.py`
- Modify: `harness4h3/operators/wan/__init__.py`
- Test: `tests/unit/test_wan_compression.py`

**Interfaces:**
- Add `CompositeCompressionOperator.name == "compose"`.
- `build_wan_compression_registry()` exposes `prune_blocks`, `prune_ffn_channels`, and `compose`.

- [ ] **Step 1: Write failing tests**

Use `FakeWanDiTAdapter` to execute a two-step plan and assert that `step-00` and `step-01` artifacts exist, the final child differs from the parent, and `structural_change.steps` preserves both results. Add a failure test proving a failed later step returns `status="failed"` and does not create a promoted child artifact.

- [ ] **Step 2: Run the focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_wan_compression.py`

Expected: composite operator import or execution failures.

- [ ] **Step 3: Implement sequential execution**

Execute each base operator into `output_dir / "step-%02d"`, carry the successful child artifact into the next step, accumulate structural changes, and return only the final artifact on success. On failure, preserve completed step metadata but return no candidate child. Reject nested/unknown/empty steps at the operator boundary too.

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_wan_compression.py`

Expected: all composite and legacy operator tests pass.

### Task 4: Make the deterministic and provider agents design algorithms

**Files:**
- Modify: `harness4h3/evolution/agent.py`
- Test: `tests/unit/test_algorithm_evolution.py`

**Interfaces:**
- Keep `AlgorithmAgent.generate()` signature unchanged.
- Deterministic source may return either a legacy single plan or a bounded `compose` plan.

- [ ] **Step 1: Add failing tests**

Assert that an exploration profile with layer latency and channel data generates a composite plan, that a quality-floor failure history generates a conservative non-composite plan, that an efficiency miss increases the exploration retention adjustment, and that the provider prompt documents `compose` without permitting unsafe fields.

- [ ] **Step 2: Run the focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_algorithm_evolution.py`

Expected: new behavior tests fail.

- [ ] **Step 3: Implement profile- and feedback-driven generation**

Rank candidate Blocks by low output sensitivity plus available latency/FLOPs evidence, choose a second FFN target only when profile data supports it, and build a deterministic bounded source. Detect quality-floor/quality-metric failures in history and disable aggressive composition for the next candidate. Update the structured-provider prompt and compact schema description to allow `compose` with its exact step structure.

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/pytest -q tests/unit/test_algorithm_evolution.py tests/unit/test_algorithm_sandbox.py`

Expected: all agent and Sandbox tests pass.

### Task 5: Persist design evidence and retrieve DesignGenes

**Files:**
- Modify: `harness4h3/evolution/loop.py`
- Modify: `harness4h3/evolution/skill_hub.py`
- Test: `tests/integration/test_evolutionary_compression.py`
- Test: `tests/unit/test_algorithm_evolution.py`

**Interfaces:**
- Each successful Sandbox round writes `rounds/Rxxxx/design.json`.
- `ExperienceRecord.training` gains `design_fingerprint`, `design_strategy`, `design_feedback`, and `design`.
- `SkillHub.search()` returns both `memory_kind="experience"` and `memory_kind="design_gene"` entries.

- [ ] **Step 1: Add failing integration tests**

Run two fake rounds with an evaluator that first rejects for quality and then accepts. Assert that both `design.json` and `design-gene.jsonl` exist, the second plan fingerprint differs from the first, rejected Child is not Active, and the retrieved memory explicitly identifies the rejected design.

- [ ] **Step 2: Run the focused integration test**

Run: `.venv/bin/pytest -q tests/integration/test_evolutionary_compression.py -k design`

Expected: failure because no design artifact/gene is persisted.

- [ ] **Step 3: Implement loop persistence**

After Sandbox success, build the design record from profile, round history, retrieved skills, plan, and hypothesis; atomically write `design.json`; include the record in decision and ExperienceRecord training. After the measured decision, append a `DesignGene` whose benefit/limitation fields come only from the actual evaluation and promotion reasons. Preserve resume behavior by loading an existing design artifact before recomputing it.

- [ ] **Step 4: Extend SkillHub retrieval**

Read genes alongside experiences, filter by architecture and optional operator family, rank accepted measured genes separately from rejected/partial genes, and mark each result with its memory kind and status. Do not drop rejected evidence.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/pytest -q tests/integration/test_evolutionary_compression.py tests/unit/test_algorithm_evolution.py`

Expected: all focused tests pass.

### Task 6: Document and verify the recursive-design contract

**Files:**
- Modify: `docs/evolutionary-compression.md`
- Modify: `docs/experiment-schema.md`
- Test: `tests/unit/test_evolution_promotion.py`

- [ ] **Step 1: Document the new artifact contract**

Describe `design.json`, composite step artifacts, DesignGene fields, and the distinction between mechanism validation and real Wan evidence. State that fake CPU runs cannot establish quality or hardware gains.

- [ ] **Step 2: Add promotion regression coverage**

Keep tests proving a composite candidate still passes through the unchanged quality, hardware, missing-metric, and Pareto gates; no test may promote a candidate solely because its design fingerprint is novel.

- [ ] **Step 3: Run the complete relevant verification**

Run: `.venv/bin/pytest -q tests/unit/test_algorithm_design.py tests/unit/test_algorithm_sandbox.py tests/unit/test_algorithm_evolution.py tests/unit/test_wan_compression.py tests/unit/test_evolution_promotion.py tests/integration/test_evolutionary_compression.py`

Then run: `.venv/bin/python -m compileall -q harness4h3/evolution harness4h3/sandbox harness4h3/operators/wan` and `git diff --check`.

Expected: all focused tests pass, compilation succeeds, and diff check has no output.

- [ ] **Step 4: Run CPU full suite and inspect scope**

Run: `.venv/bin/pytest -q`

Record the actual result and CUDA-dependent skips. Do not run the four full Wan experiments or claim real RSI improvement from CPU/fake evidence.

- [ ] **Step 5: Commit only task files**

Stage only the files listed in Tasks 1–6 and commit with:

```bash
git commit -m "feat: evolve compression algorithms from measured feedback"
```
