# Wan-Native Controller Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing Controller loop to the verified native 5s Wan 32→16 trajectory contract and run one auditable bounded recipe-search round.

**Architecture:** Extend the existing trusted Wan proposal/command boundary instead of creating a second campaign framework. A native campaign config fixes the H3 cache, immutable trajectory record, 32→16 curriculum, multi-prompt/multi-seed F3 manifest, visual veto, and frozen R2 gate; the Controller searches only loss/optimization fields inside that contract.

**Tech Stack:** Python, dataclasses, JSON Schema-style validation, YAML, pytest, remote L40 execution, vLLM JSON proposals.

## Global Constraints

- Real execution only: `offline_simulation=false`.
- Native 5s cache only: `cache_contract=native_5s`.
- Fixed real Wan trajectory: source NFE 32, target NFE 16, immutable seed-20261010 record.
- F3 uses four prompts and three seeds with the visual-structure-v1 veto.
- Legacy R2 promotion remains frozen.
- Controller cannot search architecture, 30s continuation, DMD2, or code paths.

---

### Task 1: Extend the trusted native recipe contract

**Files:**
- Modify: `harness4h3/student/wan_campaign.py`
- Modify: `tools/wan_vllm_campaign.py`
- Test: `tests/unit/test_wan_campaign.py`
- Test: `tests/unit/test_wan_vllm_campaign.py`

**Interfaces:**
- `WanTrainingPolicy.endpoint_weight` is parsed, serialized, fingerprinted, and passed to the worker.
- `build_training_command()` emits `--endpoint-weight` and native rollout-state arguments when the config binds them.
- `_validate_recipe_search_space()` accepts `endpoint_weight` and rejects values outside the configured list.

- [ ] Add `endpoint_weight` to the proposal JSON schema's required training fields and numeric properties.
- [ ] Add `endpoint_weight` to the trusted search-space value map and command builder.
- [ ] Add native config fields `training_mode`, `teacher_forced_intervals`, and `trajectory_record` with strict validation; emit `--mode native_rollout_state`, `--teacher-forced-intervals 16`, and `--trajectory-record <path>` when configured.
- [ ] Add tests asserting a rollout-only proposal with endpoint/flow/clean weights set to zero is accepted and that the generated command contains every native binding.
- [ ] Run `Harness4H3/.venv/bin/python -m pytest tests/unit/test_wan_campaign.py tests/unit/test_wan_vllm_campaign.py -q` and require all tests to pass.
- [ ] Commit only the contract and focused tests with message `feat: bind controller search to native Wan rollout state`.

### Task 2: Add an isolated native Controller campaign configuration

**Files:**
- Create: `configs/wan-vllm-native-5s-controller.yaml`
- Create: `configs/wan-native-controller-training.yaml`
- Test: `tests/unit/test_wan_vllm_campaign.py`

**Interfaces:**
- The campaign config points at the successful native parent checkpoint and fixed seed-20261010 trajectory record.
- The training config binds the native 5s cache and the real Wan 32→16 worker contract.

- [ ] Set the campaign manifest to `configs/evaluation-manifest.wan-5s-native-paper.json`, `max_cases=4`, `seed_count=3`, `inference_steps=16`, and `continuation_contract=5s`.
- [ ] Set `correctness_audit_passed=true`, bind the existing correctness evidence digest, set `legacy_r2_promotion_frozen=true`, and enable `visual_quality_gate` with the existing thresholds.
- [ ] Set the initial parent to the successful seed10 continuation checkpoint and create a new isolated output root under `/data/models/MiniMax-H3/harness4h3/wan-native-controller-search-20261011`.
- [ ] Restrict the search space to `endpoint_weight`, `velocity_weight`, `clean_anchor_weight`, `rollout_endpoint_weight`, `rollout_clean_weight`, `rollout_steps`, `rollout_every`, `learning_rate`, `train_steps`, and `teacher_reuse_steps`; exclude DMD2 and architecture fields.
- [ ] Bind `training_mode=native_rollout_state`, `teacher_forced_intervals=16`, `trajectory_record` to the immutable seed10 record, and `source_steps=32`, `target_steps=16`.
- [ ] Add a config-load test that rejects a native campaign if the trajectory path or native cache contract is missing.

### Task 3: Run and verify one bounded Controller round

**Files:**
- Create remotely: `/data/models/MiniMax-H3/harness4h3/wan-native-controller-search-20261011/`
- Preserve: F1/F2/F3 proposal, training, evaluation, decision, events, experience, and failure-memory artifacts.

**Interfaces:**
- The real vLLM Controller returns one schema-valid recipe from the fixed native search space.
- The worker and evaluator produce independent F1→F2→F3 evidence.

- [ ] Sync the focused code/config changes to the remote checkout without touching unrelated worktree edits.
- [ ] Start the controller launcher only for this isolated native campaign and run `max_rounds=1`.
- [ ] Poll the authoritative campaign state and child process; do not restart on observation timeouts.
- [ ] Verify the proposal digest, parent digest, native trajectory digest, F3 case count, visual failure count, quality/latency/VRAM metrics, and promotion decision.
- [ ] If the round fails, retain the failure memory and do not promote the child; if it succeeds, retain the accepted checkpoint and append its experience record.

### Task 4: Paper-level comparison record

**Files:**
- Create remotely: `native-comparison-summary.json`
- Modify only if needed: `docs/long-video-30s-acceptance-report.md`

**Interfaces:**
- Summary compares Base Wan32, Base Wan16, frozen R2 16, verified native student 16, and any accepted Controller child on quality, visual pass rate, pure latency, NFE, peak VRAM, and training cost.

- [ ] Generate the summary from authoritative JSON artifacts rather than hand-entered values.
- [ ] Confirm the summary records the Controller-off prerequisite result and the subsequent bounded search result separately.
- [ ] Run the final focused test suite and report only the completed, failed, or decision-required outcome.
