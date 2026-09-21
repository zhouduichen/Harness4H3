# Student Main Execution Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Student control plane the default, evidence-producing execution path for CLI and detached campaigns.

**Architecture:** Keep `StudentCampaign` as the orchestrator and wire its existing control-plane branch from one shared builder used by the CLI and detached supervisor. Extend the existing Student worker/remote protocol with parent and fidelity context, dispatch existing `h3_training` algorithms, and persist the complete trace through the current JSONL stores.

**Tech Stack:** Python 3.9+, existing frozen campaign contracts, PyTorch, existing `h3_training` algorithms and `TrainerEngine`, PyYAML, pytest, no new runtime dependency.

## Global Constraints

- Production Student runs fail closed and never fall back to the legacy single-proposal path.
- The H3 teacher remains the trusted target source; accepted Student checkpoints are inherited as the next parent.
- Candidate counts stay within the existing 3–5 control-plane contract.
- Existing `CampaignBase`, `CapabilitySnapshot`, `ReviewPipeline`, `AcceptanceGate`, `pareto_dominates`, archive, and experience implementations are reused.
- No new campaign engine, database, UI, or arbitrary LLM-generated code/commands.
- Structural proxy quality is not accepted as production semantic verification.

---

### Task 1: Build the shared default Student control-plane wiring

**Files:**
- Modify: `Harness4H3/harness4h3/student/campaign.py`
- Modify: `Harness4H3/harness4h3/harness4h3/student/config.py`
- Modify: `Harness4H3/harness4h3/harness4h3/cli.py`
- Modify: `Harness4H3/tools/student_campaign_supervisor.py`
- Modify: `Harness4H3/configs/student-campaign.example.yaml`
- Test: `Harness4H3/tests/integration/test_student_main_execution_chain.py`

**Interfaces:**
- Add one shared builder returning `CampaignBase`, `CapabilitySnapshot`, and `ReviewPipeline` for a `StudentCampaignConfig` and provider.
- Add `StudentCampaign(..., initial_parent_checkpoint=...)` and persist/load the parent path and hash through `parent.selected`.
- Add `StudentCampaignConfig.fidelity_schedule: tuple[str, ...]` with default `("F1", "F2", "F3")`.

- [ ] Add the immutable base builder using canonical hashes for the target, verifier bank, dataset/cache, evaluation recipe, provider identity, and supported Student algorithms.
- [ ] Add trusted deterministic Advocate, Critical, and Modifier agents with distinct `ActorIdentity` values and route them through the existing `ReviewPipeline`.
- [ ] Make CLI and detached supervisor call the builder and pass all control-plane arguments, initial teacher parent, fidelity schedule, and semantic quality policy.
- [ ] Reject production campaign construction when `quality_backend != "clip_temporal"` or no evaluation manifest is configured.
- [ ] Add an integration assertion that default construction has a base/snapshot/review pipeline and that a run writes `campaign.created` and `critic.completed` rather than legacy events.

### Task 2: Add parent checkpoint and multi-fidelity protocol

**Files:**
- Modify: `Harness4H3/harness4h3/student/remote.py`
- Modify: `Harness4H3/harness4h3/student/campaign.py`
- Modify: `Harness4H3/tools/student_train_worker.py`
- Modify: `Harness4H3/harness4h3/student/worker.py`
- Test: `Harness4H3/tests/integration/test_student_main_execution_chain.py`
- Test: `Harness4H3/tests/training/test_student_worker.py`

**Interfaces:**
- Extend the worker call with optional `parent_checkpoint`, `parent_candidate_id`, and `fidelity`; old injected test doubles remain callable without these keywords.
- Add `TrainingResult` evidence fields for `parent_kind`, `parent_checkpoint`, `parent_inherited`, `inherited_parameter_count`, `algorithm_name`, `algorithm_dispatch`, and `fidelity`.

- [ ] Pass the trusted H3 teacher separately from the optional Student parent in the remote command.
- [ ] Load matching parent Student tensors into the new Student graph, fail closed when no compatible tensor can be inherited, and record parent/child hashes.
- [ ] Derive deterministic step budgets from F1/F2/F3 and make each higher fidelity inherit the previous fidelity child.
- [ ] Persist the selected child checkpoint and hash in `parent.selected`; on resume, reject a missing parent instead of resetting to the teacher.
- [ ] Test two fidelities and two rounds with a recording worker and assert the exact parent path/hash sequence.

### Task 3: Dispatch existing real training algorithms

**Files:**
- Modify: `Harness4H3/harness4h3/student/worker.py`
- Test: `Harness4H3/tests/training/test_student_worker.py`

**Interfaces:**
- Add a Student-shaped adapter in the existing worker boundary implementing the existing `DenoisingModelAdapter` contract.
- Map `velocity_distill` to `ProgressiveDistillation` stages and `dmd2` to `DMD2`, both executed by the existing `TrainerEngine`.

- [ ] Convert `StudentBatch` into `PreparedBatch` and expose the real H3 target signal through the adapter-owned teacher role.
- [ ] Build the selected existing algorithm from the proposal method and learning rates; reject unknown methods before training.
- [ ] Run the algorithm through `TrainerEngine`, save the changed child with existing safe checkpoint logic, and include the algorithm path and optimizer evidence in `TrainingResult`.
- [ ] Add tests that exercise both dispatch branches and fail if the result reports the removed generic MSE path.

### Task 4: Strengthen semantic verifier, hard/Pareto gate, archive, and experience evidence

**Files:**
- Modify: `Harness4H3/harness4h3/campaign/adapters.py`
- Modify: `Harness4H3/harness4h3/student/campaign.py`
- Modify: `Harness4H3/harness4h3/student/evaluator.py`
- Modify: `Harness4H3/harness4h3/student/remote.py`
- Test: `Harness4H3/tests/integration/test_student_main_execution_chain.py`
- Test: `Harness4H3/tests/unit/test_campaign_gates.py`

**Interfaces:**
- Map semantic score/semantic verifier status, video decodability, algorithm dispatch, parent binding, fidelity, hardware metrics, and model size to `MetricEvidence`.
- Keep hard gate results distinct from Pareto dominance; archive every candidate and append actual/predicted experience.

- [ ] Require `semantic_verified=True` in production hard constraints and reject structural-only evaluation.
- [ ] Compute Pareto dominance over feasible candidates using existing `pareto_dominates`, then select only non-dominated candidates.
- [ ] Include checkpoint hashes, semantic metrics, fidelity history, gate result, and algorithm dispatch in archive and experience records.
- [ ] Assert rejected/failed candidates remain archived as evidence while only accepted parents are retained as active checkpoints.

### Task 5: Verify the complete chain and audit evidence

**Files:**
- Modify: `Harness4H3/README.md`
- Test: `Harness4H3/tests/integration/test_student_main_execution_chain.py`

- [ ] Run focused worker/control-plane tests.
- [ ] Run the complete nested Harness4H3 test suite.
- [ ] Run a CPU scripted evidence campaign and inspect its trace, archive, experience, result, and parent hashes field-by-field.
- [ ] Search the default CLI/supervisor construction for any remaining implicit legacy Student path.
- [ ] Record the exact verification commands and evidence locations in the final handoff.

