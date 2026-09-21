# Student Execution Truth Implementation Plan

> For agentic workers: This plan is executed inline in the current task. Steps use checkbox syntax for tracking.

Goal: Make every Student Campaign Candidate, Teacher call, GPU assignment, edge measurement, lineage record, and process status correspond to a real executable path.

Architecture: Keep the existing Campaign control plane as the system of record. Add an online multi-GPU H3 Teacher service, a trusted RevisionPatch canonicalization boundary, and a target-device evidence adapter; thread their digests and statuses through existing events and gates.

Tech Stack: Python 3.12, PyTorch, torch.multiprocessing, local RPC, dataclasses, JSON/SHA-256 digests, pytest, existing h3_training adapters and algorithms.

## Global Constraints

- Real H3 Teacher prediction runs online for the current noisy sample/timestep; fixed targets remain offline-test-only.
- Teacher and Student GPU assignments are distinct and satisfy separate free-memory thresholds before model load.
- Revision returns only a closed RevisionPatch; trusted code rebuilds and validates the executable Candidate.
- Server evidence can produce PROMOTABLE but can never satisfy an edge metric.
- Only INFRA_FAILURE, INTEGRITY_FAILURE, and UNHANDLED_EXCEPTION produce non-zero exits.
- Progressive Distillation uses binary-halving stages and the previous child as the next Teacher.

---

### Task 1: Enforce role-aware GPU allocation

Files:
- Modify: harness4h3/student/gpu.py
- Modify: harness4h3/student/config.py
- Modify: tools/student_train_worker.py
- Modify: harness4h3/student/remote.py
- Test: tests/unit/test_student_gpu.py
- Test: tests/integration/test_student_gpu_architecture.py

Interfaces:
- Add GPUAllocation(teacher_devices, student_device, worker_min_free_memory_gb, teacher_rank_min_free_memory_gb, student_min_free_memory_gb).
- Add select_role_gpu_allocation(...) -> GPUAllocation; reject overlap, missing ranks, and under-provisioned devices.
- Pass explicit role devices and all three thresholds to the worker.

- [ ] Step 1: Write tests where mocked nvidia-smi data assigns Teacher cuda:0/cuda:1 and Student cuda:2, and rejects role overlap.
- [ ] Step 2: Run .venv/bin/pytest -q tests/unit/test_student_gpu.py tests/integration/test_student_gpu_architecture.py; observe failure.
- [ ] Step 3: Factor nvidia-smi parsing into _query_free_memory, select Teacher ranks first, then one distinct Student GPU, and retain the old selector as a compatibility wrapper.
- [ ] Step 4: Add config validation and propagate worker_min_free_memory_gb, teacher_rank_min_free_memory_gb, and student_min_free_memory_gb.
- [ ] Step 5: Run the focused tests; expect PASS.
- [ ] Step 6: Commit with git commit -am "feat: enforce role-aware Student GPU allocation" plus the new test files.

### Task 2: Implement the online multi-GPU H3 Teacher service

Files:
- Create: harness4h3/student/teacher_service.py
- Modify: harness4h3/student/worker.py
- Modify: tools/student_train_worker.py
- Test: tests/training/test_teacher_service.py
- Test: tests/integration/test_student_teacher_service.py

Interfaces:
- Add TeacherRequest, TeacherResponse, TeacherService.start(), TeacherServiceHandle.predict(), and TeacherServiceHandle.close().
- RealH3TeacherBackend.load_teacher returns a service-backed Teacher role; teacher_targets_dir remains rejected.

- [ ] Step 1: Write a CPU fake service test sending multiple current noisy samples and asserting every rank is used and responses carry online_forward=True.
- [ ] Step 2: Run the focused tests and observe failure.
- [ ] Step 3: Implement a length-prefixed local request/response protocol. Serialize tensor metadata and CPU bytes; reconstruct on each rank model device; call real adapter forward; return CPU tensors with rank/world-size metadata. Fail closed on malformed responses or dead ranks.
- [ ] Step 4: Add teacher_predictor to StudentAlgorithmAdapter. For Teacher roles, convert inputs to the service device and outputs back to Student device. Never read StudentBatch.target in the real backend.
- [ ] Step 5: Add a parameter-free TeacherRoleProxy compatible with existing algorithm preparation and launch one real H3 service rank per allocated Teacher GPU. Persist role devices, rank usage, and online_forward=true in TrainingResult and checkpoint metadata.
- [ ] Step 6: Run .venv/bin/pytest -q tests/training/test_teacher_service.py tests/integration/test_student_teacher_service.py tests/training/test_student_worker.py; CUDA integration may skip only when CUDA is unavailable.
- [ ] Step 7: Commit with git commit -am "feat: run real H3 Teacher through multi-GPU service" plus the new module and tests.

### Task 3: Replace complete-candidate Revision with trusted RevisionPatch

Files:
- Create: harness4h3/campaign/revision.py
- Modify: harness4h3/campaign/reviews.py
- Modify: harness4h3/campaign/proposals.py
- Modify: harness4h3/student/campaign.py
- Test: tests/unit/test_revision_patch.py
- Test: tests/integration/test_revision_execution_consistency.py

Interfaces:
- Add RevisionPatch.from_dict, apply_revision_patch(proposal, patch) -> StudentProposal, and canonical_candidate_from_proposal(original, proposal, patch) -> CandidateEnvelope.
- Patch fields are candidate_id, base_digest, operations, changed_fields, resolved_objection_ids, and reason.

- [ ] Step 1: Write tests proving a patch changes a proposal, recomputes its digest, and rebuilds Candidate architecture/training/deployment/provenance from that proposal.
- [ ] Step 2: Run the focused tests and observe failure.
- [ ] Step 3: Allow only replace and add on registered architecture/training/deployment paths. Reject parent, generation, candidate id, Teacher identity, base digest, unknown paths, and values rejected by StudentProposal parsing/validation.
- [ ] Step 4: Change revision JSON schema, prompt, and parser to return RevisionPatch; preserve Advocate prediction on the rebuilt Candidate.
- [ ] Step 5: After review, rerun StudentProposal, capability, parent/generation, validate_batch, and compiler validation. Record patch, proposal digest, and compile manifest digest in DecisionTrace. Execute only the rebuilt Candidate.
- [ ] Step 6: Run .venv/bin/pytest -q tests/unit/test_revision_patch.py tests/integration/test_revision_execution_consistency.py tests/integration/test_campaign_control_plane.py; expect tampered full-candidate revisions to fail before compile.
- [ ] Step 7: Commit with git commit -am "feat: canonicalize reviewed Student proposals" plus the new module and tests.

### Task 4: Add target-device evidence and gate integration

Files:
- Create: harness4h3/student/edge.py
- Modify: harness4h3/campaign/gates.py
- Modify: harness4h3/campaign/adapters.py
- Modify: harness4h3/student/campaign.py
- Modify: harness4h3/student/remote.py
- Test: tests/unit/test_target_device_evaluator.py
- Test: tests/integration/test_target_device_gate.py

Interfaces:
- Add EdgeEvidence with artifact hash, target-device id, and measurement reference.
- Add TargetDeviceRunner.export, quantize, compile, deploy, and benchmark.
- Add TargetDeviceEvaluator.evaluate(checkpoint, proposal, round_dir) -> tuple[EdgeEvidence, ...].

- [ ] Step 1: Write tests proving server evidence yields PROMOTABLE but complete non-server evidence yields TARGET_SATISFIED.
- [ ] Step 2: Implement deterministic fake and remote target runners. Require artifact hash and target device identity in every output.
- [ ] Step 3: Run target evaluation only after server evaluation is valid/promotable; add all eight edge records to the evidence map before AcceptanceGate.
- [ ] Step 4: Require edge_exported, edge_quantized, edge_runtime, edge_device, edge_latency, edge_memory, edge_energy, and edge_thermal, with non-server profile and matching artifact hash.
- [ ] Step 5: Run the focused tests and commit with git commit -am "feat: require target-device evidence for satisfaction".

### Task 5: Separate optimization and execution status

Files:
- Modify: harness4h3/student/campaign.py
- Modify: tools/student_campaign_supervisor.py
- Modify: harness4h3/cli.py
- Test: tests/integration/test_student_campaign_status.py

- [ ] Step 1: Add a test asserting PROMOTABLE, NO_PROGRESS, and BUDGET_EXHAUSTED map to optimization_status equal to the label, execution_status COMPLETED, and exit_code 0.
- [ ] Step 2: Add optimization_status, execution_status, and exit_code to CampaignResult; map integrity, infrastructure, and unexpected exceptions to the three failure statuses.
- [ ] Step 3: Make CLI and Supervisor return result.exit_code, never an optimization label.
- [ ] Step 4: Run the focused tests and commit with git commit -am "fix: separate campaign optimization and execution status".

### Task 6: Enforce Progressive Distillation Teacher lineage

Files:
- Modify: harness4h3/student/worker.py
- Modify: harness4h3/student/campaign.py
- Test: tests/training/test_progressive_teacher_lineage.py
- Test: tests/integration/test_campaign_parent_lineage.py

- [ ] Step 1: Write a test for 16->8->4 proving stage two Teacher SHA equals stage one child SHA and every stage records Student/Teacher NFE.
- [ ] Step 2: Execute plan_binary_stages one stage at a time; publish and hash each child; feed that child as next stage Teacher; persist parent/Teacher/child hashes and stage status.
- [ ] Step 3: Make parent.selected and training.metric use the same child path/hash and reject restart events whose file hash differs.
- [ ] Step 4: Run the focused tests and commit with git commit -am "feat: persist progressive Teacher checkpoint lineage".

### Task 7: Add algorithm-aware memory estimates

Files:
- Modify: harness4h3/student/proposal.py
- Modify: harness4h3/student/compiler.py
- Modify: harness4h3/student/worker.py
- Test: tests/unit/test_student_memory_estimator.py

- [ ] Step 1: Write a test showing DMD2 estimate exceeds Student-only estimate and includes two optimizer states, critic, gradients, and activations.
- [ ] Step 2: Implement a method-specific estimator and persist its component breakdown and name in ValidationReport and CompileManifest; use it in GPU budget validation.
- [ ] Step 3: Run the focused tests and commit with git commit -am "feat: estimate Student memory by algorithm".

### Task 8: Reuse and instrument the H3 Teacher baseline

Files:
- Modify: tools/student_teacher_baseline_worker.py
- Modify: harness4h3/student/remote.py
- Test: tests/integration/test_teacher_baseline_reuse.py

- [ ] Step 1: Add a fake-loader test asserting one H3 load for all manifest cases and seeds.
- [ ] Step 2: Move model construction before the case loop; record model-load, sampling, decode, quality, and peak-memory timings separately. Keep reconstruction baseline distinct from executable H3 generation baseline.
- [ ] Step 3: Run the focused tests and commit with git commit -am "fix: reuse one H3 Teacher baseline model".

### Task 9: Configure independent review models and contract coverage

Files:
- Modify: harness4h3/student/config.py
- Modify: harness4h3/student/campaign.py
- Modify: harness4h3/campaign/reviews.py
- Test: tests/unit/test_review_model_configuration.py
- Test: tests/contracts/test_student_execution_truth.py

- [ ] Step 1: Parse controller, advocate, critical, and revision model settings with controller fallback; persist distinct role identities while retaining role-specific prompts.
- [ ] Step 2: Add contracts for distinct devices, every Teacher rank, online forward, Revision digest/capability revalidation, critic memory, stage lineage, normal PROMOTABLE exit 0, and edge-only TARGET_SATISFIED.
- [ ] Step 3: Run .venv/bin/pytest -q tests/contracts tests/unit tests/integration tests/training; only explicitly hardware-dependent tests may skip.
- [ ] Step 4: Commit with git commit -am "test: verify Student execution truth contracts".

### Task 10: Final audit and publication

Files:
- Modify: documentation only if emitted evidence requires clarification.

- [ ] Step 1: Run .venv/bin/python -m compileall harness4h3 tools and .venv/bin/pytest -q.
- [ ] Step 2: Inspect one successful fake campaign and one integrity failure across DecisionTrace, compile manifest, TrainingResult, checkpoint metadata, edge evidence, and status JSON.
- [ ] Step 3: Verify every objective invariant against direct evidence, then run git add -A, git commit -m "feat: complete Student execution truth", and git push origin main.
