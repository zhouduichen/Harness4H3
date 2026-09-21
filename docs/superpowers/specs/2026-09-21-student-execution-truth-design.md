# Student Execution Truth Design

## Goal

Make the Student Campaign's declared Candidate, algorithm, Teacher, resource allocation, device evidence, and terminal status match the code path that actually executes, while preserving the existing control-plane, review, lineage, fidelity, immutable-base, and Pareto foundations.

## Scope

This design covers the active P0/P1 execution-truth requirements:

- a real multi-GPU online H3 Teacher service with an explicit request/response boundary;
- trusted `RevisionPatch` application that rebuilds a canonical `StudentProposal` and Candidate;
- target-device export, quantization, runtime, and benchmark evidence;
- separate optimization and execution status with failure-only non-zero exits;
- progressive-distillation stage Teacher lineage;
- algorithm-aware memory estimates, one-load Teacher baseline evaluation, and configurable review models;
- contract and integration tests for every stated invariant.

The existing controller, campaign event log, immutable verification base, and Pareto/Novelty archive remain the system of record. No second control framework is introduced.

## Architecture

### 1. Real Teacher service

`student_train_worker.py` acquires one independent Student GPU and `teacher_world_size` independent Teacher ranks. A rank-0 process owns a small RPC server; each Teacher rank loads the real `RealMiniMaxH3Adapter` model on its assigned CUDA device. Requests contain CPU-serialized tensors for `x_t`, timestep, conditioning, and optional audio state. The service reconstructs tensors on the rank/model device, runs real H3 forward, moves the prediction to CPU, and returns a length-prefixed response containing shape, dtype, and tensor bytes. The Student worker owns only Student/DMD2 state and converts responses onto the Student device.

The service is transport-agnostic at the training boundary: the worker uses a local multiprocessing connection/queue protocol in tests and a TCP loopback RPC endpoint in the real launcher. The protocol includes request id, Teacher rank, request tensor metadata, and error records. No fixed target is accepted by `RealH3TeacherBackend`; proxy/fixed-target backends remain explicitly offline-only.

GPU scheduling becomes role-aware. The scheduler selects `teacher_world_size` devices satisfying `teacher_rank_min_free_memory_gb` and one distinct Student device satisfying `student_min_free_memory_gb`. `worker_min_free_memory_gb` remains the aggregate worker admission requirement. The lease records role-to-GPU assignments and the worker passes them explicitly to the launcher. Assignment validation rejects overlap and under-provisioned ranks before model loading.

### 2. Canonical Candidate execution

The Revision LLM returns only a closed `RevisionPatch` schema: candidate id, base digest, changed field paths, patch operations limited to registered proposal paths, resolved objection ids, and reason. A trusted patch applier starts from the pre-review `StudentProposal`, applies only allowlisted operations, parses and validates the result, recomputes its digest, and rebuilds the CandidateEnvelope from the proposal's architecture/training/deployment values. Parent id and generation are immutable review inputs and are checked again by `validate_batch`; capability and compiler validation run on the rebuilt Candidate.

The Candidate stores the canonical proposal in provenance, together with the exact proposal digest and review patch. DecisionTrace records both the patch and the rebuilt Candidate. The adapter derives its compile manifest and worker input only from that canonical proposal, so the trace, compile artifact, TrainingResult, and checkpoint metadata share one digest.

### 3. Target-device evidence

Server evaluation produces only `promotable_for_edge_test`. A `TargetDeviceEvaluator` consumes the full-precision/quantized artifact, executes export and target compilation, deploys it to the configured target runner, and records `EdgeEvidence` for exactly these measurements: `edge_exported`, `edge_quantized`, `edge_runtime`, `edge_device`, `edge_latency`, `edge_memory`, `edge_energy`, and `edge_thermal`. Each record carries target device identity, artifact hash, measurement reference, and validity. Server evidence cannot satisfy any edge metric because the AcceptanceGate requires a non-server device profile and complete edge evidence before `TARGET_SATISFIED`.

The edge evaluator is injectable. The local contract implementation uses a deterministic fake target runner; the remote implementation invokes the existing remote command client. Both expose the same `export -> quantize -> compile -> deploy -> benchmark` interface and persist an evidence manifest adjacent to the checkpoint.

### 4. Status and lifecycle

`CampaignResult` carries `optimization_status`, `execution_status`, `exit_code`, and a stable `terminal_status`. A valid optimization result such as `PROMOTABLE`, `NO_PROGRESS`, or `BUDGET_EXHAUSTED` has `execution_status=COMPLETED` and `exit_code=0`. Infrastructure, integrity, and unexpected exceptions set `execution_status=FAILED` and a non-zero exit. Supervisor and CLI use `result.exit_code`, never the optimization label, to determine process success.

### 5. Progressive and resource semantics

Progressive Distillation creates an explicit stage record for every binary-halving transition. Stage `16->8` uses H3 as Teacher and publishes its child checkpoint. Stage `8->4` loads that checkpoint as Teacher and publishes the next child. Each record persists parent checkpoint hash, Teacher checkpoint hash, Student/Teacher NFE, child hash, and stage status. The compile/validation path rejects non-binary-halving source/target pairs before training.

Static validation calls an algorithm-specific resource estimator. DMD2 accounts for Student, Critic, both optimizer states, gradients, and activations; Progressive Distillation accounts for Student plus Teacher forward memory. The estimate is emitted in the compile manifest and checked against the role-specific GPU budget.

The baseline worker loads H3 once before iterating the fixed EvaluationManifest. Generation timing surrounds only sampling; decode and quality timing are recorded separately. Review model configuration accepts controller, advocate, critical, and revision identities independently and preserves same-model role separation as distinct roles rather than claiming independent model diversity.

## Error handling and integrity

- A missing or overlapping GPU lease fails before model load with `INFRA_FAILURE`.
- RPC timeout, malformed tensor metadata, or Teacher forward failure produces a typed training failure and no child checkpoint.
- A revision patch touching a non-allowlisted path, base digest, parent, generation, or candidate id is rejected before compile.
- Any proposal/compiler digest mismatch is an `INTEGRITY_FAILURE`.
- Incomplete edge evidence yields `PROMOTABLE`, never `TARGET_SATISFIED`.
- A restarted campaign resumes only from a persisted child checkpoint whose hash and Candidate id match the `parent.selected` event.

## Verification plan

The repository will add tests for:

- Teacher and Student on distinct devices, real online Teacher forward, and actual use of every Teacher rank;
- no fixed-target fallback in the real backend;
- RevisionPatch schema, trusted application, digest recomputation, capability/parent/generation/compiler revalidation, and trace-to-worker digest equality;
- target evidence gating and status/exit code behavior;
- DMD2 critic-aware memory estimate;
- progressive stage Teacher lineage and binary-halving rejection;
- one-load baseline timing fields;
- role-specific review model identities;
- campaign restart and parent hash continuity.

CUDA-dependent tests are marked and run when GPUs are available. CPU tests use fake modules and the same RPC/patch/evidence contracts, so they prove the boundary without pretending to prove GPU execution.

## Acceptance invariants

For every executed Candidate:

```text
Candidate.architecture == StudentProposal.architecture
Candidate.training_recipe == StudentProposal.training
Candidate.deployment_recipe == StudentProposal.deployment
Candidate.proposal_digest == StudentProposal.digest
DecisionTrace.patch == applied patch
TrainingResult.proposal_digest == compile manifest proposal_digest
checkpoint metadata.parent_sha256 == selected parent checkpoint hash
TARGET_SATISFIED => all edge evidence exists and device_profile_id != "server"
exit_code != 0 => execution_status in {INFRA_FAILURE, INTEGRITY_FAILURE, UNHANDLED_EXCEPTION}
```
