# FSDP Teacher, Target Profile, and Review Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution to implement this plan task-by-task.

**Goal:** Replace the replicated online H3 service with one collective FSDP Teacher runtime, make target-device acceptance profile-driven, and allow same-model adversarial review roles.

**Architecture:** The FSDP service launches exactly three ranks. Rank 0 owns request ingress/response egress; every request is serialized into a rank-0 broadcast, reconstructed on every rank, and forwarded through one FSDP-sharded H3 model. The target profile is independent from training memory and is consumed by the gate only after complete edge evidence exists. Review actors remain distinct by ActorIdentity, role, and prompt version while model names may be equal.

**Tech Stack:** PyTorch distributed/NCCL, FSDP, safetensors, dataclasses, existing Student Campaign gate and remote worker contracts.

## Global Constraints

- Teacher execution must remain online and must not use precomputed teacher targets.
- `teacher_world_size` is exactly 3 for production service validation; Student allocation must be disjoint.
- Reuse `_construct_model`/`_wrap_fsdp` semantics from `tools/h3_real_train_worker.py`, including rank-0 load and `sync_module_states`.
- Do not add new Agent, Archive, Trace, or parallel control frameworks.
- `TARGET_SATISFIED` requires quality plus latency, memory, energy, thermal, model-size, and profile compatibility checks.

### Task 1: Collective FSDP Teacher service

**Files:**
- Modify: `harness4h3/student/teacher_service.py`
- Modify: `harness4h3/student/worker.py`
- Modify: `harness4h3/student/inference.py`
- Add: `tools/student_fsdp_teacher_service.py`
- Test: `tests/training/test_teacher_service.py`, `tests/integration/test_student_teacher_service.py`

Implement a spawnable rank entrypoint that initializes NCCL, imports the trusted H3 API, constructs the model with rank-0 checkpoint load, wraps it with FSDP, and broadcasts serialized request metadata/tensors. Every rank calls the FSDP model for every request; rank 0 sends CPU prediction plus participation metadata back to the parent. `TeacherServiceHandle` must expose `ranks_used`, `forward_count`, `world_size`, and `sharded=True`. `RealH3TeacherBackend` must use this service for production and reject a service configuration whose world size is not three.

### Task 2: Baseline runtime reuse

**Files:**
- Modify: `tools/student_teacher_baseline_worker.py`
- Modify: `harness4h3/student/inference.py`
- Test: `tests/unit/test_student_baseline_runtime.py`

Add a distributed baseline launcher/handle using the same FSDP service and sample through its online predictor. Preserve one fixed `EvaluationManifest`; record model load time, sampling, decode, quality latency, and peak memory. Do not call the single-GPU `sample_h3_latent` path for the production H3 baseline.

### Task 3: Target device profile and gate

**Files:**
- Modify: `harness4h3/student/config.py`
- Modify: `harness4h3/student/edge.py`
- Modify: `harness4h3/student/campaign.py`
- Modify: `harness4h3/campaign/gates.py`
- Test: `tests/unit/test_target_device_profile.py`, `tests/integration/test_target_device_gate.py`

Add typed `TargetDeviceProfile` fields for runtime, limits, supported precision/quantization, and workload shape. Carry it in campaign configuration and candidate evidence. Map edge metrics to profile constraints and explicitly reject missing/invalid metrics, unsupported formats, incorrect resolution/frames/sampling steps, or limit violations. Keep `StudentTarget.max_peak_memory_gb` as training-only and use `TargetDeviceProfile.max_memory_gb` for edge memory.

### Task 4: Review identity semantics

**Files:**
- Modify: `harness4h3/student/config.py`
- Modify: `harness4h3/student/campaign.py`
- Modify: `harness4h3/campaign/reviews.py`
- Test: `tests/unit/test_campaign_reviews.py`, `tests/unit/test_student_config.py`

Remove the distinct-model-name requirement. Validate uniqueness of ActorIdentity, role, and prompt version instead. Use the controller model as the default for all three review agents; optional advocate/critical/revision model fields override only the model endpoint payload.

### Task 5: Verification and real evidence

Run focused contracts first, then the full test suite. If a configured GPU host is available, run the FSDP online Teacher smoke, DMD2 F1, Progressive Distillation F1, three-candidate F1, F1/F2/F3 fidelity sequence, and a bounded two-to-three-round campaign. Preserve GPU allocation, rank participation, checkpoint hashes, MP4, quality/hardware metrics, fidelity decisions, Pareto decisions, and DecisionTrace paths in the final report.
