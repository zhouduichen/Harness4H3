# Real Int8 Target Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan task-by-task with a test checkpoint after every task.

**Goal:** Make the target-device runtime consume the existing int8 Student artifact without first materializing the full dequantized model, then verify the change with real GPU3 target measurements while preserving the existing Gate and evidence boundaries.

**Architecture:** Add a small runtime-only `QuantizedLinear` implementation that stores int8 weights and per-tensor scales on the target device and dequantizes only the active layer for each matrix multiplication. Keep the existing dequantizing loader for server evaluation and training; opt into the quantized loader only from `student_target_device_worker.py`. The benchmark continues to measure the complete steady-state Student sampling plus H3 VAE decode/video-write interval, with no profile or Gate changes.

**Tech Stack:** Python 3.12, PyTorch, safetensors, pytest, remote CUDA/L40 GPU3, existing MiniMax-H3 ComfyUI VAE runtime.

## Global Constraints

- Do not change Controller, Student architecture, AcceptanceGate, TargetDeviceProfile, Pareto semantics, or measurement thresholds.
- Do not label a dequantized model as int8; target evidence must record the actual runtime mode.
- Preserve exact artifact SHA256, proposal digest, generated MP4, benchmark JSON, EdgeEvidence, and DecisionTrace contracts.
- Server evaluation remains on the existing dequantizing path unless its tests require no change.
- A target benchmark that still violates memory or energy remains a truthful Gate failure; do not synthesize or relax evidence.

---

### Task 1: Add a runtime-only quantized linear module

**Files:**
- Create: `harness4h3/student/runtime_quantization.py`
- Test: `tests/unit/test_student_runtime_quantization.py`

**Interfaces:**
- Produces `QuantizedLinear.from_linear(linear, weight_int8, scale)` and a forward pass accepting the same tensor shapes and dtype as `torch.nn.Linear`.
- Produces `load_runtime_quantized_model(proposal, checkpoint, target, device)` for target inference only.

- [ ] **Step 1: Write CPU tests first.**

Test that a quantized linear has int8 storage, preserves bias, matches a reference `F.linear` using `weight_int8.float() * scale`, and that a quantized Student model loads every non-linear tensor without missing or unexpected keys.

Run:

```bash
pytest tests/unit/test_student_runtime_quantization.py -q
```

Expected before implementation: import or attribute failures.

- [ ] **Step 2: Implement `QuantizedLinear`.**

Store `weight_int8` and `scale` as non-trainable buffers, retain a zero-sized dtype anchor for callers that inspect `.weight.dtype`, and compute:

```python
weight = self.weight_int8.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)
return torch.nn.functional.linear(x, weight, bias)
```

Do not keep a duplicate floating-point full weight buffer.

- [ ] **Step 3: Implement the runtime loader.**

Read raw safetensors and metadata. Build the Student on CPU, replace every `torch.nn.Linear` using its raw `.weight` and `.__scale` tensors, dequantize only non-linear tensors that have sidecar scales, load the filtered state dict, move the resulting model to the requested device/dtype, and validate that only the expected linear keys were handled.

If metadata is not `quantization=int8`, raise a clear runtime error instead of silently falling back.

- [ ] **Step 4: Run the unit test.**

```bash
pytest tests/unit/test_student_runtime_quantization.py -q
```

Expected: all tests pass on CPU.

- [ ] **Step 5: Commit the isolated runtime loader.**

```bash
git add harness4h3/student/runtime_quantization.py tests/unit/test_student_runtime_quantization.py
git commit -m "feat: add real int8 target runtime loader"
```

### Task 2: Switch only the target-device worker to the real int8 path

**Files:**
- Modify: `tools/student_target_device_worker.py`
- Test: `tests/unit/test_student_runtime_quantization.py`
- Test: `tests/integration/test_target_device_gate.py`

**Interfaces:**
- `student_target_device_worker.py` calls `load_runtime_quantized_model` for the compiled artifact.
- Server-side `harness4h3.student.inference.load_student_model` remains unchanged for evaluator compatibility.

- [ ] **Step 1: Add a worker-level contract test.**

Monkeypatch the target worker’s loader and assert a compiled artifact tagged int8 selects the runtime quantized loader; assert a non-int8 artifact fails before benchmark evidence is written.

- [ ] **Step 2: Update the worker import and call site.**

Replace only the target worker’s `load_student_model(...)` call with `load_runtime_quantized_model(...)`. Preserve the existing resident Student+VAE measurement boundary, `runtime_resident` field, artifact SHA, MP4 output, and all nine EdgeEvidence records.

Record `quantization_mode=runtime_int8_weight_only` in benchmark metadata so the evidence distinguishes this path from the previous dequantizing implementation.

- [ ] **Step 3: Run focused tests.**

```bash
pytest tests/unit/test_student_runtime_quantization.py tests/integration/test_target_device_gate.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run the full local suite.**

```bash
pytest -q
```

Expected: the prior CUDA-only skips remain skips; no existing contract, fidelity, campaign, or target-profile test regresses.

- [ ] **Step 5: Commit and push the worker integration.**

```bash
git add tools/student_target_device_worker.py tests/unit/test_student_runtime_quantization.py tests/integration/test_target_device_gate.py
git commit -m "fix: use actual int8 runtime for target benchmark"
git push origin main
```

### Task 3: Deploy and validate a real target benchmark

**Files/evidence:**
- Remote: `/home/intern/huangjiahao/Harness4H3-codex-20260923/tools/student_target_device_worker.py`
- Remote: `/home/intern/huangjiahao/Harness4H3-codex-20260923/evidence/student-campaign-full-v25/`

- [ ] **Step 1: Copy the committed worker and verify source identity.**

Copy the worker to the remote harness, run `py_compile`, and compare local/remote SHA256 before launching any GPU job.

- [ ] **Step 2: Run one real candidate-2 target benchmark on GPU3.**

Use the existing R6 F3 checkpoint and proposal. Require `offline_simulation=false`, `runtime_resident=true`, `quantization_mode=runtime_int8_weight_only`, non-empty MP4, artifact-bound edge evidence, and the complete benchmark fields.

- [ ] **Step 3: Compare measurements against the unchanged profile.**

Check latency ≤ 1.0s, memory ≤ 6.0GB, energy ≤ 20.0J, thermal ≤ 75C, and model size ≤ 2.0GB. If any check fails, preserve the failure evidence and do not alter the profile.

- [ ] **Step 4: Run the required full campaign only if the single-candidate target benchmark is viable.**

Run the existing 3-candidate F1→F2→F3 campaign configuration, then verify training/evaluation/edge/gate event counts, MP4s, SHA lineage, effective edge metrics, Pareto archive, parent events, experience deltas, and final DecisionTrace integrity.

- [ ] **Step 5: Commit evidence references and report the authoritative result.**

The final report must state the exact campaign result and any remaining hard violations. It must never call `TARGET_SATISFIED` unless the real profile checks pass through `StudentCampaignAdapter → StudentCampaign → AcceptanceGate`.

### Task 4: Final acceptance audit

- [ ] **Step 1: Verify code and GitHub identity.**

Check clean worktree, local HEAD, and `git ls-remote origin refs/heads/main`.

- [ ] **Step 2: Verify the full objective evidence matrix.**

Confirm FSDP allocation/rank participation, Teacher and Student peak memory, optimizer and fidelity budgets, parent/child hashes, MP4s, quality/server/edge metrics, fidelity decisions, Pareto/Parent decisions, and DecisionTrace event integrity.

- [ ] **Step 3: Mark the goal complete only if every explicit requirement is proven.**

Otherwise leave the goal active with the exact measured blocker and the evidence path.
