# Existing Student Warm-Start Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the existing 1–2B H3 Student from its authenticated v12 F3 checkpoint and give the corrected H3-shifted progressive-distillation path enough FP32 optimizer-state training to learn coherent latent structure before semantic acceptance is judged.

**Architecture:** Keep the registered `video_latent_dit` graph, 36-latent Student window, real three-rank H3 Teacher, and seven-window overlap contract unchanged. Run three detached trusted worker jobs in order: F1 adds 256 optimizer steps, F2 adds 256, and F3 adds 512, all with `--fp32-optimizer-state`; each child is bound to the previous full-precision SHA. Evaluate only the final F3 int8 child with the existing eight-case, two-seed 30-second manifest.

**Tech Stack:** Python 3.12, PyTorch BF16 Student, FP32 AdamW state, real MiniMax-H3 BF16 Teacher, safetensors, ComfyUI H3 VAE, CLIP temporal evaluator, SSH detached workers.

## Global Constraints

- Preserve the existing 1–2B Student graph, latent shape, parameter names, and H3 VAE.
- Use the authenticated v12 F3 Student checkpoint as the parent; do not start a fresh teacher-seed model for this convergence run.
- Keep `offline_simulation=false`, an online sharded H3 Teacher with exactly three ranks, and real Teacher forward evidence.
- Use the shared H3 video-shifted sigma schedule (`shift=12`) in both training and inference.
- Preserve F1→F2→F3 parent SHA lineage and retain every full-precision checkpoint.
- Keep `window_count=7`, `latent_frames=36`, `overlap_latent_frames=6`, `window_stride_latent_frames=30`, 24 FPS, 256×256, and the 708–732-frame contract.
- Do not add Wan, Thor/Orin, mobile runtime, LLM code editing, or architecture search.
- A candidate is not accepted unless all structural checks pass and semantic reaches the authenticated Teacher-relative floor.

---

### Task 1: Verify the corrected remote substrate and immutable parent

**Files:**
- Read: `/data/models/harness4h3-v27-code-long-30s/student-campaign-30s-fixed-v12/student_0001_student_real_progressive_velocity_f1_F3/training-result.json`
- Read: `/data/models/harness4h3-v27-code-long-30s/student-campaign-30s-fixed-v13/student_0001_student_real_progressive_position_f1_F3/compile/compile_manifest.json`
- Read: remote `harness4h3/student/schedule.py`, `harness4h3/student/worker.py`, and `h3_training/algorithms/progressive_distillation.py`

**Interfaces:**
- Consumes the v12 full-precision parent SHA `ceafe4e82782b27c64d410c2bcfd757ffcdaa1d4846b587a990da67fcc24b1e2` and Teacher SHA `907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182`.
- Produces a clean handoff state and an auditable remote run root `/data/models/harness4h3-v27-code-long-30s/student-campaign-30s-warmstart-v16`.

- [ ] **Step 1: Confirm no worker or stale marker owns the GPUs**

Run:

```bash
ssh ai4mbse-remote 'pgrep -af "student_train_worker.py|student_evaluate_worker.py" || true; ls -l /data/models/harness4h3-v27-code/evidence/student-campaign/.controller-handoff-hold.json /data/models/harness4h3-v27-code/evidence/student-campaign/.controller-release.json /data/models/harness4h3-v27-code/evidence/student-campaign/.h3-worker-gpu-lease.json 2>/dev/null || true'
```

Expected: no active Student worker; if a marker exists, inspect its `owner_pid` and release it only when that PID is absent.

- [ ] **Step 2: Verify the parent hash and corrected code symbols**

Run:

```bash
ssh ai4mbse-remote '/home/intern/miniconda3/envs/comfy/bin/python - <<"PY"
import hashlib, json
from pathlib import Path

parent = Path("/data/models/harness4h3-v27-code-long-30s/student-campaign-30s-fixed-v12/student_0001_student_real_progressive_velocity_f1_F3/student.safetensors")
h = hashlib.sha256()
with parent.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(block)
print("parent_sha256", h.hexdigest())
result = json.loads(Path("/data/models/harness4h3-v27-code-long-30s/student-campaign-30s-fixed-v12/student_0001_student_real_progressive_velocity_f1_F3/training-result.json").read_text())
assert result["full_precision_sha256"] == h.hexdigest()
assert result["offline_simulation"] is False
assert result["teacher_world_size"] == 3
print("teacher_sha256", result["parent_sha256"])
PY
rg -n "h3_video_sigmas|student_video_sigmas|rollout_steps|fp32_optimizer_state" /home/intern/huangjiahao/Harness4H3-long-video-30s/harness4h3/student /home/intern/huangjiahao/Harness4H3-long-video-30s/h3_training/algorithms/progressive_distillation.py'
```

Expected: the computed parent SHA is the exact v12 SHA above; the code contains shared shifted schedule, rollout policy, and FP32 optimizer-state support.

### Task 2: Run corrected warm-start F1 and bind its child

**Files:**
- Remote output: `/data/models/harness4h3-v27-code-long-30s/student-campaign-30s-warmstart-v16/direct-f1-shifted-fp32`
- Remote result: `training-result.json`

**Interfaces:**
- Consumes the v12 F3 full-precision parent and the real H3 Teacher checkpoint.
- Produces an F1 Student child with `parent_inherited=true`, `offline_simulation=false`, `train_steps=256`, `cumulative_train_steps=256`, and `fp32_optimizer_state=true` in every stage lineage record.

- [ ] **Step 1: Launch the detached real H3 F1 worker**

Run the trusted worker with:

```bash
--max-steps 1024 --train-steps 256 --fidelity F1 --learning-rate 0.0003 --fp32-optimizer-state
--parent-checkpoint /data/models/harness4h3-v27-code-long-30s/student-campaign-30s-fixed-v12/student_0001_student_real_progressive_velocity_f1_F3/student.safetensors
--parent-checkpoint-sha256 ceafe4e82782b27c64d410c2bcfd757ffcdaa1d4846b587a990da67fcc24b1e2
--parent-candidate-id student_v12_progressive_velocity_f3
```

Use the existing manifest, real Teacher, cache, ComfyUI root, three Teacher ranks, and controller handoff paths from `configs/student-campaign.long-video-30s.yaml`. Run detached and write stdout/stderr to `direct-f1-shifted-fp32/worker.log`.

- [ ] **Step 2: Verify F1 before starting F2**

Require `status=success`, `parent_inherited=true`, the exact v12 parent SHA, `offline_simulation=false`, `teacher_world_size=3`, `teacher_forward_count>0`, `optimizer_steps=256`, `cumulative_train_steps=256`, and `fp32_optimizer_state=true` for every stage record. Record the printed `full_precision_sha256` as `F1_SHA`; do not infer it from a filename.

### Task 3: Run corrected warm-start F2 and F3 with exact lineage

**Files:**
- Remote outputs: `direct-f2-shifted-fp32`, `direct-f3-shifted-fp32`
- Remote results: one `training-result.json` per fidelity

**Interfaces:**
- F2 consumes `F1_SHA` and the F1 full-precision checkpoint; F3 consumes `F2_SHA` and the F2 full-precision checkpoint.
- F2 adds 256 steps (`cumulative_train_steps=512`); F3 adds 512 steps (`cumulative_train_steps=1024`).
- Each run remains online, sharded, and FP32-optimizer-state enabled.

- [ ] **Step 1: Launch F2 only after F1 verification**

```bash
F1_DIR=/data/models/harness4h3-v27-code-long-30s/student-campaign-30s-warmstart-v16/direct-f1-shifted-fp32
F1_FULL="$F1_DIR/student.safetensors"
F1_SHA=$(sha256sum "$F1_FULL" | awk '{print $1}')
--train-steps 256 --fidelity F2 --fp32-optimizer-state
--parent-checkpoint "$F1_FULL"
--parent-checkpoint-sha256 "$F1_SHA"
--parent-candidate-id student_warmstart_f1_fp32
```

- [ ] **Step 2: Verify F2 lineage**

Require `status=success`, `parent_inherited=true`, `parent_sha256=F1_SHA`, `optimizer_steps=256`, `cumulative_train_steps=512`, `offline_simulation=false`, `teacher_forward_count>0`, and `fp32_optimizer_state=true`. Record `F2_SHA`.

- [ ] **Step 3: Launch F3 only after F2 verification**

```bash
F2_DIR=/data/models/harness4h3-v27-code-long-30s/student-campaign-30s-warmstart-v16/direct-f2-shifted-fp32
F2_FULL="$F2_DIR/student.safetensors"
F2_SHA=$(sha256sum "$F2_FULL" | awk '{print $1}')
--train-steps 512 --fidelity F3 --fp32-optimizer-state
--parent-checkpoint "$F2_FULL"
--parent-checkpoint-sha256 "$F2_SHA"
--parent-candidate-id student_warmstart_f2_fp32
```

- [ ] **Step 4: Verify F3 terminal evidence**

Require `status=success`, `parent_inherited=true`, `parent_sha256=F2_SHA`, `optimizer_steps=512`, `cumulative_train_steps=1024`, `offline_simulation=false`, `teacher_world_size=3`, `teacher_forward_count>0`, and complete two-stage F3 lineage with FP32 optimizer state. Record the final full-precision and int8 hashes.

### Task 4: Evaluate the final F3 child on the unchanged 30-second contract

**Files:**
- Remote output: `/data/models/harness4h3-v27-code-long-30s/student-campaign-30s-warmstart-v16/direct-f3-eval-shifted-fp32`
- Remote result: `evaluation-result.json`

**Interfaces:**
- Consumes the final F3 int8 checkpoint and the existing eight-case manifest (4 prompts × 2 seeds).
- Produces full structural, continuation, semantic, temporal, motion, latency, memory, and model-size evidence.

- [ ] **Step 1: Run the evaluator detached**

Use the existing evaluator with `--quality-backend clip_temporal --quality-device cpu --fidelity F3 --max-cases 16 --seed-count 4 --verifier-strength full --decoded-frames-min 708 --decoded-frames-max 732 --output-fps 24 --output-width 256 --output-height 256 --duration-min-s 29.5 --duration-max-s 30.5 --window-count 7 --overlap-latent-frames 6 --window-stride-latent-frames 30 --release-controller-handoff`.

- [ ] **Step 2: Audit every acceptance requirement from JSON**

The candidate passes only if every case reports 708–732 decoded frames, 29.5–30.5 seconds, 24 FPS, 256×256, continuation verified, no decode/black-frame/truncation/repetition failure, and `offline_simulation=false`; aggregate semantic must be at least `0.603502` and the output must remain in the 1–2B lineage.

- [ ] **Step 3: Compare latent structure against v12 and v15**

Run a read-only one-prompt/one-seed probe and record spatial/temporal adjacent-latent correlation. The diagnostic target is a material move toward the Teacher range (`~0.92–0.99`), not merely a valid MP4 or structural score.

### Task 5: Update evidence documentation without claiming semantic success prematurely

**Files:**
- Modify: `docs/long-video-30s-acceptance-report.md`
- Modify: `docs/student-campaign.md` only if the final operational budget or warm-start rule becomes the new mainline

- [ ] **Step 1: Add v16 evidence**

Record the parent SHA, F1/F2/F3 child SHAs, cumulative steps, FP32 optimizer-state flag, real Teacher forward counts, corrected schedule, and exact evaluation result path.

- [ ] **Step 2: State the gate outcome separately**

If semantic remains below `0.603502`, mark v16 diagnostic and keep the goal active. Only record acceptance after the full multi-prompt/multi-seed audit passes.

- [ ] **Step 3: Run repository verification**

```bash
git diff --check
```

Do not modify unrelated dirty files or overwrite v12/v13/v15 evidence.
