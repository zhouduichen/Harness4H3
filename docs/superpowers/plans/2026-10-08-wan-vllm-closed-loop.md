# Wan vLLM Closed-Loop Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution in this side conversation; sub-agents are unavailable.

**Goal:** Run real Wan Student rounds through fixed 30-second evaluation, structured vLLM proposals, trusted training, and lineage-aware parent selection.

**Architecture:** Add one focused remote campaign supervisor that invokes the existing Wan trainer and long-video evaluator, with a strict JSON proposal validator and deterministic selection. Keep vLLM advisory and schema-bound; keep all paths, devices, Teacher identity, and execution commands trusted in the supervisor/configuration.

**Tech Stack:** Python 3.9+, PyYAML, urllib OpenAI-compatible API, existing PyTorch/diffusers Wan worker, existing H3 VAE/CLIP evaluator, pytest.

## Global Constraints

- Use the existing real H3 Teacher and `offline_simulation=false`.
- Preserve the 30-second continuation contract: 708–732 frames, 24 FPS, 256×256, decodable, no black/all-repeated frames.
- Preserve checkpoint lineage and never accept a child whose parent digest is inconsistent.
- vLLM may emit only the validated proposal JSON; it may not emit code, shell commands, paths, device IDs, or architecture changes.
- Do not stop or overwrite currently running experiment roots.

---

### Task 1: Add the constrained Wan proposal and campaign state model

**Files:**
- Create: `harness4h3/student/wan_campaign.py`
- Test: `tests/unit/test_wan_campaign.py`

**Interfaces:**
- Produces `WanTrainingProposal.from_mapping`, `WanTrainingProposal.to_dict`, `validate_proposal`, `select_parent`, and JSON serialization helpers.
- Consumes evaluation/training mappings and trusted campaign configuration.

- [ ] **Step 1: Write failing tests for schema, ranges, digest binding, and selection.**
- [ ] **Step 2: Run `pytest tests/unit/test_wan_campaign.py -q` and observe failures.**
- [ ] **Step 3: Implement the frozen proposal dataclass, strict unknown-key rejection, numeric bounds, parent SHA-256 binding, deterministic selection, and atomic JSON/event helpers.**
- [ ] **Step 4: Run the focused tests and require PASS.**

### Task 2: Add the remote Wan vLLM campaign supervisor

**Files:**
- Create: `tools/wan_vllm_campaign.py`
- Create: `configs/student-campaign.wan-vllm-loop.yaml`
- Test: `tests/unit/test_wan_vllm_campaign.py`

**Interfaces:**
- CLI: `python tools/wan_vllm_campaign.py --config <yaml> [--max-rounds N]`.
- Consumes the proposal model from Task 1, the existing `wan_student_train_worker.py`, the existing `wan_student_long_worker.py` generation contract, and the OpenAI-compatible `/v1/chat/completions` endpoint.
- Produces round artifacts and `campaign-result.json`.

- [ ] **Step 1: Write failing tests for vLLM JSON extraction, trusted command construction, and failure-stop behavior.**
- [ ] **Step 2: Run the focused tests and observe failures.**
- [ ] **Step 3: Implement config loading, proposal request/retry, long-video evaluation invocation, trusted training invocation, parent selection, and durable artifacts.**
- [ ] **Step 4: Run focused tests and a `--help`/config validation smoke check.**

### Task 3: Deploy only the new loop files and run a real one-round smoke experiment

**Files:**
- Remote: `/home/intern/huangjiahao/Harness4H3-long-video-30s/tools/wan_vllm_campaign.py`
- Remote: `/home/intern/huangjiahao/Harness4H3-long-video-30s/configs/student-campaign.wan-vllm-loop.yaml`

**Interfaces:**
- Parent checkpoint: `/data/models/harness4h3-v27-code-long-30s/wan-student-5s-w5-semantic/student.safetensors` when its result is complete and digest-valid.
- Campaign root: `/data/models/harness4h3-v27-code-long-30s/wan-vllm-loop-w6`.

- [ ] **Step 1: Verify the current w5 result and parent checkpoint before launch.**
- [ ] **Step 2: Copy only the new supervisor/config to the remote host; do not overwrite active worker source files.**
- [ ] **Step 3: Launch one real round under a new `/data` campaign root.**
- [ ] **Step 4: Verify vLLM proposal, training command, evaluation artifact, and lineage fields as they appear.**

### Task 4: Run the configured experiment budget and report evidence

**Files:**
- Remote artifacts only under the new campaign root.

- [ ] **Step 1: Increase to the configured max rounds only after the one-round smoke has complete artifacts.**
- [ ] **Step 2: Monitor process, campaign result, GPU allocation, and disk space without touching unrelated jobs.**
- [ ] **Step 3: Report accepted parent, semantic/aggregate metrics, proposal history, and any stop reason.**
