# 30-Second Overlap-Window Student Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the real-H3 1–2B Student from five-second clips to a continuous 708–732-frame video assembled from overlapping latent continuation windows.

**Architecture:** Keep the 36-latent Student graph and F1 -> F2 -> F3 checkpoint lineage. Add a shared H3 temporal/window contract, a continuation sampler that anchors each new window to the previous latent tail, tiled per-window VAE decoding with decoded overlap crossfades, and authenticated multi-prompt/seed Teacher caches consumed throughout training and evaluation.

**Tech Stack:** Python 3.12, PyTorch, OpenCV, H3 VAE/Teacher service, YAML, pytest, existing campaign/remote worker protocols.

## Global Constraints

- Final output is 708–732 decoded frames, 24 FPS, 256x256, duration 29.5–30.5 seconds.
- Seven 36-latent windows use six latent overlap frames and stride 30; exact H3 VAE mapping determines decoded overlap.
- Continuation is required; independent short clips and simple post-hoc concatenation are invalid.
- Real MiniMax-H3 Teacher, `synthetic_latent=false`, and `offline_simulation=false` are required for production evidence.
- Existing 1–2B Student architecture, F1 -> F2 -> F3 stages, and checkpoint SHA lineage remain authoritative.
- Preserve all unrelated existing worktree changes.
- Thor/Orin/mobile deployment and architecture search are not acceptance criteria for this plan.

---

### Task 1: Centralize the 30-second temporal/window contract

**Files:**
- Modify: `harness4h3/student/video_contract.py`
- Modify: `tools/h3_long_video_cache_worker.py`
- Modify: `harness4h3/student/config.py`
- Test: `tests/unit/test_student_video_contract.py`
- Test: `tests/unit/test_student_config.py`

**Interfaces:**
- `h3_decoded_frames(latent_frames: int) -> int` is the single H3 temporal mapping.
- `VideoOutputSpec` exposes `window_count`, `overlap_latent_frames`, and `window_stride_latent_frames`.
- `VideoOutputSpec.long_window_spec()` returns the validated window geometry used by runtime and workers.

- [ ] Add tests for 36 -> 120, 6 -> 18, seven-window composition -> 732, invalid overlap/count, and 708/732 final contract acceptance.
- [ ] Move the existing H3 temporal mapping into `video_contract.py` and import it from the Teacher cache worker.
- [ ] Extend `VideoOutputSpec` without changing short-profile defaults; include window fields in `to_dict()` and `digest`.
- [ ] Reject a multi-window contract when the model latent frame count, window latent frame count, or stride is inconsistent.
- [ ] Run `./.venv/bin/pytest tests/unit/test_student_video_contract.py tests/unit/test_student_config.py -q` and expect all focused tests to pass.

### Task 2: Implement overlap-window continuation and composition

**Files:**
- Create: `harness4h3/student/long_video.py`
- Modify: `harness4h3/student/inference.py`
- Modify: `harness4h3/student/evaluator.py`
- Test: `tests/unit/test_student_long_video.py`

**Interfaces:**
- `LongVideoWindowSpec` describes fixed window/overlap/count geometry.
- `sample_student_long_video(...) -> (frames, trace)` keeps one Student model resident and returns a continuation trace.
- `compose_decoded_windows(windows, overlap_frames) -> torch.Tensor` crossfades decoded overlap and appends suffixes.

- [ ] Write CPU tests with a fake sampler proving the next prefix equals the prior tail, seeds differ deterministically by window, and seven synthetic windows compose to 732 frames.
- [ ] Implement prefix anchoring during every sampling step, with the final step exactly equal to the prior tail; do not add a second independent model path.
- [ ] Implement decoded overlap crossfade and continuity statistics; reject wrong window count or wrong latent shape.
- [ ] Add `generate_long_video(...)` that loads one Student, samples all windows, decodes each with the existing H3 VAE, composes, writes one MP4, and returns trace metadata.
- [ ] Make `StudentEvaluator` require `continuation_verified=true` and a matching contract digest for multi-window output.
- [ ] Run `./.venv/bin/pytest tests/unit/test_student_long_video.py tests/unit/test_student_video_contract.py -q`.

### Task 3: Make the real Teacher cache and Student training sequence-aware

**Files:**
- Modify: `harness4h3/student/inference.py`
- Modify: `tools/h3_long_video_cache_worker.py`
- Modify: `tools/h3_real_train_worker.py`
- Modify: `harness4h3/student/evaluation_manifest.py`
- Test: `tests/training/test_real_h3_support.py`
- Test: `tests/unit/test_student_evaluation_manifest.py`

**Interfaces:**
- Teacher cache items carry `sequence_id`, `window_index`, `window_count`, `overlap_latent_frames`, `continuation_source`, and authenticated real flags.
- `_prepare_cache_items(cache_dir, seed, latent_frames) -> list[dict]` returns all validated prompt/seed/window items in deterministic order.

- [ ] Add a fake-predictor test proving Teacher continuation windows are generated from the preceding tail and preserve authenticated metadata.
- [ ] Extend the real Teacher worker to generate seven windows per prompt/seed, save every window item, and write a sequence manifest without any synthetic fallback.
- [ ] Keep evaluation cases anchored to one sequence while allowing the runtime to discover all sibling windows; preserve existing manifest digest behavior for short profiles.
- [ ] Replace the training worker's single-cache-item assumption with deterministic cycling across all authenticated cache items at every update; retain FSDP Teacher and parent/child checkpoint handling.
- [ ] Run focused training/manifest tests and confirm old five-frame smoke tests still pass.

### Task 4: Thread long-video mode through evaluator, baseline, target profile, and remote launch

**Files:**
- Modify: `tools/student_evaluate_worker.py`
- Modify: `tools/student_teacher_baseline_worker.py`
- Modify: `harness4h3/student/remote.py`
- Modify: `tools/student_target_device_worker.py`
- Modify: `harness4h3/student/quality.py`
- Test: `tests/integration/test_student_cli.py`
- Test: `tests/unit/test_student_quality.py`

**Interfaces:**
- Remote evaluator and baseline commands carry the full `VideoOutputSpec` window fields.
- Long quality evaluation samples at least 12 stratified frames and reports mean plus lower-quantile semantic evidence.

- [ ] Add CLI flags for window count, overlap, and stride and construct the same digest-bearing spec in every worker.
- [ ] Use `generate_long_video` for multi-window specs and keep the existing `generate_video` path for explicit short profiles.
- [ ] Make Teacher baseline use identical continuation geometry and quality sampling as Student.
- [ ] Require `frames` in `TargetDeviceProfile` to mean decoded frames; preserve the short profile as a separate explicit configuration and do not add mobile/edge acceptance.
- [ ] Run integration command-construction tests and quality tests.

### Task 5: Add the 30-second campaign configuration and semantic gates

**Files:**
- Create: `configs/student-campaign.long-video-30s.yaml`
- Modify: `harness4h3/student/campaign.py`
- Modify: `harness4h3/student/fidelity.py`
- Test: `tests/contracts/test_student_campaign_contracts.py`
- Test: `tests/integration/test_student_long_video_campaign.py`

**Interfaces:**
- The new config uses the authentic Teacher, long-video cache/evaluation commands, target latent frames 36, and a 708–732 decoded contract.
- F1/F2/F3 evidence includes `video_contract_digest`, `continuation_verified`, semantic evidence, and parent checkpoint hashes.

- [ ] Base the new config on the existing real campaign, changing only the video objective and long cache/evaluation paths; do not alter architecture family or lineage semantics.
- [ ] Set multi-prompt/multi-seed manifest requirements and reject structural-proxy quality at F2/F3.
- [ ] Ensure a failed F1/F2 gate cannot advance and only a valid selected child is emitted as the next Parent.
- [ ] Keep target-device execution optional and out of the 30-second acceptance gate.
- [ ] Run campaign contract tests for F1 -> F2 -> F3, digest identity, and no offline simulation.

### Task 6: Run evidence-producing smoke and audit the full objective

**Files:**
- Create: `tools/long_video_smoke.py`
- Modify: `docs/student-campaign.md`
- Create: `docs/long-video-30s-acceptance-report.md`

- [ ] Run one real Teacher and one Student sequence with multiple prompts and seeds; require every output to decode within 708–732 frames and pass continuation metadata checks.
- [ ] Verify MP4s have 256x256, 24 FPS, 29.5–30.5 seconds, no black frames, no truncation, no all-frame repetition, and `offline_simulation=false`.
- [ ] Verify F1/F2/F3 checkpoint SHA lineage and the complete Controller -> Train -> Verify -> Select -> Parent -> Next Round trace.
- [ ] Run the full local suite with `./.venv/bin/pytest -q`; distinguish CPU contract evidence from the real remote evidence.
- [ ] Write the exact evidence paths and any remaining remote blocker; do not mark the goal complete without a real selected Parent satisfying the full contract.
