# Wan2.1 GPU Profiling Implementation Plan

> **For agentic workers:** Execute this plan inline in the existing Harness4H3 worktree. Keep each milestone independently reviewable.

**Goal:** Produce a reproducible real-CUDA performance profile for Wan2.1-T2V-1.3B that separates Transformer blocks and submodules, denoising steps, text encoding, VAE, and end-to-end generation.

**Architecture:** Extend the existing `profile-wan` / `ProfileReport` path and reuse the Diffusers adapter and Wan video evaluator. Measure uninstrumented latency separately from one PyTorch Profiler trace so profiler overhead does not contaminate reported speed. Store machine-readable profile data beside a concise human report.

**Tech Stack:** Python, PyTorch CUDA Events and Profiler, Diffusers Wan2.1, existing Harness4H3 adapter/evaluator.

## Global Constraints

- Use the latest local `codex/algorithm-discovery-rsi` commit `1290ae7` as the code base; leave that worktree's uncommitted files untouched.
- Use new code and output locations on branch `codex/wan-rsi-profiling`; never overwrite existing experiment roots.
- Reuse Harness4H3's existing adapter, profiler report, evaluator, and registered Head Gating operator.
- Record unavailable values as unavailable; never substitute CPU/Fake timing or FLOP estimates for CUDA measurements.
- Measure the 256×256×81 development recipe and the requested 480p recipe at fixed 16 denoising steps; also measure 480p frame-count scaling at 17, 41, and 81 frames.
- Recheck actual GPU memory/process usage immediately before each remote run; keep GPU0 reserved for the vLLM controller.
- Keep existing model-quality and promotion thresholds unchanged; stage one measures Dense and Head Gating runtime behavior only.

---

### Task 1: Add execution measurements to the existing Wan profile

**Files:**
- Create: `harness4h3/profiler/execution.py`
- Modify: `harness4h3/profiler/report.py`
- Modify: `harness4h3/evolution/evaluator.py`

- [x] Add a versioned execution section to `ProfileReport` for run recipe, component timings, per-block and per-submodule timings, CUDA operator/kernel summaries, per-denoising-step timings, and peak allocated/reserved memory.
- [x] Reuse Wan module paths to label self-attention, cross-attention, Q/K/V projections, attention kernels, output projections, FFN, and individual Transformer blocks.
- [x] Record CUDA Events for the uninstrumented latency pass and keep the PyTorch Profiler trace in a separate diagnostic pass.
- [x] Instrument text encoding, Transformer, VAE decode, and postprocess stages around the existing evaluator pipeline.
- [x] Attribute each record to the recipe, seed, device, dtype, checkpoint digest, and warmup/repeat counts.

### Task 2: Expose the real profile through the existing CLI

**Files:**
- Modify: `harness4h3/cli.py`
- Create: `configs/wan-execution-profile-20261018.yaml`

- [x] Extend `profile-wan` with an execution-profile mode that consumes an existing Wan evaluation recipe and fixed prompt.
- [x] Configure 256×256×81 and 832×480 at 17/41/81 frames, fixed prompt/seed, explicit warmup/repeat counts, and a matched Head Gating variant.
- [x] Persist cumulative JSON after each case and retain the profiler traces under a separate trace directory.

### Task 3: Run the Dense and Head Gating measurements on remote CUDA

**Files:**
- Create: `docs/superpowers/reports/2026-10-18-wan-gpu-profiling.md`

- [x] Recheck remote GPU process ownership and free memory; use L40 CUDA:3 and the new `/data/models/harness4h3-wan-profiling-20261018/corrected-r2/` output root.
- [x] Finish the 480p×81 Dense and matched 256×256 Head Gating cases. All five configured cases are present in the remote profile.
- [x] Summarize inclusive module timings, memory peaks, step variation, and the matched evidence concerning the historical 1.47% result.
- [x] Keep the raw JSON and profiler traces in the remote output root; copy the report and selected measurements into this repository.

### Task 4: Connect measured execution evidence to RSI

**Files:** `harness4h3/cli.py`, `harness4h3/evolution/loop.py`, `harness4h3/evolution/agent.py`

- [x] Require complete real-CUDA measurements matching the active checkpoint and device before they guide a search.
- [x] Pass compact measured block, attention, FFN, timestep, memory, and kernel evidence into AlgorithmAgent.
- [x] Rank bounded block probes by measured CUDA time when available, and record the evidence source in search diagnostics.
- [x] Require an explicit complete marker and consistent case counts before a profile can guide RSI.
- [x] Compare a real SVOO generation against the matched Dense parent using the same prompt, seed, resolution, and evaluator revision.
- [ ] Run two real Wan RSI rounds with runtime benchmark results seeded into ExperienceStore; compare the second-round algorithm with its parent using measured evidence. A pilot attempt stopped before round 1 because the execution-profile checkpoint did not match the active Wan parent; no evolution result was produced.

### Task 5: Add the six-arm paper ablation protocol

**Files:** `harness4h3/evolution/protocol.py`, `harness4h3/operators/wan/search_space.py`, `tools/build_evolution_protocol_configs.py`

- [x] Preserve the four-arm validator and add Fixed, Coding Agent, Agent+Memory, fixed-space RSI, frozen-expanded RSI, and online search-space RSI arms.
- [x] Expose the existing timestep-attention operator in a distinct frozen expanded catalog; persist the default catalog for fixed-space rounds.
- [x] Generate six matched Wan configs and a stable protocol fingerprint.
- [ ] Run all arms at matched budgets and report quality, GPU latency, peak memory, parameter count, search cost, and algorithm-discovery results.

### Task 6: Reproduce cross-step and combined runtime compression

**Files:**
- Create: `docs/superpowers/reports/2026-10-18-wan-runtime-compression.md`
- Reuse: `configs/wan-runtime-baselines-20261018/`

- [x] Reuse the official SparseVideo SVOO implementation and evaluate it on the 832×480×81 recipe.
- [x] Save the paired Dense/SVOO results and record the method-dispatch diagnostic gap.
- [x] Calibrate Diffusers MagCache on the same recipe and save its emitted per-step ratios.
- [x] Evaluate calibrated MagCache on the same recipe and preserve its raw result.
- [x] Evaluate SVOO+MagCache with the same prompt and seed (39.895 s median, 21.735 GiB peak allocated; one prompt/seed, console-derived summary recorded locally).
- [x] Check whether BWCache has a supported Wan path; the official repository lists Wan 2.1 support as TODO, so no Wan reproduction is currently available.

### Completion check

- [x] Confirm the completed profile identifies real CUDA execution and contains no Fake measurements.
- [x] Confirm timing sections distinguish inclusive module time, CUDA kernel time, Transformer time, and full video-generation time.
- [x] Report measured and unmeasured cases explicitly; do not infer an unmeasured 480p result.
