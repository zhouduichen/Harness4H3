# Wan2.1-T2V-1.3B real-GPU profile

## Run identity

- Checkpoint digest: `dab2db83db4cd1f9c2a7ddb46e169711d688c84702bf78c589a3d3f3ce24776d` (1,418,996,800 Transformer parameters).
- GPU: NVIDIA L40, `cuda:3`, BF16; PyTorch 2.10.0+cu128, CUDA 12.8, Diffusers 0.40.0.
- Latency pass: synchronized CUDA Events plus host wall time. Prompt text is encoded once per sample; the reported full generation adds text encoding to denoising, VAE decode, and postprocessing.
- Dense 256×256 and Head Gating runs used 2 warmups and 5 repeats. 832×480 frame-count runs used 1 warmup and 3 repeats. All cases used 16 steps, seed 7, and the same red-fox prompt.
- The module and kernel diagnostic pass ran separately from latency measurement. The full raw profile and traces remain at `/data/models/harness4h3-wan-profiling-20261018/corrected-r2/`. The raw `profile.json` SHA-256 is `d3a414c4466f90c5e9e2190916f2b0a9da94a54ae48cd1ed7b8706718835c26a`; selected records are preserved in [wan-gpu-profile-summary.json](wan-gpu-profile-summary.json).

## Dense latency and memory

Times are medians in seconds. Allocated and reserved memory are median CUDA peaks; resident memory is measured after pipeline load. Full generation includes the T5 text encoder.

| Recipe | Full generation | Denoising + VAE + postprocess | Text encoder | Peak allocated | Peak reserved | Resident |
|---|---:|---:|---:|---:|---:|---:|
| 256×256×81 | 8.079 | 8.018 | 0.061 | 14.22 GiB | 14.76 GiB | 13.48 GiB |
| 832×480×17 | 13.800 | 13.719 | 0.081 | 17.73 GiB | 20.13 GiB | 13.49 GiB |
| 832×480×41 | 39.504 | 39.421 | 0.084 | 17.79 GiB | 20.13 GiB | 13.49 GiB |
| 832×480×81 | 97.321 | 97.237 | 0.084 | 17.90 GiB | 20.13 GiB | 13.49 GiB |

At the requested 480p×81 recipe, the Transformer root accumulated 89.9 seconds of inclusive CUDA-event time over 32 forwards (CFG runs two forwards per denoising step). The diagnostic pass measured 6.54 seconds in VAE decode and 0.93 seconds in postprocessing. Text encoding was about 84 ms in the uninstrumented pass. The VAE and postprocess measurements come from a separate instrumented pass and should not be added to produce the uninstrumented end-to-end result.

## Where the compute goes

The table reports inclusive CUDA-event totals from module hooks, in seconds. The categories are nested and can overlap; they are not an additive breakdown of Transformer wall time.

| Recipe | Transformer root | Self-attention | FFN | Cross-attention | VAE decode, diagnostic pass |
|---|---:|---:|---:|---:|---:|
| 256×256×81 | 7.305 | 3.167 | 6.682 | 0.700 | 0.876 |
| 832×480×17 | 12.777 | 6.162 | 10.640 | 0.959 | 1.377 |
| 832×480×41 | 35.815 | 20.534 | 18.507 | 1.847 | 3.309 |
| 832×480×81 | 89.886 | 61.866 | 32.814 | 3.062 | 6.539 |

The optimization priority changes with token count. At 17 frames, FFN hooks are largest; at 41 and 81 frames, self-attention grows past FFN. At 81 frames, self-attention is the clearest target, followed by FFN. Cross-attention and text encoding are smaller. The 81-frame callback intervals were nearly flat at 5.60–5.65 seconds per step; the callback’s first interval includes pipeline and scheduler setup, so first-step comparisons are not pure denoiser timings.

CUDA kernel traces were captured for the 256×256 dense case, 832×480×17 dense case, and 256×256 Head Gating case. Their dominant real CUDA entries include BF16 GEMM kernels and PyTorch FlashAttention. Traces were not requested for the 41- and 81-frame cases; their module and per-step diagnostics are present. The compact JSON summary removes profiler pseudo-events such as `ProfilerStep*` and `Command Buffer Full`.

## Matched Head Gating observation

At 256×256×81, the `weight_l2` Head Gating policy selected 75% of self-attention heads across all 30 blocks. Runtime counters recorded 77,760 selected head-calls out of 103,680 full head-calls over 8,640 calls. Parameter count remained 1,418,996,800.

| Variant | Full generation | Peak allocated | Peak reserved | Resident |
|---|---:|---:|---:|---:|
| Dense | 8.079 s | 14.22 GiB | 14.76 GiB | 13.48 GiB |
| Head Gating 0.75 | 8.114 s | 16.87 GiB | 17.64 GiB | 16.13 GiB |

This repeated matched run was 0.44% slower with Head Gating and used more measured memory. It has no video-quality evaluation, so it is only runtime evidence. The historical 1.47% latency gain came from one prompt, one seed, zero warmups, and one repeat; this profile does not reproduce it as a stable speedup. The observed head-selection count confirms the gate executed, but it did not translate into lower end-to-end latency here.

## What this means for the next experiments

1. Run training-free sparse self-attention first at 832×480×81: attention becomes the largest measured Transformer category at this length.
2. Measure timestep feature caching with matched prompts and seeds; it targets repeated Transformer work across denoising steps, while the current profile shows about 90 seconds in Transformer forwards.
3. Keep FFN pruning as a separate baseline and report its interaction with attention only after the individual methods have matched quality and latency records.
4. Treat any 480p speedup as unproven until the same 832×480×81 video-quality and latency protocol runs for the candidate.

## Limits

The profile uses one prompt and one seed. It measures runtime and memory, not generated-video quality. Hook times are inclusive diagnostics; kernel traces cover only the cases listed above. The 480p profile uses PyTorch 2.10/CUDA 12.8, while the separately paired SVOO pilot uses the remote SparseVideo environment and must be compared against its own dense run.
