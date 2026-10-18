# Wan2.1 runtime compression baseline: Dense and SVOO

**Date:** 2026-10-18
**Checkpoint:** Wan2.1-T2V-1.3B, 1,418,996,800 transformer parameters
**Device:** NVIDIA L40, CUDA:3
**Paired evaluation environment:** PyTorch 2.7.1+cu129, Diffusers 0.39.0, BF16
**Recipe:** 832×480, 81 frames, 16 denoising steps, seed 7
**Prompt:** “A cinematic shot of a red fox running through fresh snow.”

## Measurement

Both arms use the same Harness4H3 evaluator revision, cached prompt embeddings, and timing method (`denoising+vae_decode_with_case_cached_prompt_embeddings`). Each arm has one warmup and three timed generations. Text encoding is excluded; denoising and VAE decode are included. Quality was scored with the CLIP/temporal evaluator using four sampled frames, and the visual-structure gate ran on the full 81-frame output.

The detailed module profile was collected in a separate profiling environment (PyTorch 2.10.0+cu128, Diffusers 0.40.0). Its module breakdown identifies bottlenecks; compare the Dense/SVOO/MagCache table only within the paired evaluation environment listed above.

| Method | Aggregate quality | Semantic | Temporal | Motion | Median latency | Peak allocated memory | Visual gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dense | 0.67331 | 0.66164 | 0.95156 | 0.19851 | 105.996 s | 21.455 GiB | Pass |
| SparseVideo SVOO | 0.68054 | 0.66283 | 0.97671 | 0.21217 | 65.129 s | 21.735 GiB | Pass |
| Diffusers MagCache | 0.67723 | 0.66533 | 0.97191 | 0.17118 | 62.947 s | 21.455 GiB | Pass |
| SVOO + MagCache | 0.67153 | 0.65691 | 0.96700 | 0.18294 | 39.895 s | 21.735 GiB | Pass |

SVOO was 1.63× faster by median latency (38.6% lower) in this paired run. Peak allocated memory rose by 0.280 GiB (1.3%). The model remained at 1.419B parameters; this method does not add learned weights. SVOO’s aggregate score was 0.00723 above Dense in this single prompt/seed. That is a sample result, not evidence of quality improvement or statistical equivalence.

MagCache was 1.68× faster than Dense by median latency (40.6% lower) with peak memory effectively unchanged (−0.001 GiB). Its aggregate quality score was 0.00392 above Dense, while the motion subscore was 0.02732 lower. Both methods have only one prompt/seed of quality evidence. The runtime record confirms the MagCache hook and calibrated configuration; it does not currently export a per-step cache-skip count, so skip frequency remains unmeasured.

The combined SVOO+MagCache run completed on the same 832×480×81, 16-step prompt/seed recipe and passed the visual gate. Its median latency was 39.895 s (2.66× faster than Dense, 62.4% lower), with 21.735 GiB peak allocated memory. Aggregate quality was 0.00178 below Dense; semantic, temporal, and motion subscores were respectively 0.00473 lower, 0.01544 higher, and 0.01557 lower. This is one prompt/seed and does not establish quality equivalence. The combined run used evaluator v10, which snapshots the live SVOO handle after generation; the earlier standalone SVOO artifact used the prior evaluator behavior, so the two result records are not from an identical source snapshot. The combined raw evaluation JSON was not copied into this worktree; these combined metrics were transcribed from the completed run output, and per-call dispatch/cache-skip counters are not reported here.

## SVOO reproduction details

The run reused official SparseVideo commit `8cd70fd3315a32bc9ac6e6ad37f57ad7cefeda0d`, method `svoo`, with its Wan 1.3B T2V sparsity profile. Harness4H3 installed SparseVideo processors on all 30 self-attention layers. The run log reports dense warmup for step 1 and sparse attention active from step 2 of 16. The generated 832×480×81 video passed finite-pixel and visual-structure checks.

The standalone exported method diagnostic reports `total_calls: 0` and empty backend/dispatch counters despite the installed processor classes and the run-time sparse activation log. Inspection showed the earlier evaluator captured `handle.summary()` immediately after installation, before generation; those zero counters therefore do not show that the sparse backend was inactive. Evaluator v10 snapshots the live handle after generation. The combined v10 run completed, but its exact post-run counter values were not retained in the local report artifacts, so dispatch counts remain unavailable here.

## Interpretation and next runs

The measured profile identifies self-attention (61.866 s inclusive) and FFN (32.814 s inclusive) as the largest Transformer submodules in the separate 832×480×81 profile. SVOO targets the largest single component. Its memory use is slightly higher, so it is a latency result rather than a memory optimization.

MagCache calibration completed on the same checkpoint, scheduler, and 16-step recipe. Diffusers emitted two 16-step ratio traces during the evaluator generation; the hook does not identify why there are two, so the saved calibration artifact preserves both and records their per-step arithmetic mean as the benchmark vector. The run uses threshold 0.12, maximum skip 4, and retention 0.2, matching the official Wan2.1 T2V 1.3B 480p example. The first evaluation attempt reached video serialization and exited because that process lacked `cv2`; the ratio vectors were emitted before the failure. After adding the isolated OpenCV package path, the MagCache evaluation succeeded. The combined SVOO+MagCache evaluation also completed; the local record above is a concise transcription because its raw result artifact was not copied into this worktree. The RSI pilot did not produce a valid evolution round: the run stopped at the active-parent/checkpoint compatibility check, so no parent/offspring comparison is available.

BWCache has no Wan result in this campaign. Its official repository lists “Add Wan 2.1 support” as a TODO, so there is no mature Wan path to reuse without undertaking a new adaptation.

## Reproducibility artifacts

- Dense result: [2026-10-18-wan-runtime-dense-evaluation.json](2026-10-18-wan-runtime-dense-evaluation.json)
- SVOO result: [2026-10-18-wan-runtime-svoo-evaluation.json](2026-10-18-wan-runtime-svoo-evaluation.json)
- SVOO console log: [2026-10-18-wan-runtime-svoo-console.log](2026-10-18-wan-runtime-svoo-console.log)
- SVOO config: [svoo.yaml](../../../configs/wan-runtime-baselines-20261018/svoo.yaml)
- MagCache calibration config: [magcache-calibrate.yaml](../../../configs/wan-runtime-baselines-20261018/magcache-calibrate.yaml)
- MagCache calibration data: [2026-10-18-wan-runtime-magcache-calibration.json](2026-10-18-wan-runtime-magcache-calibration.json)
- MagCache calibration console: [2026-10-18-wan-runtime-magcache-calibration-console.log](2026-10-18-wan-runtime-magcache-calibration-console.log)
- MagCache evaluation config: [magcache.yaml](../../../configs/wan-runtime-baselines-20261018/magcache.yaml)
- MagCache result: [2026-10-18-wan-runtime-magcache-evaluation.json](2026-10-18-wan-runtime-magcache-evaluation.json)
- MagCache console: [2026-10-18-wan-runtime-magcache-console.log](2026-10-18-wan-runtime-magcache-console.log)
- Combined evaluation config: [svoo-magcache.yaml](../../../configs/wan-runtime-baselines-20261018/svoo-magcache.yaml)
- Combined evaluation result: [console-derived summary](2026-10-18-wan-runtime-svoo-magcache-summary.json); raw JSON was not copied locally.
- Complete profile report: [2026-10-18-wan-gpu-profiling.md](2026-10-18-wan-gpu-profiling.md)
- Official references: [SparseVideo repository](https://github.com/Mutual-Luo/SparseVideo), [SVOO paper](https://arxiv.org/abs/2603.18636), [Diffusers MagCache hook](https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/mag_cache.py), [MagCache Wan instructions](https://github.com/Zehong-Ma/MagCache/blob/main/MagCache4Wan2.1/README.md), [BWCache repository](https://github.com/hsc113/BWCache)
