# M6 Cross-Layer Runtime Memory Optimization Design

## Scope

M6 starts from the validated `M0001` NVFP4 candidate and targets the remaining strict RTX 5080 peak-VRAM constraint. It does not change model weights, controller training, Harness Evolution, surrogate modeling, kernels, compilers, or evaluator thresholds. Each runtime experiment creates an immutable child branch and keeps the parent checkpoint unchanged.

## Architecture

Runtime-memory operators are registered beside existing model operators. An operator validates one low-risk intervention, clones the parent `ModelState`, and records a serializable `runtime_policy` in `runtime_state`. The ComfyUI benchmark runner consumes that policy to make concrete workflow changes; unsupported node capabilities fail explicitly instead of silently claiming an optimization.

The first registry contains three single-intervention operators:

1. `runtime_offload`: enable low-VRAM/offload controls on model and LoRA nodes.
2. `vae_tiling`: replace the VAE decode node with a tiled decoder when the workflow exposes a compatible node; tile size and overlap are fixed operator arguments.
3. `inference_chunking`: set a compatible H3 video node's chunk-size input when present.

The controller receives these schemas and the read-only validated Design Gene in its context. It must diagnose peak memory as the sole remaining hard violation, choose one operator, and preserve the M5.5 recipe. The operator does not make TargetProfile feasibility true; only the M6 evaluator gate does.

## Acceptance

For every M6 candidate, the independent evaluator records per-run peak VRAM samples and aggregates mean, median, min, max, standard deviation, and 95% CI. The strict gate uses `peak_memory_max_gb <= TargetProfile.max_peak_memory_gb`, not the mean. M6 passes only when generation/decode are valid, black-frame rate is zero, quality drop is at most `0.05`, M5.5 model-size reduction remains intact, latency remains materially below the INT8 parent, and peak-VRAM max is at most 16GB on dev and held-out tasks. A failed branch is recorded and does not mutate the active parent.

## Failure handling and experience

Missing workflow inputs, incompatible node classes, backend failures, evaluator failures, and peak-memory violations remain explicit failure/decision records. Runtime policies are read-only Design Gene inputs; they do not automatically alter prompts, operator priors, search policy, or source code. A successful M6 branch may later be marked `transferred` only after another TargetProfile or hardware run.

