# A1-T0 H3 training reconnaissance — 2026-09-11

Status: **blocked before trainer implementation**.

This document records only source-grounded facts. No training loss, trainer,
mock, fake checkpoint, or Harness4H3-v1.0 change was introduced.

## Reconnaissance scope and evidence policy

The local repository contains the frozen Harness, a ComfyUI HTTP client, and
checkpoint-header inspectors. It does not contain the H3 PyTorch model class or
the ComfyUI custom-node source that implements H3 execution. The Windows host
`autoresearch-5080` was offline during this reconnaissance:

- Tailscale reported `Online=false`; last seen was 2026-09-10 14:53:55 +08:00.
- SSH to `autoresearch-5080` timed out.
- Therefore no current remote source file, Python environment, or runtime
  introspection result is claimed below.

Confidence levels:

- **high** — directly observed in the local source/workflow.
- **medium** — historical evidence already recorded in the repository, not
  revalidated during this run.
- **unknown** — requires the remote ComfyUI/custom-node source or runtime.

## A. Actual model implementation

| Fact | Source / symbol / path | Observed | Confidence |
|---|---|---|---|
| No trainable H3 class exists in this repository | [`harness4h3/model/minimax_h3.py:1-10`](../../harness4h3/model/minimax_h3.py) | The module explicitly provides compatibility imports for the ComfyUI artifact backend and exports `MiniMaxH3Adapter`; it does not define a model class. | high |
| The actual H3 model class is not identified | Remote ComfyUI custom-node source | Host offline; source could not be read. | unknown |
| Local normalized model state is metadata only | [`H3Inspector.inspect`](../../harness4h3/h3/inspector.py#L77-L156) | The inspector returns `weights_loaded=False` and parses tensor metadata without loading weights. | high |

Required result still missing: actual class name, module path, constructor, and
the parameter/module tree.

## B. Actual loader and checkpoint format

| Fact | Source / symbol / path | Observed | Confidence |
|---|---|---|---|
| Local safetensors handling is header inspection only | [`inspect_safetensors`](../../harness4h3/h3/checkpoint.py#L61-L115) | Reads the 8-byte header length and JSON tensor metadata; it does not materialize tensors or call a model loader. | high |
| Local GGUF handling is also inspection-only | [`H3Inspector.inspect`](../../harness4h3/h3/inspector.py#L90-L96) | Selects `inspect_gguf` for `.gguf`; no runtime model construction occurs. | high |
| ComfyUI workflow asks node `127` to load the diffusion model | [`examples/workflow_api.json:4`](../../examples/workflow_api.json#L4) | `UNETLoader(unet_name=minimax_h3_fl2va_pruned_nvfp4.safetensors, weight_dtype=default)`. | high |
| Actual `UNETLoader` implementation and loader path | Remote ComfyUI/custom nodes | Not accessible while host is offline. | unknown |
| Parent checkpoint container | [`configs/models/minimax_h3_rtx5080.yaml`](../../configs/models/minimax_h3_rtx5080.yaml#L7-L17) | Configured artifact is `.safetensors`; this config is a deployment manifest, not a loader implementation. | high |

Historical header evidence for the configured NVFP4 file records 1,132 tensors,
11,681,874,744 tensor-header parameter elements, and a complete 7/7 H3 tensor
signature. This is recorded in
[`2026-09-09-windows-rtx5080.md`](2026-09-09-windows-rtx5080.md#controller-and-inspector),
but it is not proof that the current remote loader can train the checkpoint.

## C. Actual forward path

The local workflow gives the following graph, but not the Python implementation
behind the custom nodes:

```text
UNETLoader(127)
  -> MiniMaxH3TurboLoRA(134)
  -> PathchSageAttentionKJ(137)
  -> CFGGuider(126)
  -> BasicScheduler(124) + MiniMaxH3TurboSampler(135)
  -> SamplerCustomAdvanced(125)
  -> VAEDecode(122)
  -> video output
```

Conditioning enters through:

```text
CLIPLoaderGGUF(128, type=minimax)
  -> MiniMaxH3ImageToVideo(131, prompt/width/height/length)
  -> positive conditioning(126)
  -> ConditioningZeroOut(136)
  -> negative conditioning(126)
```

Evidence: [`examples/workflow_api.json:2-17`](../../examples/workflow_api.json#L2-L17),
and [`H3BenchmarkRunner._workflow`](../../harness4h3/benchmark/h3.py#L149-L175),
which only rewrites workflow inputs and submits the workflow to ComfyUI.

The following are **not confirmed** without the custom-node source:

- model `forward(...)` signature;
- tensor shapes and dtypes at each forward input;
- whether conditioning is token, embedding, packed latent, or a custom object;
- timestep versus sigma representation;
- latent layout and packing;
- model output tensor semantics;
- whether the sampler consumes epsilon, v, flow/velocity, x0, or another
  parameterization.

## D. Actual model I/O

| Item | Current evidence | Status |
|---|---|---|
| Input model artifact | `UNETLoader` receives the filename and `weight_dtype=default` | Confirmed at workflow boundary; internal loader behavior unknown. |
| Text encoder | `CLIPLoaderGGUF` loads `qwen3vl-32B-MiniMax-H3-Q2_K.gguf` with `type=minimax` | Confirmed at workflow boundary; encoding output type unknown. |
| Video VAE | `VAELoader` nodes load `minimax_h3_video_vae_fp8mix.safetensors` and `minimax_h3_audio_vae_bf16.safetensors` | Confirmed at workflow boundary; encode/decode tensor contract unknown. |
| Latent input | `MiniMaxH3ImageToVideo` output socket `1` feeds `SamplerCustomAdvanced.latent_image` | Graph edge confirmed; latent shape/dtype unknown. |
| Model output | Sampler consumes the model through `CFGGuider` | Graph edge confirmed; parameterization unknown. |
| Save/load roundtrip | ComfyUI `SaveVideo` saves generated video; no model checkpoint save path is present | Model checkpoint save/load roundtrip unknown. |

## E. Output parameterization

**Unknown.** No local source defines the H3 denoising objective, scheduler
conversion, sigma/timestep mapping, or model-output interpretation. The names
`BasicScheduler`, `MiniMaxH3TurboSampler`, and `SamplerCustomAdvanced` are not
enough evidence to classify the output as epsilon, v, flow/velocity, x0, or
another parameterization.

No training loss is constructed as a result.

## F. Candidate training loss construction

**Blocked and intentionally not implemented.** A valid loss requires the actual
H3 forward contract plus the sampler/training parameterization. At minimum the
reconnaissance must establish:

1. how clean/noisy latent pairs are produced;
2. how sigma/timestep is encoded and passed;
3. which conditioning object is accepted;
4. what model output represents;
5. which target is used by the original implementation; and
6. which dtype/device/autocast constraints are required.

Until those are sourced from the vendor/upstream implementation, any loss would
be a guess and would violate the A1-T0 requirements.

## G. Trainable parameter candidates

**Unknown.** No model instance or `named_parameters()` output was available.
Consequently, no layer name is being guessed and no parameter is being marked
`requires_grad=True`.

The only safe future procedure is to instantiate the actual loaded model,
enumerate `named_parameters()`, record dtype/device/numel/`requires_grad`, and
select a subset only after confirming that those modules participate in the
real forward path and are compatible with the checkpoint representation.

## H. Checkpoint save strategy

**Unknown.** The local code can inspect a safetensors file but does not save a
model checkpoint. The actual `UNETLoader` and vendor model implementation must
be inspected to determine whether the supported roundtrip is:

- a full safetensors state dict;
- a quantized state dict with auxiliary scales/metadata;
- a base checkpoint plus adapter;
- or another ComfyUI-specific format.

The future smoke test must save through that real supported path, reload the
child with the same loader, and compare named tensors—not merely file bytes.

## I. Memory feasibility on RTX 5080 Laptop

Historical, not current, evidence records a Windows RTX 5080 Laptop GPU with
17,094,475,776 bytes of CUDA VRAM and ComfyUI 0.34.0 in
[`2026-09-09-windows-rtx5080.md`](2026-09-09-windows-rtx5080.md#real-comfyui-probes).
The configured NVFP4 parent is about 12.53 GB on disk. These facts do not
establish training feasibility: optimizer state, gradients, activation memory,
dequantization behavior, and the actual model loader are still unknown.

The current local virtual environment has none of the relevant training
packages:

```text
torch=False
safetensors=False
transformers=False
diffusers=False
accelerate=False
bitsandbytes=False
```

The project dependency list contains only PyYAML and OpenCV for runtime
purposes: [`pyproject.toml:11-14`](../../pyproject.toml#L11-L14).

## J. Blockers

1. `autoresearch-5080` is currently offline (`Online=false`); SSH timed out.
2. The actual ComfyUI custom-node/vendor source is unavailable locally and
   cannot be inspected remotely.
3. The real H3 model class, loader, forward signature, loss parameterization,
   trainable modules, and checkpoint save roundtrip are therefore unknown.
4. The local environment has no PyTorch or model-training dependencies.
5. A1-T0 cannot proceed without these facts; implementing a guessed trainer
   would violate the requested source-grounded constraint.

## K. Exact next implementation step

When the host is online, perform read-only source discovery before writing any
trainer:

1. enumerate `D:\ComfyUI\custom_nodes` and the ComfyUI core model-loading
   modules;
2. locate the definitions and registrations for `UNETLoader`,
   `MiniMaxH3ImageToVideo`, `MiniMaxH3TurboSampler`,
   `MiniMaxH3TurboLoRA`, and `PathchSageAttentionKJ`;
3. trace loader → model instance → sampler/denoising wrapper →
   `model.forward(...)`;
4. run a non-mutating Python introspection script to record signatures,
   parameter names, dtypes, shapes, `requires_grad`, and runtime versions;
5. locate any upstream/original H3 training code and reuse its data/target
   construction;
6. only then implement the standalone smoke-test trainer outside the Harness.

Until steps 1–5 produce source and runtime evidence, the correct status is
**A1-T0 blocked; trainer not implemented**.
