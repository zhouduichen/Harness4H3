# A1-T0 H3 training reconnaissance — 2026-09-11

Status: **reconnaissance completed; trainer implementation remains blocked**.

This document records source-grounded facts from the reachable Windows host.
No `h3_recovery_finetune` trainer, training loss implementation, mock trainer,
fake checkpoint, or Harness4H3-v1.0 change was introduced.

## Reconnaissance scope and evidence policy

Remote host and source revision:

- Host: `autoresearch-5080` (SSH reachable).
- ComfyUI root: `D:\ComfyUI`.
- ComfyUI version: `0.34.0` from `D:\ComfyUI\comfyui_version.py`.
- Git remote: `https://github.com/comfyanonymous/ComfyUI.git`.
- Git revision: `e7051b03`.
- The ComfyUI HTTP service was **not running** during this run: no Python
  process matching ComfyUI and no listener on TCP `8188`. Therefore no
  runtime model instance, `named_parameters()` dump, real forward probe, or
  benchmark was performed.

Confidence levels:

- **high** — directly observed in the current remote source, current
  checkpoint safetensors header, or current Python environment.
- **medium** — source-grounded inference across two current source paths,
  explicitly identified as an inference.
- **unknown** — requires starting ComfyUI or a separate full model-load
  probe; no implementation decision is made from it.

## A. Actual model implementation

### A.1 Active ComfyUI model class

| Fact | Source / symbol / code path | Observed | Confidence |
|---|---|---|---|
| Actual active DiT class | `D:\ComfyUI\comfy\ldm\minimax\model.py:467-508`, `MiniMaxH3Model.__init__` | The class is `comfy.ldm.minimax.model.MiniMaxH3Model`; it constructs `video_patch_proj`, `audio_patch_proj`, `condition_proj`, optional `adaln_t_table`/time embedder, `token_refiner`, 50 `DiTBlock`s, and `final_layer`. | high |
| ComfyUI model wrapper | `D:\ComfyUI\comfy\model_base.py:2136-2139`, `MiniMaxH3.__init__` | `model_base.MiniMaxH3` wraps the actual class as `unet_model=comfy.ldm.minimax.model.MiniMaxH3Model`, with `model_type=ModelType.FLOW_AV`. | high |
| Model selection | `D:\ComfyUI\comfy\supported_models.py:960-986`, `MiniMaxH3` | H3 is selected by `unet_config.image_model == "minimax_h3"`; it uses shifts video `12.0`, audio `3.0`, latent format `MiniMaxH3AV`, and supports inference dtypes `torch.bfloat16` and `torch.float32`. | high |
| Detection rule | `D:\ComfyUI\comfy\model_detection.py:390-415` | H3 is detected from `video_patch_proj.weight` plus `audio_patch_proj.weight`; architecture dimensions are derived from checkpoint tensors, not guessed from a class name. | high |

The constructor defaults and the values inferred from the current checkpoint
are:

```text
hidden_size              = 5376
num_layers               = 50
token_refiner_num_layers = 2
num_attention_heads      = 56
attention_head_dim       = 128
ffn_hidden_size          = 14336
latents_dim              = 24
audio_latents_dim        = 32
patch_size               = (1, 2, 2)
text_dim                 = 5120
sigma_shift_video        = 12.0
sigma_shift_audio        = 3.0
adaln_curve_grid         = 1025
time_embed_dim           = 8
rope_inv_freq_len        = 16
```

The `adaln_curve_grid=1025` and `time_embed_dim=8` values are observed from
the `adaln_t_table` tensor in the current checkpoint. The checkpoint does not
contain `time_embedder.proj_in.weight` or `time_embedder.proj_out.weight`, so
this is the curve-form model, not the full time-embedder branch.

### A.2 Local repository distinction

The local Harness file
[`harness4h3/model/minimax_h3.py:1-10`](../../harness4h3/model/minimax_h3.py)
contains only compatibility imports and an HTTP adapter. It is not the H3
model implementation. The actual model implementation is on the Windows
ComfyUI host above.

## B. Actual loader and checkpoint format

### B.1 Loader path

The observed active path is:

```text
workflow UNETLoader
  -> D:\ComfyUI\nodes.py:982-1005, UNETLoader.load_unet
  -> comfy.sd.load_diffusion_model
  -> D:\ComfyUI\comfy\sd.py:2355-2362
  -> load_diffusion_model_state_dict
  -> D:\ComfyUI\comfy\utils.py:158-203, load_torch_file
  -> safetensors.safe_open(..., framework="pt")
  -> D:\ComfyUI\comfy\model_detection.py:390-415
  -> D:\ComfyUI\comfy\supported_models.py:960-986
  -> D:\ComfyUI\comfy\model_base.py:2136-2139
  -> MiniMaxH3Model
  -> ModelPatcher.load_model_weights
```

`UNETLoader(weight_dtype="default")` does not itself force a training dtype.
For this model, `comfy.sd.load_diffusion_model_state_dict` selects an
inference dtype from the supported dtypes and device capabilities at
`D:\ComfyUI\comfy\sd.py:2322-2336`; an explicit node dtype only overrides that
selection at `nodes.py:993-1004`.

### B.2 Current parent checkpoint header

The current file is:

```text
D:\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_nvfp4.safetensors
size: 12,528,636,800 bytes
format: safetensors
metadata: {}
tensor records: 1,132
```

The header was read without materializing tensor payloads. Representative
records are:

```text
video_patch_proj.weight       F32   [5376, 96]
audio_patch_proj.weight       F32   [5376, 32]
condition_proj.weight         BF16  [5376, 5120]
final_layer.video_out.weight  F32   [96, 5376]
final_layer.audio_out.weight  F32   [32, 5376]
blocks.0.attn.q_norm.weight   BF16  [128]
blocks.0.attn.qkv_proj.weight U8   [21504, 2688]
blocks.0.mlp.fc1.weight       U8   [28672, 2688]
adaln_t_table                 F32   [1025, 8]
rope.inv_freq                 F32   [16]
```

The current checkpoint's quantized block records also include, for example:

```text
blocks.0.attn.qkv_proj.comfy_quant  U8   [19]
blocks.0.attn.qkv_proj.weight_scale F8_E4M3 [21504, 336]
blocks.0.attn.qkv_proj.weight_scale_2 F32 []
```

This is ComfyUI's `nvfp4` representation, not a normal BF16/FP16 state dict.
The quantized loader is implemented at
`D:\ComfyUI\comfy\ops.py:1111-1231`: it consumes `comfy_quant` and scale
records, wraps the weight as a `QuantizedTensor`, and registers the loaded
weight with `requires_grad=False`.

`D:\ComfyUI\comfy\utils.py:205-209` provides generic
`save_torch_file()` using safetensors, but the active inference path does not
provide a child-checkpoint training/save roundtrip for this quantized model.

## C. Actual forward path

### C.1 End-to-end call chain

The local workflow confirms the graph boundary:

```text
UNETLoader(127)
  -> MiniMaxH3TurboLoRA(134)
  -> CFGGuider(126)
  -> BasicScheduler(124) + MiniMaxH3TurboSampler(135)
  -> SamplerCustomAdvanced(125)
  -> VAEDecode(122)
  -> CreateVideo/SaveVideo
```

Conditioning enters through:

```text
CLIPLoaderGGUF(128, type=minimax)
  -> MiniMaxH3ImageToVideo(131)
  -> positive/negative conditioning
  -> CFGGuider(126)
```

Evidence: [`examples/workflow_api.json:2-17`](../../examples/workflow_api.json#L2-L17).
The H3-specific model path after the sampler invokes the following current
ComfyUI code:

1. `D:\ComfyUI\comfy\samplers.py:466-490` batches the model call inside
   `with torch.no_grad()`.
2. `D:\ComfyUI\comfy\model_base.py:204-253`,
   `BaseModel.apply_model`/`_apply_model`, receives `x` and sampler `sigma`,
   casts the model input, maps `sigma` to `timestep=sigma*1000`, prepares
   `context`, and calls `self.diffusion_model(...)`.
3. `D:\ComfyUI\comfy\model_base.py:2136-2139` supplies
   `MiniMaxH3Model` as `diffusion_model`.
4. `D:\ComfyUI\comfy\ldm\minimax\model.py:553-577`,
   `MiniMaxH3Model.forward`, handles the audio schedule and calls `_forward`.
5. `D:\ComfyUI\comfy\ldm\minimax\model.py:579-758`, `_forward`, constructs
   the packed sequence, runs the DiT blocks, unpatchifies video, unpacks
   audio, and returns the two model outputs.
6. The optional Turbo custom node also puts its sampler in
   `D:\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py:105-165`
   under `@torch.no_grad()`. Its current native schedule calls
   `model(x, sigmas[i] * s_in, **extra_args)` at lines 121-125 and updates
   with `(x - denoised) / sigma`.

### C.2 Forward signature and inputs

The active ComfyUI signature is directly defined at
`D:\ComfyUI\comfy\ldm\minimax\model.py:553`:

```python
forward(
    self,
    x,
    timestep,
    context,
    transformer_options={},
    minimax_payload=None,
    denoise_mask=None,
    audio_denoise_mask=None,
    **kwargs,
)
```

Observed contracts from the body:

| Input | Source / symbol | Observed shape and dtype behavior | Confidence |
|---|---|---|---|
| `x[0]` video | `model.py:579-590`, `patchify_video` at `42-49` | Video latent is `[B, 24, T, H, W]`; batch must be exactly `1`; padded to `(1,2,2)` patches; rows become `[B*T*H/2*W/2, 96]` before projection. | high |
| `x[1]` audio | `model.py:579-590`, `pack_audio` at `59-67` | Audio latent is `[B, 32, 2, T]`; channel-major rows become `[2*T, 32]`. | high |
| `context` | `model.py:587`, `690-694`; `model_base.py:218-225` | `c_crossattn` is cast to the selected inference dtype; `context[0]` is text hidden states. The model accepts Qwen states with last dimension `5120`, then projects/refines to `5376`, or accepts already-refined `5376` states. | high |
| `timestep` | `model.py:599-611` | The active BaseModel passes `sigma*1000`; H3 divides by `1000` to recover video sigma, computes `t_video=1-sigma_video`, and derives audio sigma/timestep using shifts `12.0` and `3.0`. | high |
| `minimax_payload` | `model_base.py:2164-2213` | Contains text tags, keyframes/refs, condition latents, noise augmentation, seed, audio scale, and a prebuilt `PackedLayout`. | high |

The current text encoder implementation is
`D:\ComfyUI\comfy\text_encoders\minimax.py`:

- lines `1-19`: Qwen3-VL-32B, truncated at LM layer 50; hidden states are
  unnormalized, and vision-pad/text positions receive H3 token tags;
- lines `85-103`: `MiniMaxQwen3VL.forward` records token tags and delegates to
  the Qwen3-VL implementation;
- lines `106-127`: `MiniMaxH3ClipModel` and `MiniMaxH3TEModel`;
- lines `136-202`: raw prompt/vision tokenization and H3 presentation format.

There is no evidence in the active path that the model consumes a generic
Diffusers `Transformer3DModel` signature. The installed Diffusers class is a
separate implementation, not the class returned by ComfyUI's active loader.

## D. Actual model I/O and dependencies

### D.1 Latents and VAE

| Component | Source / symbol | Observed contract | Confidence |
|---|---|---|---|
| Video latent format | `D:\ComfyUI\comfy\latent_formats.py:621-627`, `MiniMaxH3Video` | 24 channels, 3D latent, spatial downscale 16, temporal downscale 4, scale factor 1.0. | high |
| Video VAE | `D:\ComfyUI\comfy\ldm\minimax\vae.py:324-347`, `672-710`, `MiniMaxH3VideoVAE.encode/decode` | Input video `[B,3,T,H,W]`; normalized latent `[B,24,T_lat,H/16,W/16]`; decode returns video pixels. | high |
| Audio latent format | `D:\ComfyUI\comfy\latent_formats.py:657-672`, `MiniMaxH3AV` | Packed format uses 24 video channels and 32 audio channels; audio has 2 stereo channels. | high |
| Audio VAE | `D:\ComfyUI\comfy\ldm\minimax\audio_vae.py:374-440`, `MiniMaxH3AudioVAE` | Stereo waveform `[B,2,L]` at 32 kHz maps to normalized `[B,32,2,T]` at 40 latent frames/sec; no posterior sampling, mean is used. | high |
| Text encoder | `D:\ComfyUI\comfy\text_encoders\minimax.py:1-19`, `106-127` | Qwen3-VL-32B presentation; active checkpoint file is `D:\ComfyUI\models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`. | high |

The current workflow loads
`minimax_h3_video_vae_fp8mix.safetensors` and
`minimax_h3_audio_vae_bf16.safetensors`; the VAE source classes above are
ComfyUI's native implementations.

### D.2 Dtype and quantization handling

The current Windows environment has:

```text
Python:       3.11.9
PyTorch:      2.11.0+cu128
Torch CUDA:   12.8
CUDA usable:  True
GPU:          NVIDIA GeForce RTX 5080 Laptop GPU
VRAM:         17,094,475,776 bytes
safetensors:  installed
transformers: installed
diffusers:    installed
accelerate:   absent
bitsandbytes: absent
```

H3's `video_patch_proj`, `audio_patch_proj`, and `final_layer` are constructed
as float32 operations in `model.py:488-508`; `condition_proj` and the DiT
blocks use the selected model dtype. The NVFP4 checkpoint stores the large
attention/MLP matrices as U8 plus ComfyUI scale/config tensors. The actual
quantized load path is `comfy/ops.py:1111-1231`, not a generic
`bitsandbytes` loader.

## E. Output parameterization

**Confirmed as a rectified-flow data-ward velocity, with a ComfyUI sign
adapter.**

Evidence:

1. `D:\ComfyUI\comfy\ldm\minimax\model.py:1-15` documents the H3 timestep
   convention and two modality shifts.
2. `D:\ComfyUI\comfy\ldm\minimax\model.py:741-758` returns
   `[-video_out, -audio_out]` from the diffusion model.
3. `D:\ComfyUI\comfy\model_base.py:144-149` maps `FLOW_AV` to `CONST` plus
   `ModelSamplingAV`; `model_sampling.py:86-97` converts the Comfy model
   output to a denoised estimate as `model_input - sigma * model_output`.
4. The installed upstream Diffusers reference
   `D:\ComfyUI\.venv\Lib\site-packages\diffusers\schedulers\scheduling_minimax_h3.py:15-35`
   and `223-235` states that H3 predicts data-ward velocity and uses
   `x0 = x_t + sigma * v`.

Therefore the raw H3 target is the data-ward velocity
`v = (x0 - x_t) / sigma` for each modality. The active ComfyUI adapter returns
the negative of that raw velocity so its generic `CONST` wrapper produces the
same denoised estimate. This sign must be preserved if a future trainer calls
the active ComfyUI model path.

This is not epsilon prediction, standard v-prediction, or x0 prediction.

## F. Candidate training loss construction

The source supports the following **candidate**, but it has not been
implemented or runtime-validated:

```text
video: x_t_v = (1 - sigma_v) * z0_v + sigma_v * noise_v
       target_v = (z0_v - x_t_v) / sigma_v

audio: sigma_a = time_shift_sigma(sigma_v, 12.0, 3.0)
       x_t_a = (1 - sigma_a) * z0_a + sigma_a * noise_a
       target_a = (z0_a - x_t_a) / sigma_a
```

The rectified-flow interpolation is directly documented in the installed
Diffusers scheduler at `scheduling_minimax_h3.py:193-221`. The active H3 model
uses the video sampler sigma as its input and derives the shifted audio sigma
at `comfy/ldm/minimax/model.py:599-611`. A future raw-DiT loss would compare
the model's positive data-ward velocity to these targets; a loss through the
ComfyUI `BaseModel` wrapper must account for the documented negative sign.

This remains blocked for a real trainer because the reconnaissance has not
confirmed, with a live loaded model:

1. the exact training data/payload construction for the intended recovery
   sample;
2. the runtime `context` and `PackedLayout` values for that sample;
3. which current loaded parameters retain `requires_grad=True` after NVFP4
   loading; and
4. a child checkpoint format that round-trips through the same loader.

No loss code was added.

## G. Trainable parameter candidates

The source gives a safe preliminary classification, but not the required
runtime proof:

| Candidate | Evidence | Current conclusion |
|---|---|---|
| Large attention/MLP `weight` tensors | Current checkpoint has U8 + NVFP4 metadata; `comfy/ops.py:1227-1231` wraps quantized weights with `requires_grad=False`. | Not safe as ordinary trainable parameters. |
| Video/audio patch projections | `model.py:490-491` constructs float32 `operations.Linear`; checkpoint records are F32. | Candidate, but `named_parameters()` and gradient participation still require a live model audit. |
| `condition_proj` | `model.py:492`; checkpoint is BF16, not NVFP4. | Candidate, but live `requires_grad` and memory behavior are unconfirmed. |
| `final_layer.video_out` / `audio_out` and biases | `model.py:507-508`; output weights are F32 in the checkpoint. | Candidate, but live audit is required. |
| Norms and AdaLN projection parameters | Model source constructs them as ordinary modules; checkpoint contains BF16/F16 records for these paths. | Candidate only; no layer may be selected by name without a runtime enumeration. |

The current ComfyUI loader does not provide a training configuration. The
quantized loader explicitly marks loaded quantized weights and extra quant
parameters as non-trainable (`ops.py:1227-1241`). A small trainable subset may
exist among non-quantized projections, biases, norms, or a separately saved
adapter, but this is **not yet proven on an instantiated loaded H3 model**.

## H. Checkpoint save/load strategy

### Confirmed

- Load: current inference path uses `safetensors.safe_open`,
  `load_diffusion_model_state_dict`, H3 detection, `MiniMaxH3`, and
  `ModelPatcher.load_model_weights` (`sd.py:2258-2362`).
- Generic save: `comfy.utils.save_torch_file` calls
  `safetensors.torch.save_file` (`utils.py:205-209`).
- Quantized state serialization: `comfy.ops._quantized_weight_state_dict`
  starts at `ops.py:1250` and emits quantization metadata/scales for
  `QuantizedTensor` weights.

### Not confirmed

There is no active H3 training/save command in ComfyUI or the installed
Diffusers package. The correct child strategy—full re-quantized safetensors,
BF16 replacement of selected tensors, or an adapter merged by the existing
Turbo LoRA path—has not been selected or tested. In particular, generic
`save_torch_file()` alone is not evidence that a modified child will load on
the RTX 5080 through the same NVFP4 path.

## I. Memory feasibility on RTX 5080 Laptop

Current measured facts:

```text
GPU VRAM:                  17,094,475,776 bytes (~15.92 GiB)
Parent NVFP4 file on disk: 12,528,636,800 bytes (~11.67 GiB)
Model blocks:              50
Batch requirement:         1
Accelerate:                absent
Bitsandbytes:              absent
```

The checkpoint's quantized storage makes inference fit the existing machine,
but this does not establish training feasibility. Backward activations,
optimizer state, gradient buffers, dequantized trainable weights, text
conditioning, VAE memory, and ComfyUI patcher/offload behavior were not
measured. `supported_models.py:973`'s `memory_usage_factor=0.114` is an
inference memory heuristic, not a training budget.

The first live implementation must therefore begin with a read-only loaded
model audit and a minimal forward/backward probe only after a valid child save
strategy is established. No such probe was run in this reconnaissance.

## J. Blockers and failure/rejection information

1. **ComfyUI service unavailable:** the host is reachable, but no process is
   serving `127.0.0.1:8188`; no live model instance was available for
   `named_parameters()`, `requires_grad`, or a forward probe.
2. **Quantized parent:** the real M0000 candidate is NVFP4 safetensors; the
   active quantized loader marks the quantized weights non-trainable. A valid
   trainable subset and optimizer contract remain to be verified.
3. **Checkpoint roundtrip:** current inference code can load and generically
   serialize safetensors, but no H3-specific modified-child save/load
   roundtrip has been demonstrated.
4. **Training data/payload:** no source-grounded recovery dataset and exact
   `minimax_payload` construction for a training sample has been identified.
5. **Upstream training code:** the installed ComfyUI tree contains inference
   model/loader/VAE code only. The installed Diffusers tree contains the H3
   transformer, scheduler, and inference modular pipeline, but no H3-named
   train/finetune/dataset implementation. This does not prove that no
   external/original training repository exists; its source has not been
   identified and must not be guessed.
6. **No failure taxonomy from a real experiment:** no smoke test was started,
   so there is no real training, checkpoint, benchmark, rejection, or failure
   record to report.

Because blockers 1–5 remain, A1-T0 is still blocked. Implementing a trainer
now would violate the requirement to stop when a key H3 fact is unconfirmed.

## K. Exact next implementation step

Do not modify Harness4H3-v1.0 and do not add an orchestration abstraction.
After the Windows ComfyUI service is deliberately started, perform one
read-only runtime audit using the existing ComfyUI loader and the current
checkpoint:

1. load `minimax_h3_fl2va_pruned_nvfp4.safetensors` through
   `comfy.sd.load_diffusion_model`, without changing the parent file;
2. record the actual `MiniMaxH3Model` instance's `named_parameters()`, dtype,
   device, shape, `requires_grad`, quantization metadata, and module paths;
3. construct the smallest real video/audio latent and Qwen conditioning
   objects using the existing H3 nodes/VAE/text encoder, then record one
   finite inference output and its exact shapes;
4. outside inference mode, test whether a deliberately selected
   non-quantized parameter participates in a real backward pass and has a
   non-zero gradient, without saving a child yet;
5. inspect the actual state-dict output of the loaded model and prove a
   same-loader roundtrip for a copied **read-only** parent representation
   before attempting any model-changing experiment;
6. locate and verify an external/original H3 training source, if available,
   rather than inventing dataset or target semantics.

Only after those six facts are recorded should a standalone
`recovery_finetune` smoke-test trainer be designed. Until then the correct
status is:

```text
A1-T0 blocked; trainer not implemented; no real M0001 exists.
```
