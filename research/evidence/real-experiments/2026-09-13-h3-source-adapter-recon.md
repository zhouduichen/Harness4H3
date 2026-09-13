# MiniMax-H3 source adapter reconnaissance — 2026-09-13

This is a source mapping exercise, not a training claim. The checked-in
`H3AdapterContract` remains fail-closed and
`configs/experiments/a1-t0-l40x4.yaml` remains `execution_enabled: false`.

## Contract mapping

| Harness contract | Source-grounded candidate | Adapter work still required | Status |
|---|---|---|---|
| `load_role(path, trainable)` | ComfyUI `comfy.sd.load_diffusion_model` → `MiniMaxH3`/`MiniMaxH3Model`; the official Diffusers layout uses `MiniMaxH3Transformer3DModel.from_pretrained` | Pick one parent format and instantiate it without ComfyUI inference-only patcher state | blocked |
| `prepare_batch(raw, generator)` | H3 has packed video/audio latents, Qwen conditioning, and separate video/audio VAE paths | Define an exact manifest, packing layout, conditioning payload, and deterministic CPU/GPU transfer | blocked |
| `predict(role, noisy, timestep, conditioning)` | `MiniMaxH3Model.forward` handles video/audio flow inputs and the model wrapper identifies `FLOW_AV` | Expose a training-safe forward that preserves gradients and matches the loaded model dtype/device | blocked |
| `save_role(role, path)` | `comfy.utils.save_torch_file` can serialize generic state dictionaries; quantized weights carry extra scale/metadata state | Decide whether the child is re-quantized safetensors, a BF16 selected-tensor child, or an adapter artifact | blocked |
| `reload_role(path)` | The existing H3 loader is the required reload authority for inference compatibility | Prove a modified child round-trips through the same loader and ComfyUI path | blocked |
| `resolve_trainable_parameters(role, policy)` | Quantized tensors are explicitly non-trainable in the current `comfy.ops` path; patch projections, output heads, norms, and AdaLN paths are candidates | Enumerate live `named_parameters()`, `requires_grad`, dtype, device, and non-zero backward participation on the actual loaded parent | blocked |
| `schedule(nfe)` / `add_noise` | Source recon records data-ward velocity, `t = 1 - sigma`, video shift `12.0`, and audio shift `3.0` | Confirm the exact installed source revision and packed modality timestep conventions in a live read-only probe | source mapped |

## Required L40 gate

Before changing `execution_enabled` to `true`, the operator must attach all of
the following evidence to one read-only parent audit and then to a copied-child
smoke test:

1. Actual `named_parameters()` and trainable scope from the instantiated H3
   model, including quantization metadata.
2. One finite video/audio forward and one finite backward with a non-zero
   gradient on an explicitly selected trainable tensor.
3. Parent hash stability before and after the probe.
4. A child save/reload round trip through the same authoritative loader.
5. Exact data manifest and conditioning/latent packing record.
6. Peak memory and dtype/device measurements for the selected L40 execution
   path.

Until those facts exist, the TinyH3 worker and DMD2 implementation are only
hardware-independent reference mechanisms. They must not be used as evidence
that MiniMax-H3 or a different open-source model such as DeepSeek is already
adapted.

## References already recorded in the repository

- MiniMax source: <https://github.com/MiniMax-AI/MiniMax-H3>
- Diffusers scheduler source: <https://github.com/huggingface/diffusers/blob/minimax-h3-test-failures/src/diffusers/schedulers/scheduling_minimax_h3.py>
- Existing detailed source audit: `research/evidence/real-experiments/2026-09-11-a1-h3-training-recon.md`
