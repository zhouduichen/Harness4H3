# Wan Pipeline Device Placement Design

## Goal

Make real Wan evaluation use the configured inference device for all generation
components by default, while preserving an explicit CPU/offload escape hatch and
keeping baseline/candidate cache recipes unambiguous.

## Context and root cause

`WanVideoEvaluator._load_pipeline()` currently moves only `pipeline.transformer`
to `config.device`. The VAE remains on CPU, and `_generate()` unconditionally
moves latent frames to CPU before VAE decode. This makes every latency repeat pay
for a CPU decode and prevents the evaluator from using the GPU as the intended
hardware target. It also makes a future full-pipeline placement change unsafe
because the latent device is hard-coded.

## Design

1. Add an optional `pipeline_device` field to `WanVideoEvaluationConfig`.
   When omitted, it resolves to `device`; an explicit value such as `cpu` is
   supported for constrained environments.
2. Include the resolved pipeline device in the evaluator recipe digest. Existing
   results are never reused across different component-placement policies.
3. After loading Diffusers, move the pipeline's inference modules to the resolved
   device through `pipeline.to(device)` when available. If a test/dummy pipeline
   has no pipeline-level mover, move the known inference components individually
   (`transformer`, `vae`, and `text_encoder`). Always put the transformer in eval
   mode.
4. Encode cached prompt embeddings on the resolved pipeline device. Generation
   still passes them to the configured generation device, so an explicit split
   placement remains possible.
5. Decode latents on the actual VAE parameter device, rather than assuming CPU.
   This keeps CPU fallback and GPU placement both correct.
6. Keep quality evaluation independently configurable through the existing
   `quality_device`; it is not changed by pipeline placement.

## Safety and compatibility

- The default behavior changes only for real Wan evaluation; H3, Student,
  Fake-Wan, and training paths are untouched.
- CPU placement remains available for tests and memory-constrained servers.
- A pipeline that cannot move components raises the existing typed
  `WanEvaluationError`; it must not silently report a misleading measurement.
- Baseline and candidate receive the same resolved placement through the same
  evaluator config.

## Tests and acceptance

- Config tests cover default resolution, explicit CPU placement, and recipe
  digest changes when placement changes.
- A fake pipeline test records pipeline-level/component `.to()` calls and checks
  that VAE decode receives latents on the VAE device.
- Existing evaluator tests must continue to pass, including prompt caching,
  visual collapse rejection, and progress resume.
- After local tests pass, rerun the real Wan independent candidate matrix with
  `pipeline_device=cuda:<assigned GPU>` and compare candidate/baseline using the
  new recipe; old CPU-placement caches must not be reused.
