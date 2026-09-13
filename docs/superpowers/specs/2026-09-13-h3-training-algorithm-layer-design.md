# H3 Training Algorithm Layer Design

Status: approved for implementation.

## Purpose

Build the real, hardware-independent training algorithm layer that
Harness4H3 needs before the four-L40 H3 integration. The deliverable includes
a common training engine, a real TinyH3 PyTorch reference model, recovery
fine-tuning, binary progressive step distillation, a content-addressed data
cache, and a DMD2 reference skeleton. It must exercise real forward,
backward, optimizer, checkpoint, resume, child-lineage, and failure paths.

The implementation is not evidence that MiniMax-H3 itself can train on the
current machine. Only a later source-grounded H3 adapter running on the target
host may make that claim.

## Existing boundaries

Harness4H3-v1.0 keeps the Controller protocol, validation order, evaluator
authority, archive semantics, acceptance policy, and TargetProfile meaning
frozen. The training layer is a research-side execution backend and connects
through the existing `ExternalScriptOperator` and `tools/h3_model_worker.py`
JSON contracts.

The existing Harness-visible operator names remain:

- `recovery_finetune` for recovery training;
- `step_distill` for one progressive distillation stage.

The Controller chooses an operator and bounded arguments. It never supplies
training code, a shell command, a loss implementation, or an evaluator.

The user's existing uncommitted changes in
`research/experiments/m6_campaign.py` and
`research/experiments/m6_runtime_recipe.py` are outside this work and must be
preserved.

## Research judgment

Classic Progressive Distillation is the first distillation implementation
because its frozen-teacher trajectory target is deterministic and can be
verified on a tiny model. It is not treated as the only future acceleration
algorithm. Recent phased, distribution-matching, and parallel-decoding work
shows that large video and audio-video models may need different objectives
at low NFE.

DMD2 is a separate multi-role algorithm. Its required core is a student, a
frozen target teacher, a trainable fake-score model, alternating update rates,
and student EMA. A regression anchor remains available as an optional
stabilizer, but is disabled by default because removing the original DMD
offline regression-pair requirement is one of DMD2's stated changes. An
optional adversarial branch is included in the interface because it is part
of DMD2, but the project will not claim paper-level image or video results
from the toy implementation.

Parallel Decoding Distillation is an explicit later extension point rather
than part of this implementation. It changes the student output head to
predict multiple interval velocities, so folding it into binary progressive
distillation would hide a material architecture change.

## Package architecture

Training code lives in a new top-level Python package in the same repository:

```text
h3_training/
├── engine/
│   ├── trainer.py
│   ├── state.py
│   ├── optimizer.py
│   ├── checkpoint.py
│   └── evidence.py
├── algorithms/
│   ├── base.py
│   ├── recovery_finetune.py
│   ├── progressive_distillation.py
│   └── dmd2.py
├── adapters/
│   ├── base.py
│   ├── tiny.py
│   └── h3_contract.py
├── data/
│   ├── schema.py
│   ├── dataset.py
│   └── cache.py
└── tiny/
    ├── model.py
    ├── factory.py
    └── evaluator.py

tools/
└── tiny_training_worker.py

research/experiments/
└── tiny_real_closed_loop.py
```
The exact module split may be made slightly finer in the implementation plan
when a file would otherwise have more than one responsibility. It must not be
collapsed into `harness4h3/`, because the training implementation is not part
of the frozen Harness core.

PyTorch and tensor serialization dependencies are exposed through a
`training` optional dependency group. Installing or testing the ordinary
Harness must not import PyTorch.

## Responsibility model

### Trainer engine

`TrainerEngine` owns execution mechanics only:

```python
class TrainerEngine:
    def run(
        self,
        method: TrainingMethod,
        dataloader: Iterable[RawBatch],
        max_steps: int,
        resume_from: Path | None = None,
    ) -> TrainingRunResult: ...

    def backward(
        self,
        losses: Mapping[str, Tensor],
        accumulation_steps: int,
    ) -> None: ...

    def optimizer_step(
        self,
        method: TrainingMethod,
        iteration: int,
    ) -> Mapping[str, float]: ...

    def save_checkpoint(
        self,
        method: TrainingMethod,
        loop_state: LoopState,
        path: Path,
    ) -> CheckpointEvidence: ...

    def save_child(
        self,
        method: TrainingMethod,
        parent: ParentEvidence,
        path: Path,
    ) -> ChildEvidence: ...
```

`max_steps` counts completed optimizer iterations, not consumed
micro-batches. Each iteration consumes exactly
`gradient_accumulation_steps` batches. Losses are divided by the accumulation
count before backward. Gradient clipping occurs once, immediately before the
scheduled optimizers step.

### Training methods

Algorithms own role relationships, losses, optimizer selection, and
algorithm-specific state:

```python
class TrainingMethod(ABC):
    def prepare(self) -> None: ...
    def training_step(
        self,
        batch: PreparedBatch,
        iteration: int,
    ) -> StepOutput: ...
    def optimizers(self, iteration: int) -> Sequence[Optimizer]: ...
    def grad_clip_targets(
        self,
        iteration: int,
    ) -> Mapping[str, Module]: ...
    def checkpoint_state(self) -> Mapping[str, Any]: ...
    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None: ...
```

`StepOutput` contains named differentiable losses, detached scalar metrics,
and any detached outputs needed for evidence collection. The engine
rejects a missing `total_loss`, a non-scalar loss, a non-finite loss, or a
metric key that collides with a loss key.

### Model adapters and roles

Every algorithm receives named role models. A role contains its model,
adapter, trainability declaration, and role-specific scheduler state.

```python
class DenoisingModelAdapter(ABC):
    def prepare_batch(
        self,
        raw: RawBatch,
        generator: Generator,
    ) -> PreparedBatch: ...
    def add_noise(
        self,
        clean: ModalLatents,
        noise: ModalLatents,
        timestep: ModalTimesteps,
    ) -> ModalLatents: ...
    def predict(
        self,
        role: ModelRole,
        noisy: ModalLatents,
        timestep: ModalTimesteps,
        conditioning: Conditioning,
    ) -> ModalPrediction: ...
    def prediction_to_clean(
        self,
        noisy: ModalLatents,
        prediction: ModalPrediction,
        timestep: ModalTimesteps,
    ) -> ModalLatents: ...
    def scheduler_step(
        self,
        role: ModelRole,
        latent: ModalLatents,
        prediction: ModalPrediction,
        interval: ModalInterval,
    ) -> ModalLatents: ...
    def schedule(self, num_model_evaluations: int) -> ModalSchedule: ...
    def save_role(self, role: ModelRole, path: Path) -> ModelEvidence: ...
    def reload_role(self, path: Path) -> ModelRole: ...
```

The adapter owns prediction parameterization, signs, modality-specific
schedules, tensor layout, and export format. Algorithms never assume epsilon,
standard v-prediction, x0 prediction, or a DDPM scheduler.

The future H3 contract records the source-confirmed data-ward velocity,
`t = 1 - sigma`, and separate video/audio schedules with shifts 12 and 3. It
must fail with `h3_adapter_unavailable` until an actual H3 loader, trainable
parameter policy, backward path, child exporter, and reload path are supplied.

## Data contracts

The raw sample permits preprocessing to happen online or ahead of time:

```python
@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    prompt: str
    seed: int
    media: MediaReferences | None = None
    text_embedding: Tensor | None = None
    latents: ModalLatents | None = None
    noise: ModalLatents | None = None
    timesteps: ModalTimesteps | None = None
    teacher_signals: TeacherSignals | None = None
```

`ModalLatents`, `ModalPrediction`, and `ModalTimesteps` contain an optional
video value and an optional audio value. At least one modality must be
present. An algorithm validates the fields its selected data mode requires.

Supported modes are:

- `text_only`: condition plus fresh noise, for on-policy rollout methods;
- `cached_latent`: precomputed text embedding and VAE latents, used by the
  primary recovery and progressive-distillation paths;
- `real_latent`: real paired latents, used by supervised recovery and the
  optional DMD2 adversarial branch.

The schema distinguishes raw samples from `PreparedBatch`. Device placement,
batch dimensions, generated noise, sampled timesteps, and unconditional
conditioning belong only to the prepared batch.

## Content-addressed cache

The cache supports these artifact kinds:

- text-encoder output;
- video VAE latent;
- audio VAE latent;
- teacher trajectory;
- teacher prediction.

Every cache key is a SHA-256 digest over canonical JSON containing:

- cache schema version and artifact kind;
- sample or source-content digest;
- text encoder, VAE, or teacher checkpoint digest as applicable;
- complete preprocessing configuration;
- modality, dtype, and shape;
- schedule/timestep/interval data as applicable;
- seed and conditioning digest.

Tensor payloads use safetensors and non-tensor metadata uses JSON. Writes use
a sibling temporary path, fsync, and atomic replacement. Reads verify the
manifest, payload hash, declared tensor names, shapes, and dtypes. An invalid
entry raises `cache_corrupt`; a key mismatch is a cache miss, never a stale
hit. Cache creation is deterministic and safe for repeated processes racing
to publish the same key.

## Algorithms

### Recovery fine-tuning

Recovery owns a trainable `student` and may own a frozen `teacher`. Its loss
is:

```text
total_loss =
    sft_weight * weighted_flow_target_loss
  + drift_weight * weighted_teacher_prediction_loss
```

The SFT term uses the adapter's forward process and prediction
parameterization. The drift term runs the teacher under `no_grad` and prevents
an adapter or structurally modified child from moving too far from the parent.
At least one term must have positive weight.

The adapter resolves a named freeze policy into exact parameter names. The
method records the complete trainable and frozen name sets before constructing
AdamW. Empty trainable sets and unmatched freeze policies fail before the
first forward pass.

### Binary progressive step distillation

One `step_distill` invocation performs one binary stage. The stage uses
`teacher_nfe == 2 * student_nfe`; both values are explicit after resolving the
parent state and operator configuration. `NFE` means model evaluations, not
sigma-grid points.

For each sampled student interval, the teacher advances through two aligned
sub-intervals under `no_grad`. The student advances once over the identical
outer endpoints. The differentiable endpoint loss is:

```text
total_loss =
    video_weight * mse(student_video_endpoint, teacher_video_endpoint)
  + audio_weight * mse(student_audio_endpoint, teacher_audio_endpoint)
```

Weights for absent modalities must be zero. At least one present modality
must have a positive weight. Schedule validation rejects non-monotonic grids,
misaligned outer endpoints, missing terminal states, and a teacher/student NFE
ratio other than two.

Multi-stage progress is represented by immutable Harness children, for
example `M0000(16) -> M0001(8) -> M0002(4)`. The trainer never promotes its
own child or begins the next stage. The Harness evaluator decides whether a
child is retained before it can become a later parent.

### DMD2 reference skeleton

DMD2 owns:

- a trainable `student` generator;
- a frozen `teacher` target score model;
- a trainable `critic` acting as the fake-score model;
- independent student and critic optimizers/schedulers;
- a timestep/noise sampler;
- student EMA;
- optional regression-anchor and adversarial-loss components.

The critic is updated every iteration on noised, detached student samples.
The student is updated only when
`iteration % generator_update_interval == 0`. EMA advances only after a
student optimizer step. The distribution-matching pseudo-loss exposes to
autograd the normalized difference between target and fake score estimates
while treating both score estimates as stopped-gradient targets.

The toy implementation must run both `text_only` simulated rollout and
`real_latent` modes. The adversarial and regression components are exercised
by focused tests but remain configurable and off by default. Documentation
labels this code as a reference skeleton, not a validated MiniMax-H3 DMD2
recipe.

## Checkpoint and exact resume

A training checkpoint is distinct from a deployable child. It contains:

- every trainable role state;
- every optimizer and learning-rate scheduler state;
- EMA state;
- algorithm state and update counters;
- global optimizer step and accumulation position;
- stateful sampler position;
- CPU, CUDA when present, and algorithm-generator RNG states;
- parent checkpoint hash;
- canonical configuration digest;
- checkpoint schema version.

Checkpoint publication is atomic. Loading validates schema, algorithm name,
role set, model structure, parent hash, and all resume-critical configuration
before restoring state. RNG restoration happens after model, optimizer,
dataloader iterator, and sampler reconstruction so setup cannot consume the
restored sequence.

The reference test compares an uninterrupted run with an interrupted and
resumed run using the same seed and batches. Model parameters, optimizer
states, EMA, counters, and final metrics must match exactly on CPU.

## Child evidence and lineage

Before training, the worker records the immutable parent file hash and hashes
of all named tensors. `save_child` writes a model-only child through the
adapter, reloads it through the same adapter, and emits an evidence manifest.

A successful child proves:

- parent file hash is identical before and after execution;
- loss values are finite;
- at least one measured gradient norm is non-zero;
- at least one optimizer step completed;
- the trainable parameter count is positive;
- at least one intended trainable tensor changed;
- no frozen tensor changed;
- child and parent paths are distinct;
- child and parent file hashes differ;
- the child reloads and produces finite validation output;
- changed tensor and parameter counts are measured, not synthesized.

The worker reports initial/final loss, maximum observed gradient norm,
optimizer-step counts by role, trainable parameter count, parent and child
hashes, changed tensor/parameter counts, unexpected frozen changes, peak
memory when measurable, and wall time. The existing
`tools/h3_model_worker.py` stages the returned child into experiment
artifacts and normalizes it for the Harness.

## TinyH3 reference system

TinyH3 is a small joint video/audio denoiser implemented in real PyTorch. It
uses a shared conditioned backbone with separate video and audio heads and a
data-ward rectified-flow convention. It is intentionally small enough for CPU
tests and carries the architecture label `TinyH3`, making its non-H3 status
explicit while satisfying the existing worker's H3-family validation.

The synthetic dataset is deterministic from sample ID and seed. It provides
paired prompt embeddings, clean video/audio latents, and a learnable flow
target. TinyH3 checkpoints contain real tensors and model configuration, not
precomputed metrics.

`TinyCheckpointEvaluator` reloads each candidate, measures held-out loss, and
derives its quality score from the measured baseline-relative loss. Sampling
cost is derived from the candidate NFE and measured wall time. It does not use
the offline `metrics *= constant` backend.

The closed-loop experiment injects a deterministic bounded Controller during
tests and accepts any existing Harness Controller in the runnable experiment.
It exercises:

```text
ExperimentPlan
  -> ExternalScriptOperator
  -> h3_model_worker.py
  -> tiny_training_worker.py
  -> real PyTorch child
  -> TinyCheckpointEvaluator
  -> accept/reject
  -> immutable M0001/M0002 lineage
```

## Stable training failures

The training layer emits stable failure types and never substitutes a copied
checkpoint or simulated metric:

| Failure | Meaning |
|---|---|
| `unsupported_training_operator` | Worker request names an unimplemented operator |
| `invalid_training_config` | Configuration or operator arguments fail validation |
| `no_trainable_parameters` | Freeze policy selects no trainable tensor |
| `nonfinite_loss` | A loss is NaN or infinite |
| `zero_gradient` | No intended parameter receives a non-zero gradient |
| `training_oom` | PyTorch reports an out-of-memory condition |
| `checkpoint_corrupt` | Training checkpoint cannot be validated or loaded |
| `resume_mismatch` | Checkpoint parent, algorithm, roles, model, or configuration differs |
| `cache_corrupt` | Cache manifest or tensor payload fails integrity checks |
| `parent_modified` | Parent file hash changed during execution |
| `unchanged_child` | No intended trainable tensor differs from the parent |
| `frozen_tensor_changed` | A tensor outside the trainable set changed |
| `child_reload_failed` | Saved child cannot be reloaded and evaluated |

Failures preserve diagnostic artifacts and return a failed worker result.
Partial children are never returned as successful output state.

## Verification strategy

The implementation is complete only when tests prove all of the following:

1. TinyH3 performs real forward, backward, and AdamW updates on CPU.
2. Freeze policies, gradient accumulation, and gradient clipping behave as
   configured.
3. Recovery loss decreases and only declared trainable tensors change.
4. Progressive Distillation freezes the teacher, validates schedule
   alignment, computes weighted video/audio endpoint loss, and produces a
   reloadable child.
5. Cache keys invalidate on source, checkpoint, preprocessing, schedule,
   timestep, or seed changes; corrupt payloads fail closed.
6. DMD2 produces finite critic and student losses, non-zero gradients,
   alternating role updates, changing weights, and advancing EMA.
7. Interrupted/resumed recovery, progressive distillation, and DMD2 runs
   match uninterrupted CPU runs exactly.
8. NaN loss, zero gradients, checkpoint corruption, parent mutation,
   unchanged children, frozen-tensor mutation, and reload failure all return
   their stable failures.
9. The real subprocess chain through `h3_model_worker.py` stages a TinyH3
   child without changing its parent.
10. A Harness closed-loop test creates and evaluates real PyTorch `M0001`
    and `M0002` candidates with correct lineage and accept/reject records.
11. Ordinary Harness imports and its full existing test suite pass without
    the training extra installed.
12. The training test suite and `compileall` pass with the training extra
    installed.

## Delivery sequence

Implementation follows the requested priority while preserving independently
testable checkpoints:

1. trainer/method/adapter contracts and data schema;
2. TinyH3 model, deterministic data, and evaluator;
3. recovery fine-tuning plus optimizer/checkpoint/evidence paths;
4. binary progressive step distillation and stage metadata;
5. content-addressed preprocessing and teacher cache;
6. DMD2 reference skeleton;
7. worker and full Harness closed-loop integration;
8. failure injection, exact-resume, and full regression verification.

## Primary references

- Tim Salimans and Jonathan Ho, [Progressive Distillation for Fast Sampling
  of Diffusion Models](https://arxiv.org/abs/2202.00512), 2022.
- David Berthelot et al., [TRACT: Denoising Diffusion Models with Transitive
  Closure Time-Distillation](https://arxiv.org/abs/2303.04248), 2023.
- Fu-Yun Wang et al., [Phased Consistency
  Models](https://arxiv.org/abs/2405.18407), 2024.
- Tianwei Yin et al., [One-step Diffusion with Distribution Matching
  Distillation](https://arxiv.org/abs/2311.18828), 2023.
- Tianwei Yin et al., [Improved Distribution Matching Distillation for Fast
  Image Synthesis](https://arxiv.org/abs/2405.14867), 2024, with the
  [authors' implementation](https://github.com/tianweiy/DMD2).
- Zihan Ding et al., [DOLLAR: Few-Step Video Generation via Distillation and
  Latent Reward Optimization](https://arxiv.org/abs/2412.15689), revised
  2026.
- Neta Shaul et al., [Parallel Decoding Distillation for Fast Image and Video
  Generation](https://arxiv.org/abs/2607.26004), 2026.
- MiniMax and Hugging Face,
  [MiniMax-H3 model definition](https://github.com/MiniMax-AI/MiniMax-H3)
  and
  [MiniMaxH3Scheduler](https://github.com/huggingface/diffusers/blob/minimax-h3-test-failures/src/diffusers/schedulers/scheduling_minimax_h3.py).
- Hao AI Lab,
  [FastVideo training infrastructure](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/training/train_infra.md),
  used as implementation evidence for role-based training boundaries.
