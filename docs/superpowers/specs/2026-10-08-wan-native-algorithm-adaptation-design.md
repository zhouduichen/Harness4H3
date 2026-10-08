# Wan-native Student Algorithm Adaptation Design

## Goal

将已有 H3 Student 的 Progressive Distillation、DMD2、rollout、F1→F2→F3、初始化、精度和 lineage 能力适配到 Wan-native Student，而不是为 Wan 再实现一套平行训练算法。vLLM 只选择经过 capability gate 的训练策略；真实 H3 Teacher、Wan 1–2B Student、overlap-window 30 秒 evaluator 和 checkpoint lineage 保持不变。

## Existing capabilities to reuse

The implementation must reuse these existing contracts:

- `h3_training.algorithms.progressive_distillation.ProgressiveDistillation`;
- `h3_training.algorithms.dmd2.DMD2`;
- `DenoisingModelAdapter` and `TrainingMethod` interfaces;
- `progressive_rollout_policy` and the existing endpoint/velocity/clean-anchor losses;
- `StudentTrainWorker` dispatch, F1/F2/F3 stage lineage, parent binding, and evidence fields;
- existing capability snapshots, append-only experiment events, and deterministic candidate decisions.

Wan-specific code is limited to an adapter and trusted data/role bindings. The existing `wan_velocity_loss` path remains a compatibility baseline, but it is not the primary implementation for new algorithm candidates.

## Wan adapter boundary

Create a `WanTrainingAdapter` implementing the existing `DenoisingModelAdapter` contract:

- `prepare_batch` loads a real H3 cache item, decodes/resamples it through the existing H3 VAE boundary, encodes it with the frozen Wan VAE, and returns a `PreparedBatch` containing Wan-normalized clean latents and native Wan text conditioning;
- `add_noise` uses Wan Flow-Match interpolation and the Wan scheduler timestep convention;
- `predict` calls the Wan Transformer role and returns a `ModalPrediction` with the Wan velocity output;
- `prediction_to_clean` uses the Wan Flow-Match clean reconstruction equation;
- `scheduler_step` advances the Wan latent through one aligned interval;
- `schedule` returns a binary-compatible Wan sigma schedule for Progressive Distillation and bounded K-step rollout;
- `save_role`, `reload_role`, and `resolve_trainable_parameters` delegate to the existing strict Wan checkpoint and parameter-selection logic.

The adapter must not reinterpret H3 latent tensors as Wan latents. H3 Teacher outputs cross the existing RGB boundary and are then encoded by the Wan VAE. Teacher trajectory targets are cached or generated at the boundary needed by the selected method and are always detached before Student backward.

## Normalized vLLM proposal schema

Add schema version 2 while preserving readable version 1 artifacts. The normalized proposal contains only bounded policy fields:

```json
{
  "schema_version": 2,
  "proposal_id": "wan_0001",
  "parent_checkpoint_sha256": "<trusted digest>",
  "training": {
    "method": "progressive_distillation",
    "source_steps": 32,
    "target_steps": 8,
    "train_steps": 256,
    "learning_rate": 0.0001,
    "critic_learning_rate": 0.0001,
    "velocity_weight": 1.0,
    "clean_anchor_weight": 0.5,
    "rollout_steps": 4,
    "rollout_every": 2,
    "rollout_endpoint_weight": 1.0,
    "rollout_clean_weight": 0.5,
    "trainable_scope": "all",
    "fp32_optimizer_state": false
  },
  "initialization": {"mode": "inherit"},
  "precision": {"loss_reduction": "fp32", "optimizer_state": "bf16"},
  "curriculum": {"fidelity_schedule": ["F1", "F2", "F3"]},
  "data": {"manifest_digest": "<trusted digest>", "prompt_policy": "balanced", "seed_policy": "fixed"},
  "continuation": {"overlap_latent_frames": 7, "anchor_schedule": "linear"},
  "focus": "semantic",
  "hypothesis": "..."
}
```

vLLM cannot emit paths, shell commands, device assignments, model ids, Teacher identity, or code. The supervisor binds all trusted values and rejects unsupported combinations before launch. A candidate is selectable only when the Wan capability registry reports the requested method and all nested options as executable.

## Capability registry

The Wan campaign exposes a snapshot derived from the trusted adapter, not from vLLM claims. Initial capabilities are:

- Progressive Distillation: source/target binary stages, endpoint, velocity, clean-anchor, K=2/4 rollout;
- DMD2: only when the Wan DMD2 role/critic adapter and memory gate are available;
- initialization: full inherit, generation-head reset, DiT-Zero reset;
- precision: FP32 loss reduction; FP32 optimizer state only when memory admission passes;
- trainable scopes: all, heads, and validated parameter prefixes;
- curriculum: ordered F1→F2→F3 prefixes;
- continuation: the existing Wan overlap contract and validated anchor schedules.

The registry records `available`, `backend`, `constraints`, `evidence_level`, and `reason`. Unsupported options are rejected as `capability_unavailable`, never silently downgraded.

## Training and lineage flow

For each round:

1. Load the trusted parent and verify its SHA-256.
2. Evaluate the parent on the immutable 30-second manifest.
3. Provide vLLM with the bounded history, current capability snapshot, parent evidence, and failed-candidate fingerprints.
4. Normalize and validate one proposal locally.
5. Dispatch the existing Progressive/DMD2 worker through the Wan adapter.
6. Record stage lineage, rollout policy, initialization, precision, and parent/Teacher digests in the child metadata.
7. Evaluate the child with the same overlap-window contract and CLIP temporal backend.
8. Promote only a valid child that satisfies the Teacher-relative floor and deterministic improvement rule.

## Durable memory and resume

The Wan campaign state is append-only and restartable. Each campaign stores:

- immutable `campaign-base.json` with model/Teacher/evaluator/capability digests;
- per-round `proposal.json`, `training-result.json`, `evaluation-result.json`, and `decision.json`;
- append-only `events.jsonl`;
- atomic `campaign-state.json` containing the last complete round, active parent, parent digest, accepted history, and next round number;
- `fingerprints.jsonl` or an equivalent durable duplicate index keyed by proposal policy, parent digest, data manifest, and code/evaluator digests.

On startup, the supervisor loads complete round artifacts and state, marks an incomplete round as failed, resumes from the last accepted parent, and does not ask vLLM to repeat an already completed fingerprint. A changed base/config/capability digest requires a new campaign root.

## Non-goals

- no new Student architecture;
- no replacement Teacher or VAE;
- no automatic code generation by vLLM;
- no arbitrary vLLM paths, devices, or shell execution;
- no claim of semantic improvement without the fixed 30-second multi-prompt/multi-seed evaluator.

## Testing

- adapter unit tests with mocked Wan pipeline/scheduler for noise, prediction, clean reconstruction, scheduler alignment, and parameter scopes;
- proposal tests for nested schema, range validation, parent digest binding, capability rejection, and backward-compatible v1 normalization;
- worker tests proving Progressive/DMD2 dispatch reaches the shared algorithms and writes stage lineage;
- resume tests covering complete rounds, partial rounds, duplicate fingerprints, changed base digest, and accepted-parent promotion;
- integration smoke with one real vLLM proposal and one real Wan/H3 round before increasing the budget.

## Implementation status

- Proposal schema v2, bounded capability validation, campaign fingerprinting, append-only state, and restart recovery are implemented.
- The existing Progressive Distillation path is executable through `WanTrainingAdapter`, including binary stages, velocity/clean-anchor losses, rollout policy, initialization mode, trainable scope, and lineage metadata.
- DMD2 remains fail-closed for Wan until both a Wan critic adapter evidence flag and explicit critic-memory admission are present; no fallback to the legacy direct-loss loop is allowed for v2 proposals.
- The local contract suite passes with `53 passed, 1 skipped`; the skipped DMD2 test requires CUDA. A real four-GPU H3/ Wan smoke and semantic 30-second evidence remain deployment checks, not claims made by this implementation.

## DMD2 and step-level training memory extension

The next implementation increment reuses the existing DMD2 and TrainerEngine
contracts rather than introducing Wan-specific copies:

- `DMD2` remains the only DMD2 implementation. The Wan adapter supplies a
  Wan Student role, a critic role created from the same trusted Wan checkpoint,
  and the H3 RGB-boundary endpoint Teacher role.
- DMD2 is executable only when both `wan_dmd2_adapter_ready=true` and
  `dmd2_memory_admitted=true` are present in the trusted campaign config. The
  vLLM proposal cannot enable either flag.
- The existing `TrainerEngine.save_checkpoint()` and
  `load_training_checkpoint()` own step-level recovery. Wan worker execution
  will checkpoint only at optimizer boundaries and will restore Student,
  Critic, both optimizers, EMA, sampler RNG, CPU/CUDA RNG, loop state, parent
  digest, and algorithm/config digests.
- The existing `harness4h3.memory.fingerprint.experiment_fingerprint` is the
  canonical policy identity. Wan campaign state remains the scheduler state,
  while `harness4h3.memory.experience.ExperienceStore` receives one append-only
  evidence record for each completed, rejected, or failed round.
- A failed or interrupted DMD2 round is never promoted. A restart may resume
  the last valid training checkpoint only when all parent, algorithm, policy,
  capability, and evaluator digests match; otherwise it starts a new campaign
  root.

The extension is complete only when CPU/tiny DMD2 resume tests, Wan adapter
critic tests, worker checkpoint recovery tests, campaign restart tests, and a
real GPU memory-gated Wan smoke all pass. No semantic 30-second improvement is
claimed from contract tests alone.
