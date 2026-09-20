# Autonomous H3 Student Campaign Design

## Goal

让本地 LLM 只接收一个“基于 H3 自主产出 1B–2B 端侧小视频生成模型”的目标，随后通过远程 SSH 在 GPU 服务器上反复完成：学生架构提案、Harness 验证/编译、H3→Student 蒸馏与训练、量化、真实视频生成评估、失败诊断和下一轮修正。Codex 只负责部署和必要的开发，不承担长时间轮询。

最低可验收闭环是：LLM 生成合法的 1B–2B 架构；Harness 自动编译并启动训练；学生模型生成真实视频；至少完成两轮 proposal→train→evaluate→revise；中途不手改代码、不手选方案。

## Scope and non-goals

主线只覆盖一个可复现的视频 latent transformer 学生模型族、一个 H3 teacher adapter、一个蒸馏/训练 worker、一个真实视频 benchmark adapter 和一个持久化 campaign runner。现有 SSH、ComfyUI、trajectory、ExperienceStore、ModelStore、Pareto 和 checkpoint retention 作为基础设施复用。

本设计不把四卡满载、固定功耗、复杂 Experience Graph、所有 LPL/TDTM/CI-DL 变体、额外 UI 或在线监控作为验收条件。训练资源可以是 2–4 张 GPU，调度只要求安全、可恢复和能把失败原因交回 LLM。

## Design options

### A. Extend H3 pruning and fine-tuning

继续让 Controller 选择 `prune_blocks`、`distill`、`step_distill` 和 `quantize`，只改变已有 H3 的深度、采样步数或精度。

优点是复用最多、真实 H3 路径风险最低；缺点是学生模型结构仍由 H3 固定，不能证明 LLM 能自主产出新的 Student 结构。因此不作为主线。

### B. Declarative Student DSL with a trusted compiler (selected)

LLM 只输出结构化 `StudentProposal`，其中的 architecture 是受限 JSON DSL。Harness 负责 schema、参数量、形状、显存预算、拓扑和编译验证；远程 worker 只执行仓库中注册的模块和固定命令。LLM 不生成 Python、shell 或任意远程命令。

优点是同时满足结构自主设计、可审计、可恢复和可复现；架构变化仍能被控制在可支持的视频模型族内。缺点是第一版需要明确模块注册表和 H3↔Student 的适配边界。

### C. LLM-generated model code

允许 LLM 生成 Python 模型代码并在远程执行。

优点是结构自由度最高；缺点是无法建立可信执行边界，编译失败、依赖漂移、恶意/失控命令和 checkpoint 兼容问题都会破坏自治闭环。不采用。

## Architecture

```text
Goal + teacher manifest + target constraints
                    ↓
          Local LLM StudentProposal(JSON)
                    ↓
 StudentValidator: schema → params → shapes → memory → compile
                    ↓
     immutable proposal/manifest + fixed SSH worker argv
                    ↓
  H3 teacher → distillation/training → Student checkpoint
                    ↓
      quantization → real ComfyUI video generation
                    ↓
 validity + quality + latency + peak VRAM + failure code
                    ↓
 append-only experience/trajectory + bounded checkpoint retention
                    ↓
          next LLM context and revised proposal
```

### 1. Student proposal and DSL

The proposal has these top-level fields:

```json
{
  "schema_version": 1,
  "proposal_id": "student_0001",
  "parent_proposal_id": null,
  "teacher": {"checkpoint": "...", "adapter": "minimax_h3"},
  "architecture": {
    "family": "video_latent_dit",
    "latent_channels": 16,
    "hidden_size": 4096,
    "depth": 28,
    "num_heads": 32,
    "mlp_ratio": 4.0,
    "spatial_patch": 2,
    "temporal_patch": 1,
    "temporal_layers": [4, 8, 12, 16, 20, 24],
    "conditioning": "ada_norm_zero",
    "norm": "rmsnorm",
    "activation": "silu"
  },
  "training": {
    "method": "dmd2",
    "source_steps": 32,
    "target_steps": 8,
    "max_steps": 1000,
    "learning_rate": 1e-6
  },
  "deployment": {"precision": "bf16", "quantization": "int8"}
}
```

The exact values are examples, not a fixed architecture. The compiler exposes registered modules such as `VideoPatchEmbed`, `SpatialDiTBlock`, `TemporalDiTBlock`, `CrossAttention`, `AdaNormZero`, `RMSNorm`, and `VideoUnpatchify`. The LLM may select depth, width, head count, temporal placement and conditioning strategy within declared bounds; it may not add executable module code.

The validator rejects unknown keys, incompatible divisibility, unsupported module combinations, context/latent shape mismatches, non-finite values, and parameter estimates outside 1B–2B. It also rejects a proposal whose estimated peak memory exceeds the remote budget before any SSH training job is launched.

### 2. Harness compiler

`StudentCompiler` builds the registered `nn.Module` graph on the `meta` device or with fake tensors. It runs:

1. exact parameter counting from the constructed module;
2. example latent/conditioning shape propagation;
3. forward signature and output-shape checks;
4. graph export/trace validation;
5. estimated activation and optimizer memory;
6. a small real CPU smoke forward using the same graph family.

The compiler writes an immutable `compile_manifest.json` containing the canonical proposal, module graph, parameter count, shape report, compiler version, and digest. The remote worker accepts only this manifest and a trusted worker configuration; it never recompiles arbitrary source from the LLM.

### 3. Teacher/student training

`student_train_worker.py` is a fixed entrypoint. It loads the H3 teacher through the existing source-grounded adapter, constructs the compiled Student from the manifest, and runs one of the already-tested training algorithms behind a narrow teacher/student interface. The first supported method is latent/velocity distillation with optional DMD2 role updates; recovery fine-tuning remains a fallback for adapter bring-up, not the architecture proof.

Each run writes a result record with parent/child hashes, proposal digest, compiler digest, optimizer steps, final loss, gradient status, wall time, peak CUDA memory, and a classified failure code. A child is invalid if weights are copied unchanged, the required student tensors are absent, the forward contract fails, or hashes cannot be verified.

### 4. Quantization and generation evaluation

Quantization is a distinct fixed worker phase from the trained Student checkpoint. It writes a new child manifest and checkpoint, preserving the full-precision parent for rollback. The same evaluator then runs the fixed prompt/seed/sampling recipe through a Student-compatible ComfyUI adapter.

Evaluation records:

- video file exists and is decodable;
- frame count, duration, dimensions and finite pixel statistics;
- black/blank frame ratio and generation timeout;
- external quality score and quality components;
- latency and throughput;
- `torch.cuda.max_memory_allocated` plus process-level GPU memory sample;
- model/checkpoint size and quantization metadata.

Generation validity is a hard gate. A failed decode, timeout, OOM or invalid tensor produces a typed failure and never promotes the child, even if a proxy quality score exists.

### 5. Autonomous loop and failure feedback

`student-campaign` is a detached, resumable runner. Its durable state is an append-only JSONL event stream plus small pointer files. Every round follows the same sequence:

1. build bounded context from the goal, teacher manifest, current active proposal, recent evaluations and failure summaries;
2. call the configured local LLM with a strict JSON schema;
3. validate and compile the proposal locally on the control host;
4. upload only the manifest/config to the SSH host;
5. launch the fixed training/quantization/evaluation command;
6. import the result and classify success, rejection or failure;
7. retain metadata and only the active/best/direct-parent checkpoints;
8. pass the typed failure and the bounded evidence digest to the next LLM call.

The LLM may change proposal fields and training choices, but cannot lower hard gates, change the evaluator recipe, execute arbitrary commands, or promote a failed model. If the LLM returns malformed JSON or an invalid proposal, that is recorded as `proposal_invalid` and the next call receives the exact validator errors.

The campaign stores a `resume.json` and uses an SSH-side process supervisor (`tmux`/`nohup` wrapper already supported by the remote tooling) so an SSH disconnect does not stop the run. Codex needs only deploy/start/status operations; no long-running Codex poller is required.

### 6. Experience and checkpoint policy

Experience is append-only and compact. It retains proposal digest, parent/child identity, compile report, training metrics, evaluation metrics, failure code, diagnosis, and the next-round recommendation. Raw logs/videos/checkpoints are referenced by URI and are excluded from LLM context except for bounded summaries.

At most the following weight payloads are retained by default:

- active candidate;
- its direct evaluated parent;
- best feasible candidate;
- one current in-flight candidate while a round is running.

Rejected or failed child weights are deleted only after their result, hash and evidence records are durable. The retention policy never deletes the active parent before a successful child is evaluated.

## Interfaces

The main interfaces are intentionally small:

- `StudentProposal.from_dict(raw) -> StudentProposal`
- `StudentValidator.validate(proposal, target) -> ValidationReport`
- `StudentCompiler.compile(proposal, output_dir) -> CompileManifest`
- `StudentTrainWorker.run(manifest, teacher, training_config) -> TrainingResult`
- `StudentEvaluator.evaluate(checkpoint, tasks, recipe) -> EvaluationRecord`
- `StudentCampaign.run(max_rounds) -> CampaignResult`

All persisted records include schema versions and SHA-256 digests. Remote commands are fixed argument vectors derived from trusted configuration; proposal values are passed through JSON files rather than shell interpolation.

## Error handling

Errors are classified into proposal, compile, resource, training, checkpoint, generation, evaluation and infrastructure categories. Each error includes the phase, stable code, human-readable detail, relevant digest, and retryability. Retryable SSH/process failures resume the same round from its manifest; invalid proposals and failed evaluations create new LLM context instead of silently retrying the same plan.

The campaign stops only on success, explicit budget exhaustion, repeated infrastructure failure, or an unrecoverable teacher/evaluator capability failure. It never converts a blocked real run into a TinyH3 or fake-video success claim.

## Testing and acceptance evidence

Offline tests must prove:

1. two materially different DSL proposals both validate and compile;
2. a legal proposal has an exact constructed parameter count in 1B–2B;
3. illegal shape, budget, parameter and unknown-module proposals fail before SSH;
4. a fake teacher/student smoke worker writes a changed child and result evidence;
5. invalid video, OOM and quality regression become typed failures and enter the next context;
6. retention keeps only required weights while preserving metadata;
7. a two-round detached campaign runs without manual proposal selection.

Remote acceptance must additionally show real H3 teacher loading, real Student training, a changed child checkpoint, real student video generation, quality/VRAM evidence, and at least two persisted proposal/evaluation rounds. CPU/fake tests prove contracts only and are never reported as remote model evidence.

## Risks and mitigations

- **H3 output contract differs from Student input contract:** isolate conversion in the teacher adapter and fail before training if latent/conditioning shapes do not agree.
- **A nominal 1B–2B graph exceeds available memory:** perform meta-device parameter and memory checks before launch; let LLM revise the architecture.
- **ComfyUI cannot load the Student directly:** use a fixed Student adapter/export path and make incompatibility a typed generation failure, not a fake pass.
- **Long SSH job is interrupted:** persist manifest/state before launch, run detached remotely, and make resume idempotent by proposal digest.
- **LLM repeats a failed plan:** include recent proposal digests and typed failure summaries, and reject duplicate proposal digests within a configurable window.
