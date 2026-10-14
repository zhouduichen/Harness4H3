# EvoCompress 操作说明

EvoCompress 在 Harness4H3 中增加一个独立的 DiT 结构化压缩闭环：

```text
Wan checkpoint
    ↓
profile
    ↓
AlgorithmAgent 生成受限算法
    ↓
Sandbox 校验并执行
    ↓
Block/FFN Channel Pruning
    ↓
EvaluationRecord / archive / ExperienceStore
    ↓
下一轮算法
```

## CPU 契约验证

不需要 CUDA 或真实 Wan 权重时，可以使用 fake Wan DiT：

```bash
.venv/bin/python -m harness4h3 profile-wan \
  --config configs/evolutionary-compression.fake.yaml \
  --output var/evolutionary-compression-fake/profile.json --json

.venv/bin/python -m harness4h3 compress-wan \
  --config configs/evolutionary-compression.fake.yaml \
  --algorithm configs/algorithm-prune-least-sensitive.py --json

.venv/bin/python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.fake.yaml --rounds 2 --json
```

每轮会写入 `parent/`、`rounds/Rxxxx/`、`algorithms/candidates/`、
`models/` 和 `experiences.jsonl`。fake 结果显式带有
`offline_simulation=true`，不能作为真实 Wan 质量或硬件结论。

## 真实 Wan smoke

编辑 `configs/evolutionary-compression.wan.example.yaml` 中的可信本地
checkpoint、Wan 生成配置、评测 manifest 和本地 CLIP 路径，在安装 `.[wan]`
依赖并具备 CUDA 的机器上运行：

```bash
python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.wan.example.yaml \
  --rounds 1 --json
```

真实 Diffusers checkpoint 必须配置 `evaluator.backend: wan_video`；CLI
会拒绝用 structural proxy 给真实权重打分。真实 evaluator 使用固定的
prompt/seed 生成视频，用本地 CLIP-temporal backend 评估语义、时间一致性
和运动；同时使用可配置的 visual-structure hard gate 拒绝空白、近纯色或低纹理
坍缩视频，并把生成耗时、视频有效性和质量证据写入
`output_root/real-evaluation/`。当前 1.3B smoke 支持把 Transformer 放到
GPU，T5 encoder 和 VAE 留在 CPU。

`search-policy.json` 会跨次运行保存并恢复；实测可行且有改进的轮次提高
exploit 比例，失败轮次降低 exploit 比例。真实 adapter 对未知 Wan 模块布局
fail closed，不会把未知结构当作可安全裁剪的 Transformer block。

## 正式开发集 / 测试集

仓库提供两个固定 manifest：

- `configs/evaluation-manifest.wan-rsi-dev.json`：8 个 Prompt、每个 2 个 Seed；
- `configs/evaluation-manifest.wan-rsi-test.json`：独立的 8 个 Prompt、每个 2 个 Seed。

使用 `configs/evolutionary-compression.wan-formal.example.yaml` 作为模板，分别
为 dev 和 test 建立不同的 `output_root`。正式评测必须保持
`max_cases: 8`、`seed_count: 2`，并保留每次生成的 `latency_samples_s`、视频有效性、
质量指标、模型哈希和 evaluator recipe digest。smoke 配置的 1×1 结果只能证明
真实链路可执行，不能替代正式结果。每个 artifact 的评测目录还会按完成的
case/seed 原子写入 `progress.json`；进程中断后会复用已完成且视频仍存在的记录，
不会把未完成 case 当作有效证据。

当前 evaluator 的延迟方法是
`denoising+vae_decode_with_case_cached_prompt_embeddings`：固定 case 的正/负 Prompt
embedding 只编码一次，warmup 和 3 次 repeat 测量扩散 Transformer 与 VAE decode，
并在硬件指标中明确 `prompt_encoding_included=false`。T5 未被压缩，排除其重复编码
可以避免把未优化组件的 CPU 时间误归因给候选 Transformer；recipe digest 会随测量
方法变化，旧缓存不会被错误复用。

Peak memory 会针对 evaluator 配置的 `generation_device` 重置并读取；不能用默认
CUDA device 的统计值代表实际执行卡。如果运行环境无法提供 CUDA memory telemetry，
`peak_memory_gb_max` 和 `peak_memory_available` 必须分别写成 `null` 和 `false`，
不能写零，更不能解释成低显存占用。每个 case 也保留实际 device 和可用性。

### Evaluation Recipe：Wan 指标口径

同一份 Recipe 必须固定 checkpoint revision、Wan generation config、evaluation
manifest、prompt/seed、warmup、latency repeat、dtype、generation device、pipeline
device 和 quality backend。当前 latency 是
`denoising+vae_decode_with_case_cached_prompt_embeddings`，不包含固定文本编码器的
重复编码；Recipe 会记录 `prompt_encoding_included=false` 和 `latency_samples_s`。

指标含义固定如下：

- `transformer_parameter_count` 和兼容字段 `parameter_count` 只数压缩目标
  Transformer，`parameter_count_scope=transformer`；不能把 T5/VAE 参数混进压缩收益。
- `transformer_model_size_gb` 只数 Transformer 目录；`pipeline_model_size_gb` 和兼容
  字段 `model_size_gb` 数完整 Pipeline 的逻辑文件体积。
- 体积按 `(device, inode)` 去重，因此 hardlink 和指向同一文件的 symlink 不会被重复
  计数。Child 只改 Transformer 时，固定 T5/VAE 的共享体积不会被误报成压缩收益。
- 峰值显存必须来自真实 `generation_device`，任何缺失、异常或不可用 telemetry 都是
  `null`/unavailable。零不是有效的显存测量。

后续晋升协议先检查 Child 完整性、独立真实评测、aggregate quality 至少为当前
Parent 的 98%、已有可靠 semantic/temporal/motion 子指标不发生不可接受退化，以及
可用硬件指标没有超过退化容忍度。通过这些硬约束后，latency 改善 5%、peak memory
改善 5%、完整 Pipeline 体积改善 10% 三者任一达标即可进入 Pareto 非支配判断；门槛
可在 evaluator 配置中分别调整。缺失指标只记为 unavailable，不贡献改善；若所有
目标指标都缺失或都未达标，候选拒绝。这个协议只适用于新实验，不会重写历史决策。

EvolutionLoop 也会复用没有 `decision.json` 的最早未完成 round；已写入的
`profile.json`、`algorithm.json`、`sandbox.json` 和 `compression.json` 会先校验并
继续使用，避免进程崩溃后无条件新建 round 或重复执行相同算法。

Wan child 只修改 `transformer/`；未修改的 pipeline 组件会优先使用 hardlink，
跨文件系统时则引用同一台机器上不可变 Hugging Face snapshot 的绝对 symlink，
以避免在 NFS 上重复复制数十 GB 的 T5/VAE 权重。该策略写入 artifact metadata，
因此这种 child 是当前主机可重新加载的实验 artifact，不是脱离原始 snapshot 后
即可搬运的自包含发布包；需要可搬运包时应显式走 full-copy 路径并检查磁盘空间。

Dev run 完成后，对已存在的 baseline 和最终候选使用独立入口评测 Test manifest，
不要为了评测重新生成一个压缩候选：

```bash
HARNESS4H3_SOURCE_REVISION=$(git rev-parse HEAD) \\
  .venv/bin/python tools/evaluate_wan_checkpoint.py \\
  --config configs/evolutionary-compression.wan-formal-test.example.yaml \\
  --baseline /path/to/parent \\
  --candidate /path/to/final-child \\
  --output-root var/rsi/test/fixed-algorithm \\
  --json
```

该命令只写 `independent-evaluation.json` 和 evaluator 产物，不修改模型晋升状态；
晋升仍只由 Dev round 的独立评测和 `PromotionPolicy` 决定。

确定性探索算法的 FFN 保留比例由
`controller.explore_ffn_retention_ratio` 控制。Memory 中若发现上一轮模型大小
改善低于 10%，下一轮探索会自动收紧该比例，形成可审计的算法设计反馈，而不是
只重新选择一个 Block。

## RSI 算法设计记录

当前探索 Agent 不只返回一个固定操作。若 profile 同时提供层级敏感度和计算开销，
它可以生成最多 3 步的受限 `compose` 计划，例如先移除低敏感度的高开销 Block，再
对另一个瓶颈 Block 做 FFN 通道裁剪。计划中的每一步仍只能调用注册的
`prune_blocks` 或 `prune_ffn_channels`，不能嵌套 `compose`，也不能指定路径、设备、
命令或 evaluator。

通过 Sandbox 的计划会在 `rounds/Rxxxx/design.json` 写入稳定的算法 fingerprint、
诊断出的瓶颈、使用的 profile 字段、历史经验 ID、失败反馈、算法假设和最终计划。
组合计划的每一步分别写入 `checkpoint/step-00/`、`step-01/` 等目录；只有最后一步
通过完整性和独立评测，整个 Child 才能进入晋升判断。

真实 Diffusers Wan 的可重载 checkpoint 要求剩余 Transformer Block 的 FFN 宽度保持
对齐。组合算法先删除 Block 后，Wan operator 会依据当前 checkpoint 的 Block 数量把
FFN 步骤规范化到所有剩余 Block；该规范化后的参数会写入结构化压缩结果，Fake adapter
则保留其可局部裁剪的测试语义。

完成评测后，系统将实际质量/硬件收益和拒绝原因写入 `DesignGeneStore`。预测收益只
保留在算法候选 metadata 中，不能写入 DesignGene 的实测 benefit。质量失败会让后续
确定性 Agent 退回保守单 Block 计划；效率或体积不足则允许继续探索组合策略。因而
CPU fake 测试可以验证“反馈改变算法结构”和 artifact 可重放，但不能证明 Wan2.1
真实视频质量或硬件收益。

## 四组协议预检

如果四组 YAML 尚未建立，可以从一份已固定代码版本的基线配置生成它们。生成器只改
控制变量和独立输出目录，并把相对模型/评价路径固定为绝对路径：

```bash
.venv/bin/python tools/build_evolution_protocol_configs.py \
  --base-config configs/evolutionary-compression.wan-coding-formal-v13.yaml \
  --output-dir configs/generated-rsi \
  --run-root /data/models/harness4h3-evolutionary-compression-20261014 \
  --source-revision a9c1ebd
```

生成器要求基线使用真实 Wan evaluator 和非 deterministic 的 Coding Agent provider；
如果模型资产只在远程主机上，须在远程主机执行，或显式使用
`--allow-missing-assets` 只生成待补齐的配置，不能把它当作实验结果。

正式启动前，先用协议预检工具把四份配置锁成同一组可比条件。工具不会加载 Wan
模型或执行 GPU 推理；它会检查 checkpoint、Wan 配置、评价 manifest、dtype、设备、
seed、`8 × 2 = 16` 个评价单元、三次延迟测量和三轮预算，并要求四个控制臂分别命名为
`fixed`、`coding-agent`、`agent-memory`、`rsi`：

```bash
.venv/bin/python tools/validate_evolution_protocol.py \
  --run fixed=configs/rsi-fixed.yaml \
  --run coding-agent=configs/rsi-coding.yaml \
  --run agent-memory=configs/rsi-memory.yaml \
  --run rsi=configs/rsi-full.yaml \
  --rounds 3 --expected-cases 16 --latency-repeats 3 \
  --source-revision "$(git rev-parse HEAD)" \
  --write-manifests --output var/rsi/protocol.json --json
```

成功后每个 `output_root/run-manifest.json` 都会保存协议指纹、臂名称、代码 revision、
评价资产摘要、预算和控制变量。远程资产在本机不可见时，预检必须在实际运行主机上执行；
`--allow-missing-assets` 只适合生成待补齐的远程预检记录，不能把缺失资产当成已经完成的
实验。四组清单缺失、指纹不一致或臂名称不完整时，汇总器会将结果标为不可直接比较。

## 对比实验协议

要验证“算法演化”本身的收益，使用同一 checkpoint、manifest、seed、预算
和输出指标，分别建立四个独立 `output_root`：

1. `fixed`：deterministic agent，`exploit_ratio: 1.0`，关闭策略更新；
2. `coding-agent`：只接入结构化 provider，不传入历史经验；
3. `agent-memory`：接入 SkillHub/ExperienceStore，但固定搜索策略；
4. `rsi`：当前默认路径，算法 provider、经验检索和
   exploit/explore 策略都启用。

对应配置开关是 `memory_enabled` 和 `adapt_search_policy`；前三个 baseline
只需在各自 YAML 中关闭相应能力，且必须使用独立的 `output_root`。

每个 run 至少记录：当前 Active 的 `quality_score`、`quality_delta`、压缩后参数/模型
大小、平均生成耗时、有效实验数、失败实验数、总 GPU wall time，以及在新
prompt/新 seed manifest 上的迁移结果。汇总器还会分别记录
`active_model_id`、`last_candidate_model_id`、`historical_best_candidate_model_id`、
`rejected_model_ids` 和 `final_source`。`final` 只能来自 Active 的真实测量，或来自
明确以 Active 为 parent 的 baseline 测量；没有这些证据时字段保持 `null`，不能用
最后一个候选补齐。`rounds/*/decision.json`、
`experiences.jsonl` 和 `search-policy.json` 是对比实验的机器可读输入；不要
把不同 manifest 或不同质量阈值的结果直接横向比较。

正式实验还必须记录同一份代码版本。由于远程部署目录可能不是 Git checkout，
启动命令应显式设置 `HARNESS4H3_SOURCE_REVISION`（或在 evaluator 配置中设置
`source_revision`）；评测 recipe 和每个结果都会保存该值。缺失或不一致的代码
版本会让对比器将四路结果标为不可直接比较。

完成四路 run 后，用只读汇总器检查协议一致性并生成机器可读对比：

```bash
.venv/bin/python tools/compare_evolution_runs.py --json \
  --expected-cases 16 \
  --run fixed=var/rsi/fixed \
  --run coding-agent=var/rsi/coding-agent \
  --run agent-memory=var/rsi/agent-memory \
  --run rsi=var/rsi/rsi
```

汇总器不会把缺失指标填成零，也不会把被拒绝候选当作 active model；历史最佳候选
和最后候选只用于候选实验记录。Active 指针读取 `models/active.json` 的 `id` 字段；
不同 `evaluator_recipe_sha256`、代码 revision 或 case 数量不一致时，结果会标记为
不可直接比较。它还会检查四组 `run-manifest.json` 的协议指纹、唯一臂名称和预算；
当前 Active 没有可靠指标时，汇总器会明确报告数据缺失。

当前仓库先完成 Wan2.1-1.3B 的真实闭环；H3 迁移只依赖同一组
`WanModelAdapter`/`CompressionOperator`/`EvaluationFn` 边界，待 Wan 的
算法候选、质量证据和策略演化稳定后，再新增 H3 adapter，不修改现有 H3
operator 和 student training 路径。

不要让两个命令共享同一个 `output_root` 并发运行；每个实验使用独立的
output root，避免 parent checkpoint 和 round artifacts 互相覆盖。

## 设计边界

- 算法源码只能返回 `operator` 和 `args`，不能指定路径、设备、命令或
  evaluator；
- sandbox 先做 AST/import/输出约束，再在独立进程中启用资源上限和 Python audit
  deny，禁止 shell、网络、文件访问、动态代码和未 allowlist 的 import；当前主机
  未提供可用的 network namespace，因此正式部署仍必须放在远程主机级隔离环境中；
- 结构修改必须经过 forward shape、finite parameter 和 save/reload 校验；
- 失败算法同样写入 ExperienceStore，供后续 SkillHub 避免重复失败；
- 现有 H3 operator、Wan student training 和 remote campaign 接口保持不变。
