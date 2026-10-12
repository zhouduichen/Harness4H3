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

Dev run 完成后，对已存在的 baseline 和最终候选使用独立入口评测 Test manifest，
不要为了评测重新生成一个压缩候选：

```bash
HARNESS4H3_SOURCE_REVISION=$(git rev-parse HEAD) \\
  .venv/bin/python tools/evaluate_wan_checkpoint.py \\
  --config configs/evolutionary-compression.wan-formal-test.yaml \\
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

## 对比实验协议

要验证“算法演化”本身的收益，使用同一 checkpoint、manifest、seed、预算
和输出指标，分别建立四个独立 `output_root`：

1. `fixed-algorithm`：deterministic agent，`exploit_ratio: 1.0`，关闭策略更新；
2. `coding-agent`：只接入结构化 provider，不传入历史经验；
3. `agent-memory`：接入 SkillHub/ExperienceStore，但固定搜索策略；
4. `algorithm-evolution`：当前默认路径，算法 provider、经验检索和
   exploit/explore 策略都启用。

对应配置开关是 `memory_enabled` 和 `adapt_search_policy`；前三个 baseline
只需在各自 YAML 中关闭相应能力，且必须使用独立的 `output_root`。

每个 run 至少记录：最终 `quality_score`、`quality_delta`、压缩后参数/模型
大小、平均生成耗时、有效实验数、失败实验数、总 GPU wall time，以及在新
prompt/新 seed manifest 上的迁移结果。`rounds/*/decision.json`、
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
  --run fixed-algorithm=var/rsi/fixed-algorithm \
  --run coding-agent=var/rsi/coding-agent \
  --run agent-memory=var/rsi/agent-memory \
  --run algorithm-evolution=var/rsi/algorithm-evolution
```

汇总器不会把缺失指标填成零，也不会把被拒绝候选当作 active model；不同
`evaluator_recipe_sha256` 或 case 数量不一致时，结果会标记为不可直接比较。

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
