# Evolution Evaluation and Promotion Audit Design

**Date:** 2026-10-14

## Goal

修复 Harness4H3 的进化压缩统计和后续晋升协议，使实验报告不会把被拒绝的 Child 当成最终模型，并使 Wan 评测中的质量、性能和模型体积口径可审计。历史实验的 `decision.json` 不回写、不用新协议追认。

## Scope and boundaries

- 修改范围限于 `tools/compare_evolution_runs.py`、`harness4h3/evolution/evaluator.py`、必要的测试和 EvoCompress 评测文档。
- 仅在确有必要时补充 `harness4h3/archive/model_store.py` 的只读 Active 指针契约；不改变已有 `active.json` 的 `{"id": ...}` 格式。
- 不修改 Student、H3 训练、蒸馏和远程 campaign 代码。
- 当前工作区已有的未提交修改视为用户资产；提交时只 stage 本设计和本任务产生的文件。

## Data-flow design

### Run summary

`compare_evolution_runs.py` 读取 `models/active.json` 的 `id` 字段，并验证对应候选文件。汇总结果明确包含：

- 当前 `active_model_id` 及其指标是否可用；
- 最后一轮候选的 ID、状态和是否被拒绝；
- 按可复现的 `search_score`/质量优先顺序选出的历史最佳候选；
- 所有被拒绝 Child 的 ID；
- `final` 及其 `final_source`。

`final` 只允许来自当前 Active 的实测指标，或来自明确关联到该 Active parent 的独立评测中的 `baseline_quality_score` / `baseline_hardware_metrics`。如果两者都缺失，最终指标保持 `null` 并提供数据缺失原因；最后候选和历史最佳候选只用于候选实验记录，不能填充最终成绩。

### Promotion policy

后续协议的硬约束按顺序检查：

1. Child 完整性通过；
2. `valid is True` 的独立评测存在；
3. aggregate quality 不低于 Parent 的 `quality_floor_ratio`，默认 0.98；
4. 可用的 semantic、temporal、motion 等质量子指标不低于相同质量下限；
5. 每个可比较硬件指标的退化不超过 `max_metric_regression_ratio`。

通过硬约束后，候选必须在 latency、peak memory、完整 Pipeline model size 三个目标中至少一个达到各自门槛。默认门槛为 latency 5%、显存 5%、model size 10%。缺失指标不贡献 improvement；缺失全部目标指标或没有任何目标达到门槛则拒绝。若候选被当前 Pareto front 支配，仍拒绝。所有失败原因和实际 improvement 写入 decision。

保留 `min_efficiency_improvement` 作为旧配置的兼容输入，映射到 latency 和 memory 两项；新配置可分别使用 `min_latency_improvement`、`min_memory_improvement`。

### Metric definitions

- `transformer_parameter_count` / `parameter_count`：压缩目标 Transformer 的参数量，明确 `parameter_count_scope=transformer`。
- `transformer_model_size_gb`：只统计目标 Transformer 目录的去重文件大小。
- `pipeline_model_size_gb` / 兼容字段 `model_size_gb`：完整 Pipeline 目录的逻辑文件大小；硬链接和指向同一 inode 的软链接只计一次。
- `peak_memory_gb_max`：在 `generation_device` 对应 CUDA device 上清零并读取的每 case 峰值最大值。CUDA 不可用、device 不匹配或 telemetry 异常时写 `null`，不写零。
- latency：固定 manifest、seed、warmup 和 repeat 下的 denoising + VAE decode wall time；prompt embedding 是否复用由 recipe 明确记录。

## Testing design

- 汇总器测试覆盖 `active.json.id`、Active 已测量、Active 未测量、关联 baseline 恢复、连续拒绝、最后 Child 被拒绝、缺失基线和候选/最终字段隔离。
- 晋升测试覆盖 latency-only、memory-only、model-size-only、质量不足、硬件退化、指标缺失、零显存、Pareto 支配和多项同时达标。
- Wan evaluator 测试覆盖真实 device 的 peak memory、不可用 telemetry 为 `null`、去重文件大小和 Transformer/Pipeline 字段。
- 运行相关单测、CPU 全量测试、Python 编译和 CI 配置对应的依赖/命令检查；不运行四组完整 GPU 实验。

## Evidence conclusion boundary

本轮只审查工作区可读的日志、配置、代码和历史记录。若缺少四组真实 RSI 对照的逐 case 评测、profile、算法历史和拒绝记录，报告会明确区分“代码可证明的原因”“现有记录支持的观察”和“尚无证据的假设”，并给出证明经验积累有效所需的最小后续实验。
