# 受限组合压缩算法与经验反馈设计

**日期：** 2026-10-14

## 目标

在现有 Wan 演化闭环中增加一个可审计的“算法设计”层，使系统能根据层级 profile、硬件瓶颈和历史成功/失败记录，生成由多个安全注册操作组成的压缩算法。该层必须能在 CPU fake 模型上重放和验证，但不能把 fake 结果当作 Wan2.1-T2V-1.3B 的真实收益。

## 当前缺口

现有 `AlgorithmAgent` 已接收 Block/Channel/时间步敏感度和历史经验，但 Sandbox 只允许单个 `prune_blocks` 或 `prune_ffn_channels` 操作。`SearchPolicy` 主要调整 explore/exploit 比例，`DesignGeneStore` 也没有进入检索路径。因此当前实现仍主要是受限操作搜索，尚不足以证明后续算法结构会因实验反馈而改变。

## 设计

### 1. 受限组合计划

新增 `compose` 计划。计划包含 2–3 个有序的基础操作，每一步只能是现有 `prune_blocks` 或 `prune_ffn_channels`，禁止嵌套 `compose`、路径、设备、命令和 evaluator 字段。组合执行时每一步产生独立 artifact，只有最后一步的 checkpoint 可以成为候选 Child；任一步失败都拒绝整组计划并保留失败原因。

单操作计划保持向后兼容，旧算法文件和 CLI 仍可执行。

### 2. 算法设计记录

每个成功进入 Sandbox 的计划生成 `design.json`，记录：

- `algorithm_id`、父算法、搜索选择和稳定 fingerprint；
- 诊断出的主要瓶颈；
- 使用的 profile 字段和历史经验 ID；
- 有序计划、假设和预期目标；
- 从历史失败中提取的反馈。

该记录同时写入 `ExperienceRecord.training`，并生成一个 `DesignGene`。DesignGene 的收益、局限和风险必须来自实际评测/决策，不得把预测收益写成实测收益。

### 3. 反馈驱动的确定性算法

离线 Agent 保持完全确定性，但不再只有一个静态模板：

- 先用敏感度、FLOPs 和层延迟选择低敏感度且具备瓶颈贡献的 Block；
- 在探索路径中，若 profile 支持，则组合 Block 裁剪和另一个 Block 的 FFN 通道裁剪；
- 若历史包含质量下限失败，则退回较保守的单 Block 计划；
- 若历史包含体积或效率不足，则提高组合强度或增加 FFN 步骤；
- 计划 fingerprint、父算法和反馈摘要随候选保存。

结构化 Provider 使用同一组计划约束，仍不能改变 parent、评测器或执行环境。

### 4. 经验检索

`SkillHub.search()` 同时检索 ExperienceRecord 和 DesignGene，按架构、操作族、实际 reward/confidence 和时间排序。检索结果明确标记来源类型，失败经验不能被误认为成功经验。

### 5. 不变式

- 模型晋升仍完全由现有真实独立评测和 `PromotionPolicy` 决定；
- rejected Child 永远不会成为 Parent；
- fake evaluator 继续显式标记 `offline_simulation`；
- 不修改 H3 operator、Wan student training、蒸馏和历史实验决策；
- 组合计划的每一步、最终 checkpoint hash、评测 recipe 和代码 revision 都进入 round artifacts。

## 验收

CPU 测试必须覆盖：组合计划边界、嵌套/超长/不安全计划拒绝、逐步 artifact 和失败回滚语义、profile 驱动的算法生成、质量失败与效率失败导致的不同后续算法 fingerprint、DesignGene 检索、round 重启和旧单操作兼容。

这些测试只验证 RSI 机制的可执行性与可追溯性，不证明真实 Wan 的质量或硬件收益。真实结论仍需相同预算、相同评测配方下的四组 Wan 对照实验。
