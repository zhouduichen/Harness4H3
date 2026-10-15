# Algorithm Discovery 与 Search-Space Evolution 设计

**日期：** 2026-10-15  
**基线：** `codex/tica-prune-v2` + `codex/tica-p0-acceptance` 必要修复  
**实验对象：** Wan2.1-T2V-1.3B（Fake Wan CPU 作为无 GPU 时的可执行验证对象）

## 目标

把 Harness4H3 当前“Agent 生成 Recipe、TICA 固定生成候选”的路径改成可执行的 Algorithm Program 路径：Agent 生成的源码能够在真实搜索循环中决定候选生成、重要性评分、候选排序和停止条件；已验证的 SearchSpace Operator 能被后续算法读取、生成候选并执行。保留现有 Sandbox、Profiling、Evaluator、Memory、SkillHub、AlgorithmStore、ModelStore 和 EvolutionLoop，不另建隔离、记忆或规划系统。

## 当前边界与选择

TICA-Prune 保留为人工 Baseline 和第一代算法，但只负责风险测量、预算记账、候选评测和兼容旧 Recipe。新的 Algorithm Program 可以完全绕过 `_root_states()` 和 `_expand_state()` 的固定候选规则。

首个非等价新操作采用 `timestep_block_gate`。已检查当前 Wan Diffusers 接口：`WanTransformer3DModel.forward` 接受原始 `timestep`，并在调用每个 Transformer Block 前生成 `timestep_proj`。因此 Adapter 可以在保存配置中记录按时间步跳过 Block 的规则，重载时恢复运行时 gate。该操作不删除参数，参数量减少必须记录为零；动态跳过率、Block 执行次数、实际延迟和质量单独记录。

## 设计

### 1. Algorithm Program

Algorithm source 保持 `propose(profile, history)` 兼容，并支持新的可选入口：

```python
def search(profile, history, state, search_space):
    return {
        "candidates": [
            {"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
        ],
        "rank": [0],
        "stop": False,
    }
```

约定如下：

- `profile` 是当前 Calibration/Validation split 的诊断摘要，包含 Block 风险、时间步敏感性、FFN 贡献、成本和适配器 provenance；
- `history` 是已有 Memory/SkillHub 检索结果；
- `state` 包含当前搜索深度、父状态、已评测候选、预算快照和当前 frontier；
- `search_space` 是当前 `SearchSpaceVersion` 的 OperatorSpec 列表和版本 ID；
- `candidates` 必须是可 JSON 序列化的压缩计划；候选 operator 必须存在于当前有效 registry；
- `rank` 是候选下标顺序，未列出的候选按生成顺序追加；
- `stop=true` 只表示算法不再生成后继候选，预算和安全边界仍由 Harness 控制。

Sandbox 只增加通用的指定入口/JSON payload 执行能力，继续复用当前 AST 策略、子进程隔离、审计 hook、CPU/地址空间/文件大小/超时限制。入口缺失时回退到 `propose`，因此旧算法和旧断点仍可恢复。

TICA 搜索器保留以下 Harness-owned 责任：候选计划 schema 校验、Operator Registry 校验、模型执行、fast/full risk scale 校验、交互测量、共享预算、Pareto 选择和 Test 隔离。Algorithm-owned 责任是候选集合、评分、顺序和停止条件。每次生成和排序都保存源码 digest、调用 payload digest、返回候选及轨迹，便于证明结果由源码产生。

### 2. SearchSpace Evolution

`SearchSpaceManager.active` 是唯一能力来源。每轮初始化时：

1. 将 active `OperatorSpec` 注入 Sandbox 的扩展 schema；
2. 由 `registry_for(active)` 创建实际可执行的 `WanCompressionRegistry`；
3. 将同一份 spec 摘要传入 Algorithm Program；
4. 对返回计划同时执行 schema、registry、Adapter 和 reload 检查。

Search-space proposal 继续使用已有 `SearchSpaceProposal -> validate -> register -> SearchSpaceVersion` 流程。Proposal 必须包含原因、适用条件、所需诊断、预期收益、参数 schema 和 `implementation_source`。验证阶段通过现有 Sandbox 执行 source 生成一个合法计划，再由可信 Adapter factory 执行并重载检查；只有 validated proposal 才能注册并激活。新版本会被写入下一轮上下文，不能只作为名称或 metadata 存在。

### 3. `timestep_block_gate` Operator

计划 schema：

```json
{
  "operator": "timestep_block_gate",
  "args": {
    "rules": [
      {"block_id": "block.0", "min_timestep": 0.0, "max_timestep": 0.3}
    ],
    "policy": "skip_when_in_range"
  }
}
```

可信执行路径：

- Fake Wan：在模型 forward 中按归一化 timestep 跳过指定 block，并把规则、执行次数和跳过次数写入 checkpoint config；Fake loader 根据 config 重建运行行为；
- Diffusers Wan：Adapter 在 Transformer config 中保存规则，load/reload 后在 Transformer forward 上保留原始 timestep，并为实际 Block 安装 gate；未知 block layout 或无法恢复时失败关闭；
- `CompressionResult.structural_change` 明确标记 `kind=runtime_timestep_gating`、`parameter_reduction_ratio=0`、规则 digest 和运行时统计；
- Evaluator 将静态结构指标和运行时指标分开，禁止把动态跳过率冒充参数量下降。

### 4. Algorithm Evolution 与 Model Evolution

算法验证和模型晋升使用两个明确的状态边界：

- Algorithm Evolution：固定 parent checkpoint、Calibration profile、Validation profile、SearchSpaceVersion、seed 和搜索预算；父源码与子源码分别运行完整搜索，比较各自选中的候选和 Validation 结果；失败子代保留但不进入 SkillHub；
- Model Evolution：只接收已执行的压缩结果和现有 PromotionPolicy 的质量/效率/完整性判断；只有模型候选被接受时才更新 Active Model；算法验证成功本身不会改变 Active Model；
- 每个子代继承 `parent_algorithm_id`、`parent_source_sha256`、source diff、SearchSpaceVersion、实验反馈和验证记录；成功算法进入 SkillHub，供后续 Agent 使用。

## 错误与恢复

- 算法入口、返回格式或 operator 不合法：记录 `sandbox`/`algorithm_output_invalid`，不执行模型；
- 生成了未注册能力：记录 `operator_not_in_search_space`，本轮拒绝，不隐式注册；
- 新算子实现、Adapter 结构或重载失败：Proposal 标记 rejected，不激活 SearchSpaceVersion；
- 动态 gate 规则不能恢复：压缩结果失败，不能晋升模型；
- 搜索中断：沿用现有 round state、缓存和断点恢复，已完成候选不重复执行；
- 无 GPU/真实模型资产：运行 Fake CPU 验证并在报告中标注 `offline_simulation=true`，不伪造真实 Wan 结果。

## 测试与验收证据

新增单元和集成测试覆盖：

1. 两段 `search` 源码在相同 profile/state 下生成不同候选集合或顺序，且记录不同 source digest；
2. Algorithm Program 的评分/生成/排序代码在真实 TICA 搜索循环中被多次调用；
3. Agent proposal 经过 Sandbox、SearchSpaceManager、registry、Adapter、保存和重载后，`timestep_block_gate` 真的改变执行轨迹；
4. 连续至少三代源码继承 parent source 并读取真实 experiment feedback；
5. 已注册新 SearchSpaceVersion 被下一代算法读取和使用；
6. 父子算法使用相同模型、预算、Calibration 和独立 Validation；ModelStore 晋升与 AlgorithmStore/SkillHub 状态相互独立；
7. 旧 `propose`、普通 Block/FFN/Compose、Evaluation、Memory、Sandbox 和断点恢复测试保持通过；
8. Fake Wan CPU 端到端产出算法源码、search trace、compression、reload、evaluation 和 decision 证据；真实 Wan 配置和启动命令可用于 GPU 主机，但缺少 GPU 时不宣称真实性能结果。

## 取舍

不把 TICA 的内部实现抽成大型插件框架，不增加第二套 Sandbox/Memory/Planner，也不把动态 gate 包装成静态剪枝。后续 Agent 若提出其他能力，沿用同一 `OperatorSpec`、Sandbox、可信 factory 和 Adapter reload contract 扩展。
