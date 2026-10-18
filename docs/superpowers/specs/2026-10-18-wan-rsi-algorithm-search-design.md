# Wan RSI 搜索改进设计

状态：按用户 2026-10-18 的明确任务范围执行。

## 证据基线

远程 `remote-rsi-f082f19-20261016` 的 R0001–R0010 每轮生成 12 个计划、尝试 8 个。80/80 条 AlgorithmModelEvaluator 测量均无效：70 条因 Diffusers Pipeline 的 `parameters` 映射被当作函数调用，5 条真实 Wan 结构 Head Pruning reload 维度不匹配，5 条 Hybrid 记录 `kv_token_selection_failed`。尽管没有有效分数，旧选择器仍从无效记录回退选择，并进入独立视频评测。奇数轮测 rank 0–7，偶数轮测 11–4；外层报告索引还把 rank 顺序位置误记为候选索引。

各轮 A0002–A0010 的 `propose` 主体相同，选择最低 Block importance；搜索候选池仍为固定的 12 个计划。源码尾部只按算法 ID 奇偶反转排序，改变预算截断顺序，没有按评测证据改变候选生成或评分。十个算法都以 A0001 为父代。SearchSpace P0001–P0008 均提议相同的 `timestep_block_gate`，都以同一 finite-check 错误被拒绝。

Block Pruning 的独立视频结果显示 temporal `0.955881 → 0.926123`，延迟约改善 4.08%，参数下降约 3.27%，逻辑文件尺寸下降约 0.36%，峰值显存从约 18.77 GB 增至约 19.28 GB。Head Gating 的 semantic `0.594424 → 0.547947`，总分下降约 3.06%，延迟仅改善约 1.47%，参数和模型尺寸基本不变，显存上升。历史完整评测只有 1 Prompt、1 Seed、0 warmup、1 latency repeat。旧 profile 有 30 组整体 Block 灵敏度、360 组逐 Block/Head 灵敏度和 3 个 Token 能量估计，但时间步灵敏度仅测参考 `block.0`。

单独的真实 Wan runtime probe 证明 Token Pruning、Head Gating、Sparse Runtime 和 Attention Window 在该 probe 中实际执行：token 选中 360/480、跳过 120；head 选中 1080/2160；sparse 选中 360/720；window 选中 K/V 360/720。该 probe 不包含 Token Merge 或 Timestep Attention，也不是 10 轮所选 Head Gating 的计数。10 轮内部候选测量在有效性检查前失败，不能将它们写成已执行的 runtime 路径。

## 设计

沿用现有 SearchSpace、Operator Registry、Algorithm Program、Sandbox、AlgorithmModelEvaluator、EvolutionLoop、独立 Wan Video Evaluator 与 PromotionPolicy。保留现有算子集合和质量阈值。Profiler 增加按 Block 和高噪声/中间/低噪声区间的配对输出敏感度记录；Agent 候选使用逐 Block/Head 与 Token/Attention 灵敏度，不对缺失证据作猜测。标准搜索候选池可含结构、Head、Token/KV/Sparse/Window、Timestep、Sampling 和受支持的组合，明确保留忽略/不兼容原因。

每轮仍最多执行 8 个 full candidate measurements。预算调度基于同一基础 Checkpoint 上各家族的实际评测次数，优先偿还探索欠账，再遵循 Agent 家族内排序。相同 parent checkpoint 下所有候选使用相同 Calibration probes。有效候选以共同质量、效率代理和风险指标构造 Pareto front；无效候选永不参与排名。若没有候选通过 Calibration 质量门槛，可选一个有效 Pareto near-miss 进入独立 Validation，明确标记为仅供确认；PromotionPolicy 仍执行原始质量与硬件门槛。若全部无效则失败关闭，不执行任意候选。

短程真实实验在同一 output root 顺序执行 1 轮固定算法参考和 3 轮 vLLM 算法改进。每轮以原始 Wan2.1-1.3B Checkpoint 为模型 parent、8 次搜索评测预算比较算法；后代源码必须记录父算法 ID、父 Source SHA、Controller 模型及服务版本、API 响应 ID。搜索 Calibration 使用多个固定 transformer probe seeds；独立 Validation 使用至少 3 个不同 Prompt、2 个 Seed，延迟预热 2 次并重复 5 次。两阶段使用相同清单和协议，且不改动历史产物或放宽质量阈值。

## 验收

- 复核并保存 R0001–R0010 全部生成、尝试、跳过、失败、最终候选、视频指标和 SearchSpace 提案记录。
- 测试 Pipeline finite check、全无效失败关闭、原始候选索引、跨轮家族公平预算、Pareto 有效性、敏感度驱动候选和运行时计数。
- 真实 vLLM 端点产生 API 响应；R0002–R0004 保存真实模型身份、response ID、父代来源与子代源码差异。
- 输出报告以真实执行的候选和 GPU 评测作判断；若没有 Pareto 或模型晋升，照实报告负结果。
