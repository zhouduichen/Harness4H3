# Harness4H3 进化压缩评测审查报告

**日期：** 2026-10-14
**范围：** EvoCompress 统计器、Wan evaluator、PromotionPolicy，以及当前工作区可读取的实验记录。

## 结论摘要

本轮修复了两个会直接改变实验结论的确定性错误：汇总器之前漏读 `active.json` 的 `id`，并且在 Active 没有指标时把最后一个 Child 当成最终结果；旧晋升协议把效率改善和模型体积改善做成 AND，导致有实际延迟/显存收益但体积改善不足的候选被拒绝。

新的后续协议已经改为：完整性、真实独立评测、质量下限和硬件退化是硬约束；延迟 5%、显存 5%、完整 Pipeline 体积 10% 三项中任一达标后，再检查 Pareto 非支配。历史 `decision.json` 没有被回写，也没有用新协议追认旧实验。

## 本轮修改

- `tools/compare_evolution_runs.py`
  - 读取 `models/active.json` 的 `id`，并保留旧字段的只读兼容。
  - 单独报告 Active、最后候选、历史最佳候选、被拒绝 Child 和 `final_source`。
  - 只有 Active 自身测量，或明确以 Active 为 parent 的 baseline 测量，才可以形成最终成绩。
  - 没有可靠关联时最终指标为 `null`，不会从最后一个候选补齐。
- `harness4h3/evolution/evaluator.py`
  - 新增可分别配置的 latency/memory/model-size 阈值，保留旧 `min_efficiency_improvement` 配置兼容。
  - 质量子指标存在可靠 Parent 对照时也执行 98% 下限。
  - 缺失指标、无效零显存、非目标硬件退化和 Pareto 支配都有明确记录。
  - 将 Transformer 参数量/权重大小与完整 Pipeline 体积分开记录。
  - 硬链接和软链接按 inode 去重；CUDA telemetry 不可用时峰值显存为 `null`，不再返回零。
- 测试覆盖了 Active 已晋升、未晋升、连续拒绝、缺失 baseline、单目标达标、质量不足、硬件退化、零/缺失指标、Pareto 支配和体积去重。
- `docs/evolutionary-compression.md` 与 `docs/experiment-schema.md` 已写入新的 Evaluation Recipe 和字段定义。

在本轮后续实现中又补上了 RSI 算法设计层：

- Sandbox 现在允许最多 3 步、不可嵌套的 `compose` 计划；每一步仍只能调用已注册的 Block/FFN 压缩 operator。
- 确定性 Agent 会结合层敏感度、FLOPs/延迟和 Channel profile 生成组合算法；出现质量门槛失败时会退回保守的单 Block 策略。
- 每个可执行 round 新增 `design.json`，记录瓶颈、证据字段、历史反馈、计划和稳定 fingerprint；实际结果会生成 `DesignGene`，失败设计也会作为负面经验检索。
- 组合计划逐步保存 `checkpoint/step-00/` 等 artifact；只有最终 Child 进入原有完整性、真实评测和 PromotionPolicy 判断，算法新颖性本身不能晋升模型。

## 验证结果

- 相关 Wan evaluator、Sandbox、组合 operator、算法进化和 EvoCompress 集成回归：`29 passed`。
- CPU 全量：`928 passed, 4 skipped`，耗时 `99.66 秒`。
- 4 个 skip 均是既有环境条件：真实 Wan GPU smoke、两项 CUDA-only training 测试和 CUDA RNG parity；不是本轮失败。
- RSI 相关目录的 `.venv/bin/python -m compileall` 通过，`git diff --check` 通过。
- `git diff --check` 通过。

没有执行四组完整 GPU 实验，也没有修改历史实验记录。

## 四组 Wan 实验能支持什么

在当前工作区中，没有找到 Fixed、Coding Agent、完整 RSI、Agent + Memory 四组实验各自完整的机器可读输出目录，也没有找到能把每轮算法源码、profile、16 cases、每 case 3 次 latency、CUDA device/负载和拒绝原因对应起来的成套记录。仓库里现有的 `var/evolutionary-compression-fake` 是离线 fake 结果，不能证明真实 Wan 的语义质量或硬件收益；现有真实 Wan 配置说明了评测方法和预算，但不是四组结果本身。

因此，题目中给出的四个百分比可以作为待核对的实验摘要，不能在本审查中当成已由仓库证据复核的统计结论。尤其不能从这些摘要单独推出 RSI 的算法设计能力。

对 Agent + Memory 优于完整 RSI 的解释目前分两层：

1. 有代码层面的事实：Memory 路径会检索架构/算子相关经验，搜索策略会根据上一轮可行性和收益调整 exploit/explore；完整 RSI 同时打开算法生成、经验检索和搜索策略适应。因此两者并非只差一个开关，候选来源和探索轨迹可能不同。
2. 仍未被实验日志证明的解释：Memory 组可能选中了更合适的压缩位置、剪枝幅度或 FFN 保留比例，也可能只是四组的模型版本、GPU 负载、评测 recipe、随机种子或候选数量不一致。没有逐轮 `operator_args`、profile 敏感度、经验检索 ID、设备 telemetry 和独立 quality 记录，不能把其中任何一个说成原因。

## 三个验收问题

### 1. 当前系统是否已经具备可信的自动压缩与模型晋升闭环？

实现层面现在已经有“profile 诊断 → 设计组合算法 → 沙箱 → 逐步压缩 → 独立评测 → 晋升/拒绝 → DesignGene/ExperienceStore 反馈”的闭环，并且本轮修复了最终结果污染和晋升门槛错误。就当前真实实验的证据状态而言，答案仍是“尚未完全具备可信闭环”：目前只能证明机制可执行、可追溯，真实四路对照记录仍不完整。完成一轮口径一致的 Wan 评测后，才能把它称为经过实验验证的可信闭环。

### 2. 为什么 Agent + Memory 可能优于完整 RSI？

仓库能支持的最强结论是：Memory 现在不仅检索历史经验，还能把质量失败、效率失败和设计 fingerprint 传给下一轮；完整 RSI 额外改变搜索策略，并允许 profile 驱动的组合算法和质量失败后的保守回退。因此两组的候选分布确实可能不同。仓库仍没有四组真实逐轮证据，无法判断优势究竟来自经验复用、探索策略、压缩位置、剪枝幅度，还是设备/recipe 差异。21.5% 延迟和 7.1% 显存改善如果复核成立，只能说明该候选在当时测量条件下硬件收益明显，不能单独证明 RSI 已经获得或失去算法设计优势。

### 3. 证明“通过经验积累持续改进压缩算法”所需的最小实验是什么？

最小的可信设计是三条严格匹配的短轨迹：Memory-off 控制、真实 Memory、以及使用相同数量但打乱/错误关联经验的伪 Memory。三条轨迹从同一 checkpoint 开始，固定 controller 版本、GPU、Wan 配置、manifest、16 cases、每 case 3 次 latency、质量阈值和随机种子，各跑至少 3 个连续 round；每轮保存算法源码与参数、profile、检索到的经验 ID、搜索策略状态、独立评测和晋升结果。最终只比较 Active，并在独立 held-out manifest 上复测。

如果真实 Memory 在至少两个连续 round 中相对两个对照都产生可复现的有效候选/更好的 Pareto 进展，同时质量不低于 98%、设备和 recipe 无漂移，才有足够证据声称“经验积累改善了压缩算法”。单个成功 Child 或单次四组均值不足以支持这个结论。
