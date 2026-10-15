# Harness4H3 TICA-Prune v2 验收报告

**日期：** 2026-10-15
**分支：** `codex/tica-prune-v2`
**基线：** `b6b8b18`

## 结论

本轮修复了会让 TICA 搜索和 RSI 算法比较失真的几类确定性问题，但没有重新启动 Wan GPU 全量实验，也没有修改历史四组实验的决策记录。因此当前结论是：代码层面的闭环和隔离规则已经明显更可信；真实 Wan 上的 RSI 算法改进结论仍需要新的受控实验，不能用本轮 CPU 或旧日志宣称已经证明。

## 实际修改

- TICA 快速筛选现在使用独立的 `local_relative_output_error / profile_relative_error` 近似风险；完整评测必须返回 `normalized_mse / normalized_mse`、有限数值、`valid=true` 且标记为非近似。完整风险超标、尺度不匹配、缺失、计算失败或交互测量无效都会拒绝候选，不能进入 Pareto Front。
- 完整 TICA 模型评测现在把风险契约同时写在结果顶层和 `output_metrics` 内，确保搜索器能读到真实模型输出的完整 normalized MSE，而不是把已有测量误判成风险缺失。
- Evolution 只使用 Calibration 和 Validation。Evolution 不读取 Test manifest 或 Test tensor，也不把 Test 写入循环的 `tica-splits.json`；Test 由独立的 `final-evaluate` 冻结流程读取，只输出报告，不更新 ModelStore、AlgorithmStore、Memory、SearchPolicy 或 Agent 上下文。
- TICA 的算法验证不再只比较 Sandbox 产生的第一个 CompressionPlan。父算法和子算法现在分别调用完整的 `TICAPruner.search()`，使用同一个父 checkpoint、Calibration profile、搜索空间、种子、预算和独立缓存命名空间，再用 Validation 评估选中的候选。
- Controller 的 TICA 请求明确返回受限的 `tica_prune` 搜索规则；返回规则经过 Sandbox 后实际传入搜索器。普通模式仍保持原有三类基础算子协议。
- 搜索结果新增真实深度、逐层候选/可行数量、终止原因、快速/完整/交互评测次数、共享 profiling 成本和总成本。默认四个 baseline 作为候选种子共享搜索预算，不再和四次独立算法实验混淆。
- Wan 指标补充 Transformer 与完整 Pipeline 的范围说明、固定组件大小和 inode 去重；峰值显存只接受实际 CUDA 设备上的有效正数，缺失或不可用时保持缺失，不填零。新增无副作用的 Wan checkpoint/manifest 预检，实际模型 FFN 和 Hook 仍需在目标 GPU 主机运行时验证。

## 验收结果

本轮使用仓库虚拟环境 `/Users/huangjiahao/MinMax-H3/Harness4H3/.venv`，其中有 `torch 2.14.0` 和 `diffusers 0.39.0`；`torch.cuda.is_available()` 为 `False`，因此没有加载大型 Wan，也没有启动 GPU。实际执行结果如下：

- CPU Fake Wan 端到端（TICA recursive loop、真实 Fake 模型压缩/重载、Calibration 误差和交互评测）：`22 passed`。
- 重点回归套件：`89 passed, 1 skipped, 3 deselected in 1.97s`。
- TICA、风险、Controller、数据隔离重点套件：`36 passed, 3 deselected in 1.64s`。
- CPU 全量测试：`985 passed, 4 skipped in 131.27s`。
- `python3 -m compileall -q harness4h3 h3_training research tools`：通过。
- `git diff --check`：通过。
- CLI 帮助检查通过，已出现 `final-evaluate` 命令。
- 系统 Python 3.9 的一次独立收集曾因没有可选依赖得到 `692 collected, 37 errors`；该解释器没有被用于最终验收，最终验收使用了上面的仓库虚拟环境。

默认 TICA 示例配置的离线搜索轨迹为：4 个 baseline 根节点、`max_full_evals=4`，实际深度 1，完整评测 4 次，因 `budget_exhausted:full` 终止。配置中的 `max_search_depth=3` 只是上限，不代表实际一定能到第三层。

## 当前实验能支持什么

已有独立候选复测显示 Agent + Memory 候选约有 21.46% 延迟、7.06% 峰值显存和 5.14% Pipeline 体积改善；完整 RSI 候选约有 3.30%、0.49% 和 0.36% 改善，但质量变化分别约为 -1.95% 和 -3.71%。这能支持“Memory 候选在那次复测中压缩得更激进、硬件收益更大”，不能支持“经验积累已经带来算法设计能力提升”。两者使用了不同压缩位置和幅度，且设备和历史协议存在差异；这是混杂因素，不是因果证明。

最直接的证据是：Memory 候选做了全 block FFN 宽度压缩，保留约 18.75% channel；RSI 候选只删除了一个 block。Memory 因此有更大的硬件收益机会。至于经验检索、探索/利用选择或失败反馈是否造成了这个差异，目前仍是待验证解释。

## 对最终问题的回答

1. **是否具备可信闭环？** 代码层面已经形成“Calibration profile → 受限算法规则 → Sandbox → 完整 TICA 搜索 → Validation 独立比较 → Promotion/经验记录”的闭环，并且 Test 已从自适应路径隔离。由于本轮没有真实 Wan GPU 复跑，不能把它表述为已经完成真实环境验收的闭环。

2. **为什么 Agent + Memory 优于完整 RSI？** 已有证据只说明 Memory 选择的候选更激进、目标范围不同，所以硬件收益更大；“Memory 利用了历史经验”“完整 RSI 的探索策略损伤了质量”等只能作为假设，当前日志和设备差异不足以证明。

3. **证明持续改进所需的最小实验？** 从同一个冻结 parent 开始，固定 checkpoint、Controller 版本、Calibration/Validation/Test manifest、搜索空间、种子、预算和评测规则，做三条配对轨迹：Memory-off、真实 Memory、错误关联或打乱的 Memory 控制。每条至少连续两轮保存检索到的经验 ID、算法源码、搜索状态、Pareto Front、Validation 结果和晋升决策；最后冻结算法和模型，只在一次未参与搜索的 Test 上评估。只有真实 Memory 在重复轨迹中稳定超过两个控制且质量门槛不退化，才可以声称经验积累改善了压缩算法。

本报告不追认旧实验，也不把 CPU Fake 的结构/质量代理指标当作真实 Wan 延迟、显存或视频质量证据。
