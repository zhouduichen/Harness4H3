# Harness4H3 进化压缩评测审查报告

**日期：** 2026-10-14
**范围：** 模型归档与对照统计、Wan evaluator、PromotionPolicy、Evaluation Recipe，以及已有真实 Wan 实验记录。

## 结论先说

本轮修复了会直接污染实验结论的确定性错误，但目前还不能据此声称 RSI 已经带来了可重复的算法设计能力提升。

实现层面，系统现在能区分 Active、最后候选、历史最佳候选和被拒绝 Child；被拒绝 Child 不会再被统计成最终模型。后续晋升协议也已改成“质量和测量完整性是硬约束，延迟、显存、Pipeline 体积三个目标任一达到门槛即可进入 Pareto 判断”。历史 `decision.json` 没有回写，也没有用新协议追认旧实验。

实验层面，已有日志支持“Memory 候选曾经比 RSI 候选获得更大的硬件收益”这一描述，但不能支持“收益来自经验积累”或“完整 RSI 变差是由某一个具体机制造成”的因果结论。四组严格匹配的完整新协议实验没有完成；因此本报告明确保留证据边界。

## 实际修改

- `tools/compare_evolution_runs.py`
  - 正确读取 `models/active.json` 的 `id`，同时保留旧字段的只读兼容。
  - 单独输出 Active、最后候选、历史最佳候选、拒绝列表和 `final_source`。
  - Active 没有实测指标时，只能从明确以 Active 为 parent 的同轮 baseline 恢复；否则最终指标为缺失，不再拿最后一个候选填补。
  - 统计器是只读的，不会修改历史决策。
- `harness4h3/archive/model_store.py`
  - 保持 `active.json` 的规范写法为 `{"id": ...}`，并以原子方式更新指针。
- `harness4h3/evolution/evaluator.py`
  - 延迟、峰值显存、完整 Pipeline 体积使用可分别配置的门槛；默认分别为 5%、5%、10%，三者任一达标即可通过效率目标层。
  - 完整性、真实独立评测、总体质量和 semantic/temporal/motion 质量下限、非目标硬件退化、缺失指标、Pareto 支配均单独记录原因。
  - 缺失指标不算改善；零或非正峰值显存视为不可用，不会被解释为显存降到零。
  - 已晋升模型作为下一轮 Parent，而不是继续拿最初 baseline 比较。
  - 记录 Transformer 参数量/权重大小和完整 Pipeline 逻辑体积；硬链接、软链接和共享 inode 不重复计数。
  - 峰值显存绑定实际评测设备。CUDA 监控不可用时返回不可用状态，不填零。
- `harness4h3/evolution/agent.py`、`harness4h3/sandbox/runner.py`
  - 修正了控制器计划格式、Wan 全 block FFN 计划、沙箱安全迭代和真实控制器的上下文长度问题。
  - 记忆经验发送给控制器前只保留可检索摘要；完整评测 artifact 仍留在磁盘，避免把 case/video 路径和整份历史测量塞进 16k context。
  - 记录算法设计、证据字段、历史 ID、搜索选择和失败反馈；算法新颖性本身不能绕过质量和测量门槛。
- `docs/evolutionary-compression.md`、`docs/experiment-schema.md`
  - 写入统一 Evaluation Recipe 和字段定义，明确区分压缩目标 Transformer 与完整 Wan Pipeline。

## 指标口径

- `transformer_parameter_count`、`transformer_model_size_gb`：只统计压缩目标 Transformer。
- `pipeline_model_size_gb` / `model_size_gb`：统计完整 Pipeline 的逻辑唯一文件体积，去除硬/软链接和共享权重重复计数；固定的文本编码器、VAE 等组件不会因链接布局变化被误算成压缩收益。
- `parameter_count`：在当前 Wan 评测中按 Transformer 参数量记录，并带有 scope 字段，不能当成整个 Pipeline 参数量。
- 延迟：denoising 加 VAE decode，按 case 缓存 prompt embedding；1 次 warmup、3 次正式测量，覆盖 16 个 case/seed 单元。
- 峰值显存：PyTorch CUDA peak memory，绑定 `cuda:1`（或每次运行明确记录的实际设备）；`peak_memory_available` 和 `peak_memory_device` 必须同时可查。
- 质量：CLIP temporal 的 semantic、temporal、motion，加视频结构 gate。当前质量下限是 Parent 的 98%；有可靠子指标时同样检查 98% 下限。

远端本次环境的 `nvidia-smi` 因 NVML driver/library mismatch 不可用，所以没有把功耗、利用率或 NVML 显存当作零。PyTorch CUDA 仍能运行，峰值显存可测；这项限制已作为实验限制保留。

## 已有真实实验证据

### 同版本闭环核验

在同一 v6 协议中，Fixed 三轮均完成真实 Wan 评测并拒绝，Active 最终仍是 `M0000`。Coding Agent 三轮也均完成评测并因 semantic 质量低于 Parent 的 98% 下限而拒绝。Agent + Memory 的第一轮同样完成评测并因 semantic 质量失败而拒绝；第二、三轮不是候选结果，而是控制器上下文超过 vLLM 16k 上限后生成失败，因此不能拿来证明 Memory 算法失败。

同一批 v6 记录还暴露了几个已修复的执行层问题：控制器返回顶层 `steps`、使用旧的 `prune_blocks` 别名、FFN 计划只覆盖部分 Wan block，以及把完整 Memory artifact 发送给控制器导致 context overflow。它们说明旧日志中有“实验未完成”和“候选被拒绝”两类状态，统计时不能混为一谈。

### Memory 与 RSI 候选的已有独立复测

远端 `formal-v14-independent-candidates-final-v8-rsi-memory-placement` 对两个明确选定的候选使用了同一个 baseline checkpoint hash、同一个 16 case/seed 评测清单和 3 次 latency 测量。它不是完整四臂匹配协议：Memory 在 `cuda:2`，RSI 在 `cuda:3`，也没有 Fixed/Coding Agent 的同批独立复测。因此下面只作为候选级证据，不当作严格四组因果实验。

| 候选 | 延迟改善 | 峰值显存改善 | Pipeline 体积改善 | 总质量变化 | Transformer 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Agent + Memory | 21.46% | 7.06% | 5.14% | -1.95% | 747,853,600 |
| 完整 RSI | 3.30% | 0.49% | 0.36% | -3.71% | 1,372,556,096 |

这组记录能确定两件事：Memory 候选的硬件收益确实更大；两者都没有在新协议下被历史决策自动追认。Memory 候选在旧日志中是全 block 的 FFN 宽度压缩，保留约 18.75% channel；RSI 候选则是一次单 block 删除，压缩幅度和目标明显不同。这是“候选设计不同”的直接证据，也是不能把差异直接归因于 Memory 的原因。

### 中断的 v7 新协议运行

源码 `05e2785` 下生成了新的四臂协议，指纹为：

`cc2887076db4840a37c5662ee02123f7d88a5d364dfd44c30fc68585d24999bd`

它统一了 checkpoint、Wan 配置、评测清单、16 case/seed、3 次 latency 测量和 3 轮预算。为遵守 GPU 预算，实际只让 Fixed 的 R0001 走完真实评测后停止后续全量运行；这一轮删除 `block.17`，延迟改善 2.25%、显存改善 0.41%、Pipeline 体积改善 0.36%，完整性和独立评测通过，但按新协议因三个目标都未达门槛而拒绝。该结果只作为低风险端到端核验，不与旧版本混合成四组统计。

## 对三个验收问题的回答

### 1. 当前系统是否已经具备可信的自动压缩与模型晋升闭环？

代码层面：基本具备。流程已经是 profile → 生成算法 → 沙箱 → 压缩 → 独立评测 → 质量/硬件/Pareto 决策 → 经验记录；Active 指针和拒绝候选的统计语义也已修正，Fixed 真实 Wan 运行证明这条链能正确保持 Active 不变。

实验结论层面：还不能说“已经被充分验证”。缺少同一源码、同一设备条件和同一 protocol fingerprint 下四臂完整结果；同时 v6 Agent + Memory 的后两轮是基础设施失败，不是算法拒绝。因此现在可以说闭环实现可信、统计口径可信，但还不能说 RSI 的持续改进结论已经可信。

### 2. 为什么 Agent + Memory 的效果优于完整 RSI？哪些是证据，哪些是猜测？

已知事实是：独立候选复测中 Memory 的候选明显更激进，采用全 block FFN 宽度压缩，参数量从约 14.19 亿降到约 7.48 亿；RSI 候选只从约 14.19 亿降到约 13.73 亿。前者自然有更大的潜在硬件收益，但也带来质量风险。v6 的 Coding Agent 和 Memory R0001 也出现了“大硬件收益伴随 semantic 质量失败”的记录，说明质量门槛确实在发挥作用。

因此，最有证据的解释是“候选压缩位置和幅度不同”，而不是“Memory 一定更会设计算法”。经验检索、explore/exploit 选择、质量失败后的回退是否导致了这个选择，当前仍是猜测：旧实验的 GPU 不同，历史版本和计划格式也有漂移，且没有严格控制候选操作和压缩预算。不能把一次 Memory 候选成功或 RSI 候选较保守，升级为 RSI 优势已被证伪的结论。

### 3. 距离证明系统通过经验积累持续改进压缩算法，还缺少什么最小实验？

最小实验应包含三条严格匹配的轨迹：

1. Memory-off；
2. 真实 Memory；
3. 数量相同但打乱或错误关联的伪 Memory 控制。

三条轨迹从同一 Parent 开始，固定源码、controller、GPU、Wan checkpoint、评测清单、随机种子、16 case/seed、每 case 3 次延迟测量和 3 轮预算。每轮必须保存 profile、候选源码和参数、检索到的 experience ID、搜索策略状态、独立质量/硬件结果及晋升决策；最后在未参与搜索的 held-out 清单上复测。

只有当真实 Memory 相对两个控制在至少两个连续 round 中产生可重复的更好 Pareto 进展，且质量始终不低于 98%、设备和 recipe 没有漂移，才能把结果表述为“经验积累改善了压缩算法”。单个大幅压缩候选、一次四组均值，或把同一个模型的多个 case 当成多个独立算法实验，都不够。

## 验证与交付

- CPU 全量测试：`931 passed, 4 skipped in 101.35s`。
- 4 个 skip 均为明确的 CUDA/真实 Wan 条件跳过，不是本轮失败。
- `python -m compileall -q harness4h3 h3_training research tools` 通过。
- `git diff --check` 通过。
- 本轮真实 GPU 预检和 Fixed R0001 证明真实控制器、Wan checkpoint reload、独立质量评测、峰值显存记录和拒绝决策链可以运行；没有继续消耗资源完成四组全量重跑。

本报告只更新审查记录；工作区中其他 Student、H3 训练和蒸馏改动未纳入本次提交。
