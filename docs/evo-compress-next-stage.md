# EvoCompress 下一阶段 CPU 验证接口

本分支新增能力默认关闭，因此不改变既有 Wan2.1-1.3B 四组实验的执行逻辑。下一阶段的 FakeWanDiT 试运行可以在独立输出目录中打开：

```yaml
backend: fake
checkpoint: fake://baseline
output_root: var/evolutionary-compression-evo-search
advanced_profiling: true
algorithm_improvement: true
search_space_evolution: true
candidate_screening: true
```

再运行现有的 `evolve-compression` 命令即可。真实 Wan 配置保持四个开关为 `false`，直到独立实验窗口允许验证。

每轮输出包括：

- `rounds/Rxxxx/profile.json`：校准样本、Block/Channel/Timestep 关系及真实测量/估算/不可用证据。
- `rounds/Rxxxx/algorithm.json`：算法源码、父算法 ID、父源码哈希、真实 Source Diff、修改理由和验证状态；`algorithms/candidates/Axxxx.diff` 保存同一差异的独立文本副本。
- `rounds/Rxxxx/algorithm-validation.json`：相同父模型、history、Sandbox 预算和多个固定诊断 Profile 下的新旧算法比较；CPU Fake 证据不能用于真实 Wan 晋升。
- `rounds/Rxxxx/algorithm-validation-profiles.json`：算法级比较使用的固定 Profile 集合，随中断恢复，不重新随机生成条件。
- `search-space-feedback.json`：由已完成轮次的失败、重复计划和 Pareto 停滞推导出的触发证据；初始化不会预注册扩展。
- `search-space/proposals/`、`search-space/versions/` 和 `search-space/active.json`：搜索空间提案、拒绝原因、不可变版本及当前 Active 指针。只有触发证据达到阈值后才会尝试提案；`prune_ffn_by_weight_threshold` 和 Agent 生成规则默认归类为 `algorithm_rule_evolution`，不能仅凭新名称声称基础搜索空间扩展。
- `rounds/Rxxxx/algorithm-validation-measurements/`：父子算法在相同参考 Checkpoint 和固定 Probe 下的真实 Fake 模型级压缩、重载、输出和成本测量。
- `rounds/Rxxxx/tica-search.json`：TICA-Prune v2 的规范化 SearchState、候选计划摘要、选中候选、Pareto 前沿、预算账本和缓存命中记录。交互分析的独立模型执行同时计入 `full_evals`。
- TICA 指标口径固定为：`parameter_count`/Transformer 权重大小只描述压缩目标 Transformer；`pipeline_model_size` 描述包含文本编码器、VAE 和配置的完整 Pipeline 逻辑大小，并按 inode 去重硬链接/软链接；CUDA 峰值显存只接受实际执行 Device 上的正数监控值，不可用时写 `null`。
- TICA 的校准输入只能用于敏感度和候选搜索；算法比较使用独立 Validation manifest，最终 Wan 视频评价使用 held-out Test manifest，Test split 不会进入搜索或反馈。`TICAEvaluationSplits` 会拒绝复用 sample ID 的划分；真实 Wan TICA 运行必须显式提供三份 manifest。
- 真实 Wan TICA 的 Evolution 配置必须将 `evaluator.evaluation_split` 设为 `validation`；Test 只能在算法、模型、源码版本和评测协议冻结后，通过独立的 `final-evaluate` 命令运行，不能把 held-out 指标回流到晋升或记忆。
- TICA 搜索结果是最终执行计划的唯一来源。`decision.json` 同时记录 `selected_candidate_id`、`selected_plan_digest` 和 `final_plan_digest`，摘要不一致时不会进入压缩和晋升流程。
- 评测报告区分 Transformer 参数/权重、完整 Pipeline 唯一文件体积、固定文本编码器/VAE 体积、实际 CUDA 设备峰值显存和去噪+解码延迟；缺失硬件值保持缺失。搜索结果还记录每层实际深度、Fast/Full/Interaction 预算、Calibration profiling 和缓存命中，不能把同一模型的多 case 测量当成多次算法实验。
- `rounds/Rxxxx/screening/screening.json`：静态、诊断、完整评测三阶段及每阶段成本/证据等级。
- `design-genes.jsonl` 与 `algorithm-skills.jsonl`：可供下一轮 SkillHub 检索的算法、源码哈希、改动理由、验证状态和搜索空间经验。

筛选器不会替代 `PromotionPolicy`：静态/诊断结果只能过滤候选，完整质量评测仍由原有独立评测器提供，Fake 或 proxy 证据不会被当作真实 Wan 质量证明。

算法父本选择会分别记录 latest、effective、historical-best、pending 和 rejected
分支；最新生成但验证失败的算法不会自动覆盖此前有效的算法。算法源码变化、算法级
性能改善和模型 Promotion 是三个独立事实。

当前验收命令：

```bash
/Users/huangjiahao/MinMax-H3/Harness4H3/.venv/bin/python -m pytest -q \
  tests/test_config.py \
  tests/unit/test_compression_config.py \
  tests/unit/test_profiler_calibration.py \
  tests/unit/test_profiler_sensitivity.py \
  tests/unit/test_wan_profiler.py \
  tests/unit/test_algorithm_improvement.py \
  tests/unit/test_algorithm_evolution.py \
  tests/unit/test_algorithm_sandbox.py \
  tests/unit/test_search_space.py \
  tests/unit/test_candidate_screening.py \
  tests/integration/test_evolutionary_compression.py \
  tests/unit/test_evolution_promotion.py
```
