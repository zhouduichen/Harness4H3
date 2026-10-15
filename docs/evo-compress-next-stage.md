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
