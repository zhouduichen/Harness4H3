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
- `rounds/Rxxxx/algorithm.json`：算法源码、父算法 ID、父源码哈希、修改理由和验证状态。
- `rounds/Rxxxx/algorithm-validation.json`：同一 profile、history 和 sandbox 预算下的新旧算法比较；CPU Fake 证据不能用于真实 Wan 晋升。
- `search-space/proposals/` 与 `search-space/versions/`：搜索空间提案、拒绝原因和不可变版本。CPU Fake 会演示 `prune_ffn_by_weight_threshold`，并明确标记为 `weight_proxy_only`、`not_real_wan_validated`。
- `rounds/Rxxxx/screening/screening.json`：静态、诊断、完整评测三阶段及每阶段成本/证据等级。
- `design-genes.jsonl`：可供下一轮 SkillHub 检索的算法和搜索空间经验。

筛选器不会替代 `PromotionPolicy`：静态/诊断结果只能过滤候选，完整质量评测仍由原有独立评测器提供，Fake 或 proxy 证据不会被当作真实 Wan 质量证明。

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
