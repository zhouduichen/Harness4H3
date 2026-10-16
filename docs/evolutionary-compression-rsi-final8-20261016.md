# Full RSI 重跑记录（2026-10-16）

配置：[`evolutionary-compression.fake-rsi-final8-20261016.yaml`](../configs/evolutionary-compression.fake-rsi-final8-20261016.yaml)。执行命令：

```bash
.venv/bin/python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.fake-rsi-final8-20261016.yaml \
  --rounds 10 --json
```

结果：10 轮全部完成，`failed_rounds=0`；其中 5 轮 promoted、5 轮 rejected。算法源码为 `A0001`–`A0010`，模型为 `M0001`–`M0010`，SearchSpace 从 `S0001` 扩展到 `S0003`，并验证了两个提案：`timestep_block_gate`、`timestep_attention_policy`。

运行时证据来自三维 CPU Fake probe（8 个 token、batch 内 `[100, 900]` 混合 timestep），不是结构字段占位：

| 候选 | 实际 trace 结果 |
| --- | --- |
| `prune_tokens` | token reduction `25%` |
| `kv_token_select` | K/V reduction `25%–33.3%` |
| `sparse_runtime` | K/V reduction `25%–33.3%` |
| `timestep_block_gate` | skip ratio `约 9.1%` |
| `timestep_attention_policy` | mixed-timestep K/V reduction `约 41.7%` |
| `sampling_step_skip` | schedule reduction `50%` |

可复核原始产物目录（本地 `var/` 默认被 Git 忽略）：

`var/evolutionary-compression-fake-rsi-final8-20261016/`

该实验是 CPU Fake RSI 证据，不代表 Wan2.1-1.3B GPU latency、显存、专用稀疏 Kernel 或视频质量提升。真实 Wan2.1-1.3B checkpoint 与 CUDA 环境可用后，应使用同一配置协议做 Dense 对照和独立 Validation/Test 评测。
