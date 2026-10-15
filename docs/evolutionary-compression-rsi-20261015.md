# EvoCompress RSI 实验记录（2026-10-15）

本记录对应 `codex/algorithm-discovery-rsi` 的最终工作树版本。实验按实现方案执行了四个 CPU Fake 对照臂；所有结果都保留 `offline_simulation=true`，用于验证算法发现、记忆反馈、Sandbox、SearchSpace 进化和可重启证据链，不代表 Wan2.1-1.3B 的真实 GPU 延迟、显存或视频质量。

## 最终代码验证

```text
focused algorithm/runtime tests: 30 passed
full regression: 1020 passed, 4 skipped, 149.27s
```

4 个 skip 是真实 Wan checkpoint 未设置、CUDA 不可用和 CUDA RNG parity 条件；没有测试失败。

## 四组实验

| 实验臂 | 轮数 | 状态 | 算法源码 SHA 数 | 结果摘要 |
| --- | ---: | --- | ---: | --- |
| Fixed Algorithm | 3 | promoted / promoted / promoted | 3 | `prune_blocks`；估计计算比 `0.80 → 0.75 → 0.6667` |
| Coding-agent proxy | 3 | promoted / promoted / promoted | 3 | 确定性 controller 代理；同一 CPU Fake 协议，不宣称外部 LLM 已参与 |
| Agent + Memory | 3 | promoted / rejected / rejected | 3 | 记忆检索和算法改进链路工作；第 2 轮后收益低于晋级阈值 |
| Full RSI | 10 | 1 promoted / 9 rejected | 10 | 第 3 轮从 `S0001` 进入 `S0002`；反馈触发四类瓶颈信号 |

Full RSI 的 SearchSpace 反馈为：`pareto_stagnation`、`repeated_algorithm_plan`、`repeated_failure` 和 `latency_diminishing_efficiency_gain`。这证明搜索空间扩展不是无条件注册，而是先从已落盘轮次收集证据，再经过 proposal、Sandbox、adapter 执行、保存/重载和 finite 检查。

四组配置：

- [`fixed-final2 config`](../configs/evolutionary-compression.fake-protocol-fixed-final2-20261015.yaml)
- [`coding-final2 config`](../configs/evolutionary-compression.fake-protocol-coding-final2-20261015.yaml)
- [`memory-final2 config`](../configs/evolutionary-compression.fake-protocol-memory-final2-20261015.yaml)
- [`full-rsi-final2 config`](../configs/evolutionary-compression.fake-rsi-final2-20261015.yaml)

## 可复核证据

- Fixed：[`R0003 decision`](../var/evolutionary-compression-fake-fixed-final2-20261015/rounds/R0003/decision.json)
- Coding-agent proxy：[`R0003 decision`](../var/evolutionary-compression-fake-coding-final2-20261015/rounds/R0003/decision.json)
- Agent + Memory：[`R0003 algorithm validation`](../var/evolutionary-compression-fake-memory-final2-20261015/rounds/R0003/algorithm-validation.json)
- Full RSI：[`R0010 decision`](../var/evolutionary-compression-fake-rsi-final2-20261015/rounds/R0010/decision.json)
- Full RSI 搜索空间反馈：[`search-space-feedback.json`](../var/evolutionary-compression-fake-rsi-final2-20261015/search-space-feedback.json)
- Full RSI 搜索空间版本：[`R0003 search-space.json`](../var/evolutionary-compression-fake-rsi-final2-20261015/rounds/R0003/search-space.json)

## 运行命令

四组命令均使用同一入口，仅替换配置和轮数。例如完整 RSI：

```bash
./.venv/bin/python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.fake-rsi-final2-20261015.yaml \
  --rounds 10 --json
```

## 真实性边界

当前主机没有可用的 Wan2.1-1.3B checkpoint/CUDA 实验条件，因此没有伪造 GPU 结果。真实验收仍需使用 Wan 配置中的 checkpoint、Calibration/Validation/Test manifest，在固定 GPU、Prompt、Seed、Scheduler、warm-up/repeat 下重新记录 Dense 对照、DiT/Kernel/视频三级耗时、显存和视频质量。
