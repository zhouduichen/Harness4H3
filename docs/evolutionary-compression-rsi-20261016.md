# EvoCompress RSI 实验记录（2026-10-16）

本记录对应 `codex/algorithm-discovery-rsi` 的当前实现。六组实验均使用 CPU Fake Wan 协议，`offline_simulation=true`；它们验证算法发现、记忆反馈、冻结/演化 SearchSpace、KV Token Selection 和可重启证据链，不代表 Wan2.1-1.3B 的真实 GPU 延迟、显存或视频质量。

## 代码验证

```text
focused algorithm/runtime tests: 36 passed
full regression: 1035 passed, 4 skipped
```

完整回归的 4 个 skip 是真实 Wan checkpoint、CUDA、CUDA RNG parity 和训练运行条件未满足，不是实现失败。新增的 KV 路径覆盖 Fake 执行与重载、按 Block 目标选择、KV + Sparse 组合，以及 Diffusers Tiny Wan 的真实注意力张量形状/统计检查；本轮还覆盖 Head + Sparse、Layer + Sampling 的有序组合、Provider 对 Active SearchSpace 的读取，以及 Head/FFN 的输出扰动敏感度测量和显式回退。

## 六组实验

| 实验臂 | 轮数 | 状态 | 不同算法源码 SHA 数 | 观察 |
| --- | ---: | --- | ---: | --- |
| Fixed Algorithm | 3 | promoted / promoted / promoted | 3 | 固定协议完成 3 代；每代都保存独立算法源码与验证记录；profile 使用 FFN output perturbation |
| Coding-agent proxy | 3 | promoted / promoted / promoted | 3 | 确定性 controller 代理改变候选生成/排序；不宣称外部 LLM 已参与 |
| Agent + Memory | 3 | promoted / rejected / rejected | 3 | 记忆反馈进入下一代；低于晋级阈值的候选仍保存为负向经验 |
| RSI Fixed Space | 3 | 1 promoted / 2 rejected | 3 | 算法/记忆可改，SearchSpace 固定；轨迹仍读取完整默认算子闭包 |
| Full RSI | 10 | 5 promoted / 5 rejected | 10 | `S0001 → S0002`；active space 从 12 扩展到 13 个算子；包含 `kv_token_select` 与 `timestep_block_gate` |
| Frozen Expanded Space | 3 | 1 promoted / 2 rejected | 3 | 固定 `S0001` 扩展空间；后代实际生成 `kv_token_select` 候选，没有注册新版本 |

Full RSI 的反馈文件记录了 `pareto_stagnation`、`repeated_algorithm_plan`、`repeated_failure` 和 `latency_diminishing_efficiency_gain` 四类信号。SearchSpace 扩展先经过 proposal、Sandbox、adapter 执行、保存/重载和 finite 检查，再注册到 `S0002`，下一轮 Program 能读取该版本。

本轮六组配置（`final6`，均启用 `channel_method: output_perturbation`）：

- [`fixed-final6 config`](../configs/evolutionary-compression.fake-protocol-fixed-final6-20261016.yaml)
- [`coding-final6 config`](../configs/evolutionary-compression.fake-protocol-coding-final6-20261016.yaml)
- [`memory-final6 config`](../configs/evolutionary-compression.fake-protocol-memory-final6-20261016.yaml)
- [`full-rsi-final6 config`](../configs/evolutionary-compression.fake-rsi-final6-20261016.yaml)
- [`rsi-fixed-space-final6 config`](../configs/evolutionary-compression.fake-protocol-rsi-fixed-space-final6-20261016.yaml)
- [`frozen-expanded-final6 config`](../configs/evolutionary-compression.fake-frozen-expanded-final6-20261016.yaml)

## 可复核证据

- Fixed：[`R0003 decision`](../var/evolutionary-compression-fake-fixed-final6-20261016/rounds/R0003/decision.json)
- Coding-agent proxy：[`R0003 decision`](../var/evolutionary-compression-fake-coding-final6-20261016/rounds/R0003/decision.json)
- Agent + Memory：[`R0003 algorithm validation`](../var/evolutionary-compression-fake-memory-final6-20261016/rounds/R0003/algorithm-validation.json)
- Full RSI：[`R0010 decision`](../var/evolutionary-compression-fake-rsi-final6-20261016/rounds/R0010/decision.json)
- Full RSI 搜索空间反馈：[`search-space-feedback.json`](../var/evolutionary-compression-fake-rsi-final6-20261016/search-space-feedback.json)
- Full RSI 搜索空间版本：[`R0003 search-space.json`](../var/evolutionary-compression-fake-rsi-final6-20261016/rounds/R0003/search-space.json)
- Full RSI 提案：[`P0001.json`](../var/evolutionary-compression-fake-rsi-final6-20261016/search-space/proposals/P0001.json)
- Head/FFN 输出扰动 profile：[`R0001 profile.json`](../var/evolutionary-compression-fake-rsi-final6-20261016/rounds/R0001/profile.json)
- RSI Fixed Space：[`R0003 decision`](../var/evolutionary-compression-fake-rsi-fixed-space-final6-20261016/rounds/R0003/decision.json)
- Frozen Expanded Space：[`R0003 search trace`](../var/evolutionary-compression-fake-frozen-expanded-final6-20261016/rounds/R0003/algorithm-program-trace.json)

## 真实性边界与 GPU 阻塞

本机 `torch.cuda.is_available()` 为 `False`，且没有 `nvidia-smi`。本地 HuggingFace Wan snapshot 的配置/权重索引可读，但 `transformer/` 目录缺少两个索引声明的权重分片：

```text
diffusion_pytorch_model-00001-of-00002.safetensors
diffusion_pytorch_model-00002-of-00002.safetensors
```

因此本轮没有伪造真实 Wan2.1-1.3B GPU 结果。真实验收仍需在具备完整 checkpoint 和 CUDA 的机器上，使用固定 GPU、Prompt、Seed、Scheduler、warm-up/repeat 与 Calibration/Validation/Test manifest，记录 Dense 对照、DiT/Kernel/视频三级耗时、显存和视频质量。
