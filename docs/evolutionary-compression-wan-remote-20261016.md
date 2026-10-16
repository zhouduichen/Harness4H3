# Wan2.1-1.3B 远程真实 RSI 记录（2026-10-16）

本实验严格在远程 `ai4mbse-remote` 执行，本机没有下载 Wan 权重，也没有运行真实 Wan pipeline。

## 最新：10 轮 Full RSI（2026-10-16/17）

本节是当前验收批次；后文的首轮 smoke 和基础设施 smoke 是历史记录。

- Code staging：`/home/intern/huangjiahao/Harness4H3-rsi/work/codex-algorithm-discovery-rsi-f082f19/`
- Wan checkpoint：`/home/intern/hf_cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b`
- Output root：`/data/models/harness4h3-evolutionary-compression-20261016/remote-rsi-f082f19-20261016`
- Device：`cuda:1`；GPU0 的 vLLM controller 未触碰
- 命令：

```bash
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
/home/intern/miniconda3/envs/comfy/bin/python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.wan-remote-rsi-f082f19-20261016.yaml \
  --rounds 10 --json
```

结果为 `failed_rounds=0`。R0001–R0010 全部完成 `algorithm → sandbox → validation → compression → evaluation → decision`，每轮 `evaluation_status=measured_video`、`offline_simulation=false`、`generation_device=cuda:1`；10 个算法 source hash 全部不同，A0001–A0010 均生成 M0001–M0010 子代。所有候选均被 promotion policy 拒绝，原因是质量/效率阈值未同时满足，而不是运行失败。

最终被评测的计划在本批次交替出现 `prune_blocks` 和 `head_gating`。完整的候选搜索、递归 search 轨迹和其他多粒度算子保存在各轮 `algorithm-program-trace.json`、`algorithm-validation.json` 和 `decision.json` 中。真实视频示例可从 R0010 的 `evaluation.json` 路径字段取得；视频为 256×256×81、1 case、1 seed，并通过 visual-structure gate。

### SearchSpace 修复与下一代传播

原批次的 P0008（`timestep_block_gate`）曾被记录为 `"'dict' object is not callable"`。完整 traceback 显示错误发生在 `_assert_finite(reloaded)`：Diffusers pipeline 的 `parameters` 是 mapping，不能按 module iterator 调用。修复后改为在 pipeline 上检查 `transformer.parameters()`，并记录异常类型/traceback。

修复后的定点远程验证：

- 目录：`/data/models/harness4h3-evolutionary-compression-20261016/search-space-fix-smoke-S0002/`
- P0008 在真实 Wan checkpoint 上保存/重载成功：`status=validated`、`child_reload_verified=true`、`dynamic_compute=true`、`parameter_reduction_ratio=0.0`。
- 注册结果：`S0002`，从 S0001 的 13 个算子扩展到 14 个；runtime-only gate 不记录参数减少。
- 下一代传播证据：`next-generation/algorithm-program-trace.json` 使用 A0010 source 和 R0010 profile 读取 `S0002`，14 个 operator 名称均进入 sandbox，并生成 `timestep_block_gate` 候选。

S0002 验证是结构保存/重载和有限值检查，不冒充完整视频质量或专用 CUDA kernel 证据；10 轮 Full RSI 的 `measured_video` 结果承担视频级证据。

本批次没有任何本机 Wan 权重或本机真实实验产物。

## 历史：首轮真实结果

## 远程执行边界

- Code staging：`/home/intern/huangjiahao/Harness4H3-rsi/work/codex-algorithm-discovery-rsi-254959e`
- Wan checkpoint：`/home/intern/hf_cache/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b`
- Output root：`/data/models/harness4h3-evolutionary-compression-20261016/remote-rsi-gpu1-20261016`
- Device：`cuda:1`；GPU0 保留给远程 vLLM controller
- 评测输入：256×256、81 frames、16 denoising steps、1 development case、1 seed

远端已验证：Diffusers `0.40.0`、PyTorch `2.10.0+cu128`、4 张 L40；Wan transformer 30 层、FFN 8960。远端 `nvidia-smi` 存在 NVML driver/library mismatch，但 GPU1 的 PyTorch CUDA 分配和完整 Wan pipeline 加载/生成均成功，因此 GPU1 被固定为本实验设备。

## 首轮真实结果

首轮命令：

```bash
cd /home/intern/huangjiahao/Harness4H3-rsi/work/codex-algorithm-discovery-rsi-254959e
HF_HUB_OFFLINE=1 /home/intern/miniconda3/envs/comfy/bin/python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.wan-remote-rsi-20261016.yaml \
  --rounds 1 --json
```

首轮 A0001 从真实 profiling 选择删除 `block.17`，结构诊断通过并 reload 为 29 层；随后 independent Wan video evaluator 生成基线和候选视频，`offline_simulation=false`，视觉质量 gate 通过，视频帧数和像素均有效。

| 指标 | Baseline | Candidate | 结论 |
|---|---:|---:|---|
| CLIP-temporal quality | 0.615681 | 0.618910 | aggregate 略升 |
| semantic | 0.594424 | 0.602816 | 上升 |
| temporal | 0.955881 | 0.926122 | 下降，触发 quality floor |
| generation wall time | 7.611 s | 7.266 s | -4.54% |
| pipeline logical size | 24.311 GB | 24.225 GB | -0.36% |
| transformer parameters | 1,418,996,800 | 1,372,556,096 | -3.27% |
| peak memory | 18.773 GB | 19.279 GB | 变差 |

因此 A0001 按 promotion policy **rejected**：真实执行与质量/硬件数据齐全，但 temporal 质量门槛和效率改善门槛均未满足。首轮证据位于：

`/data/models/harness4h3-evolutionary-compression-20261016/remote-rsi-gpu1-20261016/rounds/R0001/`

## 历史：后续 RSI（旧批次，已停止）

首轮后曾短暂启动剩余 9 轮，但根据“先完成基础设施、只跑基础 smoke”的执行要求，已主动停止该长时进程。R0002 只保留 profiling/搜索中间产物，仍为 `in_progress`，没有最终 decision 或可用于晋升的独立视频证据；不把它计入完成轮次，也不继续复用它启动长跑。旧日志保留在：

`/data/models/harness4h3-evolutionary-compression-20261016/remote-rsi-gpu1-20261016/remote-rsi-remaining9.log`

基础设施修复后的远程 staging revision 为 `c2788a5`（包含 `e826fd3` 的 runtime manifest replay 修复以及 Variant/Registry 恢复幂等性修复）。后续只允许使用 `configs/evolutionary-compression.wan-remote-infra-smoke-20261016.yaml` 的新独立 output root 执行单轮远程 smoke，确认 Variant/Registry/恢复链路后再由用户明确授权长跑。

## 历史：基础设施 smoke 结果

使用上述配置通过非交互 SSH 在远程 GPU1 执行一轮，成功产物位于：

`/data/models/harness4h3-evolutionary-compression-20261016/remote-infra-smoke-c2788a5-nopty/`

- `failed_rounds=0`，完整 round 状态已落盘；`offline_simulation=false`，生成和 pipeline 均为 `cuda:1`。
- `compression.json` 成功保存 30→29 层的真实结构 checkpoint；`variant_manifest.json` 保存有序 `prune_blocks` 操作、结构 checkpoint 路径和 variant digest。
- baseline/candidate 各生成 1 个 256×256×81 视频，视频有效性和 visual-structure gate 均通过；candidate latency 为 7.316 s，logical model size 为 24.225 GB。
- 本轮 candidate 按 promotion policy 被拒绝：temporal 质量下降，约 3.26% latency 改善和约 0.36% logical size 改善均未达到阈值，且 peak memory 上升。该 rejection 是有效评测结论，不是基础设施失败。

首次使用带伪终端的前台 SSH 运行曾产生 `[Errno 32] Broken pipe`，产物保留在独立目录 `remote-infra-smoke-c2788a5`，不计入有效评测；改用非交互 SSH 后链路成功。

本次远程真实配置暂未开启 TICA：远程主机未发现可用的真实 Wan calibration tensor manifest，因此不能伪造 TICA 证据。TICA 仍需单独准备远程 calibration/validation/test 缓存后再执行。

## 历史：多粒度 runtime 基础设施 smoke（`f082f19`）

按“先完成基础设施、只跑短 smoke”的要求，未启动长时间 RSI。代码从
`codex/algorithm-discovery-rsi` 的 `f082f19` 同步到新的远程 staging：

`/home/intern/huangjiahao/Harness4H3-rsi/work/codex-algorithm-discovery-rsi-f082f19/`

使用独立配置
`configs/evolutionary-compression.wan-remote-multigranular-smoke-20261016.yaml`
执行一轮真实 Wan smoke，输出位于：

`/data/models/harness4h3-evolutionary-compression-20261016/remote-multigranular-smoke-f082f19/`

本轮结果：

- `failed_rounds=0`，`source_revision=f082f19`，`offline_simulation=false`，真实生成设备为 `cuda:1`。
- baseline/candidate 均完成 256×256×81 视频生成；candidate 视频有效、visual-structure gate 通过。
- 确定性搜索本轮选择已有的 `prune_blocks(block.17)`，候选按质量/效率 promotion policy rejected；这不是基础设施失败，也没有继续启动 RSI。
- 新增配置、runtime manifest、结构 checkpoint 和完整评测证据均保留在上述 output root 的 `rounds/R0001/`。

### 真实 Wan 多粒度 runtime probe

为确保确定性 Block 候选没有掩盖新增运行时路径，在同一远程 Wan2.1-1.3B Transformer 上执行了一个单次短 probe，依次安装并执行：

`prune_tokens(0.75) → head_gating(0.5) → sparse_runtime(0.5) → attention_window(4)`。

证据文件：

`/data/models/harness4h3-evolutionary-compression-20261016/remote-multigranular-smoke-f082f19/multigranular-runtime-probe.json`

实际结果：

| 运行时统计 | Full | Selected/Executed |
|---|---:|---:|
| Token selection | 480 | 360 |
| Self-Attention heads | 2160 | 1080 |
| Sparse K/V tokens | 720 | 360 |
| Window Attention K/V | 720 | 360 |

输出形状保持 `[2, 16, 1, 8, 8]`，并且 runtime manifest 保存了四个操作及其参数。该 probe 证明执行路径确实缩短了后续张量/Head/KV 工作，不是完成 dense 计算后再置零；它不是长时间质量实验，不能替代后续完整视频级 kernel/latency 对照。
