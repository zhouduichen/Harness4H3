# Harness4H3 多粒度 RSI 验收记录

本记录对应 `codex/algorithm-discovery-rsi`，并按
`EvoCompress_Codex_Implementation_Plan.md` 区分 CPU/结构证据与真实 GPU 证据。

## 已完成并验证

- Layer/Block、全局对齐 FFN、Attention/Head 结构剪枝：Fake 保存、重载、参数量变化和前向形状检查通过；Diffusers Tiny Wan 也覆盖了 fused QKV 先 unfuse 再 packed head 的路径。新增 runtime Head Gating 按层/时间步选择 Head，直接切 Q/K/V 行与输出投影列，不计算后再乘零。
- Token、KV Token Selection、Sparse Runtime、Attention Window、Timestep、Sampling Step：策略持久化，重载后恢复，并在 Fake/Diffusers Wan Block 路径实际改变执行输入、Q/K/V 长度、局部 K/V 范围、每个 batch 样本的 Block 执行或 scheduler 步数。KV 选择保留完整 Q，仅按 score 选择 K/V，并支持按 Block 定位；新增 `timestep_attention_policy` 按每个样本的 timestep 选择不同 K/V 保留比例，混合 batch 不取第一个样本的 timestep。
- 有序 Hybrid：`Head + Sparse`、`Token + Sparse`、`Head + Sparse + Window`、`Timestep + Head + Sparse`、`Layer + Sampling Step` 已有组合执行测试；`compose` 保留步骤顺序并支持最多 6 步。
- `ModelVariant`：通过 sidecar manifest 区分结构 checkpoint、runtime policy、sampler policy、算法版本和 SearchSpace 版本；runtime-only 变体不填写 structural checkpoint，参数减少固定为 0。
- Algorithm Discovery：Sandbox 中执行 Agent 的 `search()`，轨迹记录候选集合、排序和选中计划；搜索源码纳入逻辑摘要，递归子代保留父子源码差异。
- Search-Space Evolution：提案经过 Sandbox、真实 adapter 执行、保存/重载、finite 检查后注册；下一代 Program 读取新版本并能生成新算子计划。执行器通过受限的 `implementation_ref` loader 恢复，不要求为每个新类修改管理器分支。
- 新增的 timestep attention 扩展在激活后的 SearchSpace smoke 中进入 `max_candidates=8` 的实测预算，候选结果 `valid=true`；候选未被 Pareto 选择不等于未执行，搜索轨迹保留了该评测结果。
- 标准 Algorithm Search 的每个候选现在持久化真实执行证据：`runtime_execution.stats/derived`、结构/FLOPs/参数变化和硬件测量边界；CPU Fake/Tiny 的运行时计数不会被写成 GPU latency 或 VRAM。
- 对照实验：新增 `RSI Fixed Space` 与 `Frozen Expanded Space`；后者固定版本 `S0001`，但下一代实际读取并生成扩展算子候选，用于区分“算子更多”和“空间动态演化”的收益。
- Provider Agent：初始算法生成请求现在携带当前 SearchSpace 的 operator/schema/约束摘要，提示词不再把候选限制为 Block/FFN；因此真实 Provider 路径也能提出 Head、KV、Sparse、Window、Timestep 和 Sampling 计划。
- Sensitivity：Head profiling 在有 Attention-shaped 序列样本时逐个复制 Attention、执行真实 Head surgery 并测量输出 leave-one-out 差异；真实 Wan 的 5-D latent 会先通过一次模型 forward 捕获每个 Attention 的真实参数，再只复制 Attention 做 probe，避免复制 1.3B 模型。FFN channel 支持 `channel_method: output_perturbation`，通过实际 `up_proj` hook 和完整模型输出差异评分。没有可用样本或 hook 时仍明确返回 unavailable/weight proxy，不填零分。
- 远程真实 Wan 基础设施 smoke：`f082f19` 在 CUDA1 上完成真实 Diffusers Wan 加载、单轮生成、结构 checkpoint 保存/重载和独立视频评测；另以同一远程 1.3B Transformer 执行 `Token + Head + Sparse K/V + Window` 组合 probe，四类统计均显示实际 selected/executed 工作量下降，输出形状保持不变。

## 部分完成 / 有条件完成

- Diffusers Wan 的 Head/Token/Sparse/Attention Window 代码已按当前安装版本（Diffusers 0.39.0）的真实模块签名实现，并有 Tiny Wan 结构 smoke；Algorithm evaluator 现在还提供真实适配器的 sequence-shaped mixed-timestep runtime probe；完整 Wan2.1-1.3B 视频、GPU Kernel、显存和端到端延迟尚未在本机执行。
- Sparse Runtime 当前是可执行的 Top-k KV 选择后端；它保留 Wan Q/K 归一化、RoPE 位置和输出投影，并确实缩短 K/V 输入，不宣称 SVOO/SparseVideo CUDA Kernel。独立的 KV Token Selection 也通过同一真实 K/V gather 路径执行，可与 Head、Token Merge、Attention Window 和 Timestep Attention 联合；Runtime-only 不记录参数/存储减少。
- Attention Window 当前同时支持历史 flat contiguous window 与可选 `(grid_shape, window_radius)` 的 3-D local Self-Attention；3-D 路径保持完整 Q，按原始 `(T,H,W)` 位置选择邻域 K/V 并切 RoPE，网格不匹配会显式失败。它仍未宣称 Flex/SVOO 三维 BlockMask 或专用 CUDA Kernel。
- Timestep Gate 对 batch 内不同 timestep 使用 active-subset 执行；`timestep_attention_policy` 对匹配的每个 batch 子集实际缩短 Self-Attention K/V，并记录 rule、完整/选中 token 数与重载后的统计；它可与 Head、Sparse 和 Window 组合，混合 batch 不取第一个样本的 timestep。
- Token 当前包含 Self-Attention 输入选择/恢复和 `token_merge` 路径；Token Selection 提供 `(T,H,W)` 网格下的逐帧 Anchor 保留，避免纯一维 top-k 丢失整帧；`token_merge` 同时支持兼容旧配置的 1-D adjacent 以及 `(T,H,W)` cuboid TokenMap，保存 anchor/group shape，在合并后的序列上执行 Attention，再 unmerge 回残差长度。三维专用 CUDA Kernel 仍未宣称完成。
- FLOPs 为 adapter 的结构估计；真实性能结论必须使用 WanVideoEvaluator 的固定 GPU、Prompt、Seed、Scheduler 和 warm-up/repeat 结果。
- 候选级 `runtime_execution` 是 post-forward trace，用于证明 Token/KV/Window/Timestep 等策略确实改变执行；GPU kernel、端到端延迟和显存仍由独立真实 Wan Evaluator 提供。
- 远程 10 轮 Full RSI（见下节）返回 `failed_rounds=0`；R0001–R0010 均完成真实 Wan 视频评测，10 个算法 source hash 均不同。该批次在修复前运行，因此 SearchSpace 提案 P0001–P0008 均停留在 S0001，P0008 的拒绝原因是 Diffusers pipeline 的 finite-check 兼容性 bug，而不是新算子执行失败。

## 最新基础设施回归

在 `f082f19` 基础上，本轮多粒度执行路径和组合修复后的本地全量回归为 `1055 passed, 4 skipped`。4 个 skip 仍是本机未提供真实 Wan checkpoint、CUDA 或 CUDA RNG parity 条件；真实 Wan smoke 只在远程执行。

远程单轮 Wan smoke 和多粒度 runtime probe 均已在独立 output root 完成：`failed_rounds=0`、`offline_simulation=false`、真实视频与 visual-structure gate 均有效；Block 候选因质量/效率门槛未达标而 rejected，但 Token/Head/Sparse/Window 组合 probe 的真实执行统计均有效。远程证据详见 [`evolutionary-compression-wan-remote-20261016.md`](evolutionary-compression-wan-remote-20261016.md)。

之前的 `1043 passed, 4 skipped` 和 CPU Fake RSI 结果保留为历史记录，不覆盖本次回归结果。

## 远程 10 轮 Full RSI 与 SearchSpace 修复证据

本次实验严格在 `ai4mbse-remote` 的 GPU1 执行，本机没有 Wan 权重。使用配置
[`evolutionary-compression.wan-remote-rsi-f082f19-20261016.yaml`](../configs/evolutionary-compression.wan-remote-rsi-f082f19-20261016.yaml)
运行：

```bash
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.wan-remote-rsi-f082f19-20261016.yaml \
  --rounds 10 --json
```

远端产物根目录为：
`/data/models/harness4h3-evolutionary-compression-20261016/remote-rsi-f082f19-20261016/`。

- R0001–R0010 全部终态为 `rejected`，但 `failed_rounds=0`；每轮都有 `algorithm.json`、程序轨迹、压缩、决策和 `evaluation.json`。
- 每轮 `evaluation_status=measured_video`、`offline_simulation=false`、`generation_device=cuda:1`，视频为 256×256×81、1 case、1 seed，并通过 visual-structure gate。
- 10 个算法 source hash 全部唯一，满足多代 Algorithm Evolution；每轮的模型子代 M0001–M0010 均实际生成并评测，但没有达到 promotion 的质量/延迟/显存/模型大小阈值，因此没有被晋级。
- 真实评测中实际被选为最终计划的算子包括 `prune_blocks` 和 `head_gating`；多粒度候选搜索仍在每轮的 algorithm validation/search trace 中执行，不能把最终未晋级误认为路径未执行。

该批次发现了 SearchSpace 验证 bug：真实 Diffusers `DiffusionPipeline` 的 `parameters` 在当前版本是 mapping，旧代码调用 `model.parameters()`，导致 P0008 被记录为 `"'dict' object is not callable"`。修复后 `_assert_finite` 在 pipeline 上回退到 `pipeline.transformer.parameters()`，并保留异常类型/traceback 以便后续审计。

修复后的远程定点验证目录为：
`/data/models/harness4h3-evolutionary-compression-20261016/search-space-fix-smoke-S0002/`。

- P0008 在真实 Wan 父检查点上完成保存、重载和 finite 检查，`status=validated`、`child_reload_verified=true`、`dynamic_compute=true`、`parameter_reduction_ratio=0.0`。
- P0008 注册为 `S0002`，算子数从 13 增加到 14，保留 runtime-only 的参数收益为 0。
- `next-generation/algorithm-program-trace.json` 用 A0010 的 search 源码和 R0010 profile 读取 S0002，明确记录 `search_space_version=S0002`、14 个 operator 名称，并生成了 `timestep_block_gate` 候选；这闭合了“提案→验证→注册→下一代使用”的链路。
- 该 S0002 smoke 是真实 Wan 结构保存/重载验证，不冒充完整视频质量或 GPU kernel 加速证据；完整视频证据来自上述 10 轮真实 evaluator。

## 历史测试证据

```text
focused algorithm/runtime tests: 38 passed
focused profiler/config/runtime selection: 54 passed
full regression: 1043 passed, 4 skipped
```

4 个 skip 分别是未设置真实 Wan checkpoint、CUDA 不可用或 CUDA RNG parity 条件，不是实现失败。

2026-10-16 的六组 CPU Fake RSI 结果和证据路径见
[`docs/evolutionary-compression-rsi-20261016.md`](evolutionary-compression-rsi-20261016.md)；2026-10-15 报告保留为历史记录。

## 真实 Wan 最小命令

先替换 `configs/evolutionary-compression.wan-algorithm-discovery.example.yaml` 中的 checkpoint、三份 Calibration/Validation/Test manifest、Wan 配置、CLIP 和 controller：

```bash
uv run --with torch --with diffusers python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.wan-algorithm-discovery.example.yaml \
  --rounds 10 --json
```

真实 GPU 验收必须额外记录 Dense 对照、稀疏 Kernel/DiT/视频三级耗时、显存、视频质量和重启后的 runtime policy 恢复；CPU Fake 结果不能替代这些证据。
