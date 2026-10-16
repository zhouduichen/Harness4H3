# Harness4H3 多粒度 RSI 验收记录

本记录对应 `codex/algorithm-discovery-rsi`，并按
`EvoCompress_Codex_Implementation_Plan.md` 区分 CPU/结构证据与真实 GPU 证据。

## 已完成并验证

- Layer/Block、全局对齐 FFN、Attention/Head 结构剪枝：Fake 保存、重载、参数量变化和前向形状检查通过；Diffusers Tiny Wan 也覆盖了 fused QKV 先 unfuse 再 packed head 的路径。
- Token、KV Token Selection、Sparse Runtime、Attention Window、Timestep、Sampling Step：策略持久化，重载后恢复，并在 Fake/Diffusers Wan Block 路径实际改变执行输入、Q/K/V 长度、局部 K/V 范围、每个 batch 样本的 Block 执行或 scheduler 步数。KV 选择保留完整 Q，仅按 score 选择 K/V，并支持按 Block 定位；新增 `timestep_attention_policy` 按每个样本的 timestep 选择不同 K/V 保留比例，混合 batch 不取第一个样本的 timestep。
- 有序 Hybrid：`Head + Sparse`、`Token + Sparse`、`Layer + Sampling Step` 已有组合执行测试；`compose` 保留步骤顺序并支持最多 6 步。
- `ModelVariant`：通过 sidecar manifest 区分结构 checkpoint、runtime policy、sampler policy、算法版本和 SearchSpace 版本；runtime-only 变体不填写 structural checkpoint，参数减少固定为 0。
- Algorithm Discovery：Sandbox 中执行 Agent 的 `search()`，轨迹记录候选集合、排序和选中计划；搜索源码纳入逻辑摘要，递归子代保留父子源码差异。
- Search-Space Evolution：提案经过 Sandbox、真实 adapter 执行、保存/重载、finite 检查后注册；下一代 Program 读取新版本并能生成新算子计划。执行器通过受限的 `implementation_ref` loader 恢复，不要求为每个新类修改管理器分支。
- 新增的 timestep attention 扩展在激活后的 SearchSpace smoke 中进入 `max_candidates=8` 的实测预算，候选结果 `valid=true`；候选未被 Pareto 选择不等于未执行，搜索轨迹保留了该评测结果。
- 标准 Algorithm Search 的每个候选现在持久化真实执行证据：`runtime_execution.stats/derived`、结构/FLOPs/参数变化和硬件测量边界；CPU Fake/Tiny 的运行时计数不会被写成 GPU latency 或 VRAM。
- 对照实验：新增 `RSI Fixed Space` 与 `Frozen Expanded Space`；后者固定版本 `S0001`，但下一代实际读取并生成扩展算子候选，用于区分“算子更多”和“空间动态演化”的收益。
- Provider Agent：初始算法生成请求现在携带当前 SearchSpace 的 operator/schema/约束摘要，提示词不再把候选限制为 Block/FFN；因此真实 Provider 路径也能提出 Head、KV、Sparse、Window、Timestep 和 Sampling 计划。
- Sensitivity：Head profiling 在有 Attention-shaped 序列样本时逐个复制 Attention、执行真实 Head surgery 并测量输出 leave-one-out 差异；真实 Wan 的 5-D latent 会先通过一次模型 forward 捕获每个 Attention 的真实参数，再只复制 Attention 做 probe，避免复制 1.3B 模型。FFN channel 支持 `channel_method: output_perturbation`，通过实际 `up_proj` hook 和完整模型输出差异评分。没有可用样本或 hook 时仍明确返回 unavailable/weight proxy，不填零分。

## 部分完成 / 有条件完成

- Diffusers Wan 的 Head/Token/Sparse/Attention Window 代码已按当前安装版本（Diffusers 0.39.0）的真实模块签名实现，并有 Tiny Wan 结构 smoke；Algorithm evaluator 现在还提供真实适配器的 sequence-shaped mixed-timestep runtime probe；完整 Wan2.1-1.3B 视频、GPU Kernel、显存和端到端延迟尚未在本机执行。
- Sparse Runtime 当前是可执行的 Top-k KV 选择后端；它保留 Wan Q/K 归一化、RoPE 位置和输出投影，并确实缩短 K/V 输入，不宣称 SVOO/SparseVideo CUDA Kernel。独立的 KV Token Selection 也通过同一真实 K/V gather 路径执行，可与 Sparse Runtime 合并，但不会与 Token Merge 或 Attention Window 静默叠加。
- Attention Window 当前同时支持历史 flat contiguous window 与可选 `(grid_shape, window_radius)` 的 3-D local Self-Attention；3-D 路径保持完整 Q，按原始 `(T,H,W)` 位置选择邻域 K/V 并切 RoPE，网格不匹配会显式失败。它仍未宣称 Flex/SVOO 三维 BlockMask 或专用 CUDA Kernel。
- Timestep Gate 对 batch 内不同 timestep 使用 active-subset 执行；新增的 `timestep_attention_policy` 对匹配的每个 batch 子集实际缩短 Self-Attention K/V，并记录 rule、完整/选中 token 数与重载后的统计。`Sparse Runtime + Attention Window` 当前明确拒绝，因为尚未实现两种 Attention wrapper 的联合编译，不会静默忽略其中一个。
- Token 当前包含 Self-Attention 输入选择/恢复和 `token_merge` 路径；后者同时支持兼容旧配置的 1-D adjacent 以及 `(T,H,W)` cuboid TokenMap，保存 anchor/group shape，在合并后的序列上执行 Attention，再 unmerge 回残差长度。三维专用 CUDA Kernel 仍未宣称完成。
- FLOPs 为 adapter 的结构估计；真实性能结论必须使用 WanVideoEvaluator 的固定 GPU、Prompt、Seed、Scheduler 和 warm-up/repeat 结果。
- 候选级 `runtime_execution` 是 post-forward trace，用于证明 Token/KV/Window/Timestep 等策略确实改变执行；GPU kernel、端到端延迟和显存仍由独立真实 Wan Evaluator 提供。
- 最新 10 轮 Full RSI 重跑（`fake-rsi-final8-20261016`）返回 `failed_rounds=0`，SearchSpace 从 `S0001` 演化到 `S0003`；产物中的 Token/KV/Sparse/Timestep/Sampling 候选出现非零实际削减比例，说明 runtime trace 已由真实三维 mixed-timestep probe 触发，而非零值占位。

## 测试证据

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
