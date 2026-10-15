# Harness4H3 多粒度 RSI 验收记录

本记录对应 `codex/algorithm-discovery-rsi`，并按
`EvoCompress_Codex_Implementation_Plan.md` 区分 CPU/结构证据与真实 GPU 证据。

## 已完成并验证

- Layer/Block、全局对齐 FFN、Attention/Head 结构剪枝：Fake 保存、重载、参数量变化和前向形状检查通过；Diffusers Tiny Wan 也覆盖了 fused QKV 先 unfuse 再 packed head 的路径。
- Token、KV Token Selection、Sparse Runtime、Attention Window、Timestep、Sampling Step：策略持久化，重载后恢复，并在 Fake/Diffusers Wan Block 路径实际改变执行输入、Q/K/V 长度、局部 K/V 范围、每个 batch 样本的 Block 执行或 scheduler 步数。KV 选择保留完整 Q，仅按 score 选择 K/V，并支持按 Block 定位。
- 有序 Hybrid：`Head + Sparse`、`Token + Sparse`、`Layer + Sampling Step` 已有组合执行测试；`compose` 保留步骤顺序并支持最多 6 步。
- `ModelVariant`：通过 sidecar manifest 区分结构 checkpoint、runtime policy、sampler policy、算法版本和 SearchSpace 版本；runtime-only 变体不填写 structural checkpoint，参数减少固定为 0。
- Algorithm Discovery：Sandbox 中执行 Agent 的 `search()`，轨迹记录候选集合、排序和选中计划；搜索源码纳入逻辑摘要，递归子代保留父子源码差异。
- Search-Space Evolution：提案经过 Sandbox、真实 adapter 执行、保存/重载、finite 检查后注册；下一代 Program 读取新版本并能生成新算子计划。执行器通过受限的 `implementation_ref` loader 恢复，不要求为每个新类修改管理器分支。

## 部分完成 / 有条件完成

- Diffusers Wan 的 Head/Token/Sparse/Attention Window 代码已按当前安装版本（Diffusers 0.39.0）的真实模块签名实现，并有 Tiny Wan 结构 smoke；完整 Wan2.1-1.3B 视频、GPU Kernel、显存和端到端延迟尚未在本机执行。
- Sparse Runtime 当前是可执行的 Top-k KV 选择后端；它保留 Wan Q/K 归一化、RoPE 位置和输出投影，并确实缩短 K/V 输入，不宣称 SVOO/SparseVideo CUDA Kernel。独立的 KV Token Selection 也通过同一真实 K/V gather 路径执行，可与 Sparse Runtime 合并，但不会与 Token Merge 或 Attention Window 静默叠加。
- Attention Window 当前是保留 Q 长度的 block-local Self-Attention，按原始扁平 Token 位置切 RoPE，实际把每个 Q block 的 K/V 限制在同一局部块；尚未宣称三维时空 BlockMask 或专用 CUDA Kernel。
- Timestep Gate 对 batch 内不同 timestep 使用 active-subset 执行；`Sparse Runtime + Attention Window` 当前明确拒绝，因为尚未实现两种 Attention wrapper 的联合编译，不会静默忽略其中一个。
- Token 当前包含 Self-Attention 输入选择/恢复和 `token_merge` 路径；后者保存连续分组的 anchor/TokenMap 语义，在合并后的序列上执行 Attention，再 unmerge 回残差长度。三维 BlockMask 和自定义 CUDA Kernel 仍未宣称完成。
- FLOPs 为 adapter 的结构估计；真实性能结论必须使用 WanVideoEvaluator 的固定 GPU、Prompt、Seed、Scheduler 和 warm-up/repeat 结果。

## 测试证据

```text
focused algorithm/runtime tests: 33 passed
full regression: 1023 passed, 4 skipped
```

4 个 skip 分别是未设置真实 Wan checkpoint、CUDA 不可用或 CUDA RNG parity 条件，不是实现失败。

2026-10-16 的四组 CPU Fake RSI 结果和证据路径见
[`docs/evolutionary-compression-rsi-20261016.md`](evolutionary-compression-rsi-20261016.md)；2026-10-15 报告保留为历史记录。

## 真实 Wan 最小命令

先替换 `configs/evolutionary-compression.wan-algorithm-discovery.example.yaml` 中的 checkpoint、三份 Calibration/Validation/Test manifest、Wan 配置、CLIP 和 controller：

```bash
uv run --with torch --with diffusers python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.wan-algorithm-discovery.example.yaml \
  --rounds 10 --json
```

真实 GPU 验收必须额外记录 Dense 对照、稀疏 Kernel/DiT/视频三级耗时、显存、视频质量和重启后的 runtime policy 恢复；CPU Fake 结果不能替代这些证据。
