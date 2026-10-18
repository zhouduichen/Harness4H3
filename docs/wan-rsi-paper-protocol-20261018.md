# Wan2.1 RSI 对照协议（六臂）

## 实验问题

对同一 Wan2.1-T2V-1.3B checkpoint、验证集、设备、dtype、seed、生成预算和质量门限，比较 Agent、Memory、算法改进、固定扩展算子空间和在线搜索空间演化分别带来的收益。真实 CUDA profile 作为所有 Agent 臂共用的计算代价证据；压缩收益必须通过真实 Wan 生成测量，不使用结构代理值作为速度结论。

## 六个对照臂

| Arm | Controller | Memory | 搜索策略适应 | 算法改进 | 搜索空间 | 对照作用 |
|---|---|---:|---:|---:|---|---|
| `fixed` | Deterministic | 关 | 关 | 关 | 默认目录固定 | 固定算法基线 |
| `coding-agent` | 同一 LLM | 关 | 关 | 开 | 默认目录固定 | 普通 Coding Agent |
| `agent-memory` | 同一 LLM | 开 | 关 | 开 | 默认目录固定 | Memory 增量 |
| `rsi-fixed-space` | 同一 LLM | 开 | 开 | 开 | 默认目录固定 | 搜索/反馈 RSI 增量 |
| `rsi-expanded-frozen` | 同一 LLM | 开 | 开 | 开 | 扩展目录冻结 | 可用算子扩展增量 |
| `rsi` | 同一 LLM | 开 | 开 | 开 | 在线演化 | 完整搜索空间 RSI |

冻结扩展目录基于 Harness4H3 现有 Wan 搜索空间，仅加入已有 `timestep_attention_policy` 算子，版本化为 `S0002`。它不会新增 kernel，也不会在实验中继续变化。在线演化臂从默认目录开始，按已有的提案验证和注册流程演化。这样可以把“固定扩展空间”和“演化空间”拆成两个可识别因素。

## 当前配置与预算

六臂配置位于 [`configs/wan-rsi-paper-20261018/`](/Users/huangjiahao/MinMax-H3/worktrees/Harness4H3-wan-rsi-profiling/configs/wan-rsi-paper-20261018/)；对应本地预检报告位于 [`wan-rsi-paper-protocol-20261018.json`](/Users/huangjiahao/MinMax-H3/worktrees/Harness4H3-wan-rsi-profiling/docs/wan-rsi-paper-protocol-20261018.json)，当前配置指纹为 `3241bbc8706670815f9a9ac667dd2ccb3109c495c522ba64cf535c4cd2c273ff`。本机未挂载模型和 profile，因此该报告只验证六臂配置匹配，不代表远端资产存在。配置中的 checkpoint、CLIP、评测配置、manifest、profile 和输出根均使用相对路径；从服务器 `ai4mbse/configs/wan-rsi-paper-20261018` 运行时，所有实验输入和输出都必须位于 `ai4mbse` 内。

六份配置固定源码标识为 `1290ae7+source-tree-v2:ac9be420e65ec3708b555db3957674748e4d20f12a8b5714c9d15e895e9b4580`。它按文件相对路径排序，对 `harness4h3/`、`tools/`、`pyproject.toml` 和 `requirements*.txt` 的路径名与文件内容计算 SHA-256，忽略 `__pycache__` 和 `.pyc`。复制到 `ai4mbse` 后必须保留这些源码文件字节；若代码有变化，应重算源码标识并重建协议 fingerprint。

真实 CUDA profile 的历史测量和 checkpoint 摘要见 [Wan GPU profiling report](</Users/huangjiahao/MinMax-H3/worktrees/Harness4H3-wan-rsi-profiling/docs/superpowers/reports/2026-10-18-wan-gpu-profiling.md>)。启动六臂前，先把匹配 checkpoint、CLIP 模型、配置、验证 manifest 和完整 profile 放到 `ai4mbse/models/` 与 `ai4mbse/experiments/wan-rsi-hybrid-20261018/`，并确认 profile 中 checkpoint SHA 与模型一致。正式六臂尚未运行。预检命令只校验配置，不会执行模型或访问 `ai4mbse` 之外的服务器目录。运行六臂期间不要并行占用同一 GPU；每臂输出根独立保存在 `ai4mbse/experiments/wan-rsi-paper-20261018/`。

```bash
python3 -m tools.validate_evolution_protocol --paper \
  --run fixed=configs/wan-rsi-paper-20261018/fixed.yaml \
  --run coding-agent=configs/wan-rsi-paper-20261018/coding-agent.yaml \
  --run agent-memory=configs/wan-rsi-paper-20261018/agent-memory.yaml \
  --run rsi-fixed-space=configs/wan-rsi-paper-20261018/rsi-fixed-space.yaml \
  --run rsi-expanded-frozen=configs/wan-rsi-paper-20261018/rsi-expanded-frozen.yaml \
  --run rsi=configs/wan-rsi-paper-20261018/rsi.yaml \
  --rounds 3 --expected-cases 6 --latency-repeats 3
```

正式运行某一臂时，从服务器 `ai4mbse/configs/wan-rsi-paper-20261018` 执行，并用 `PYTHONPATH=../..:../../third_party/SparseVideo python -m harness4h3 evolve-compression --config <arm>.yaml --rounds 3`；这样 checkpoint、CLIP 和评测文件的相对路径以及输出根都限定在 `ai4mbse` 内。

## 结果表必填项

- **生成质量：** Validation CLIP/质量指标、视觉结构门限通过率、相对 Dense parent 的质量差；Test split 只用于最后一次冻结候选评估。
- **真实性能：** 实际 CUDA 端到端延迟、Transformer/去噪延迟、每步延迟和峰值 allocated/reserved 显存；分别报告未压缩生成和候选生成。
- **模型规模：** 压缩前后可训练参数/总参数、参数缩减比例和 checkpoint 大小；动态运行时稀疏单独报告，不能记作参数缩减。
- **搜索成本：** 每臂总 GPU 时间、真实模型评估数、候选数、失败/无效候选数、Agent 调用数和可用时的 token 数。失败尝试计入预算。
- **算法发现：** 生成算法数、有效且新颖的算法数、算法类别、通过真实 Wan 质量门限并进入 Pareto 集的数量、父子算法 lineage 和代码摘要。

按每个 arm 报告 raw runs、均值/中位数和离散度；配对 case 与 seed 计算 arm 间差异。任何没有真实模型评估、违反质量门限或无法加载的候选都保留在搜索成本统计中，但不得作为成功压缩或算法发现。

## 解释边界

曾有一轮远程 RSI 搜索 80/80 候选测量失败（映射型参数接口、head-prune reload 维度和 Hybrid KV 选择问题）。这些记录只能作为失败基线；只有当前代码中 `valid=true`、完成真实 CUDA/质量评估并满足同一 parent/预算条件的候选才能计为有效结果。当前六臂只是已预检的实验设计，不代表 RSI 或任一压缩方法已经优于基线。
