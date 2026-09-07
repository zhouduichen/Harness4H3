# Harness4H3

Harness4H3 是面向冻结 MiniMax-H3 ComfyUI 视频生成后端的轻量级 self-improving harness。它确定性地执行任务、保存完整 trajectory、通过独立 evaluator 评分，并只在 sanity、replay、dev benchmark 全部通过后提升新的 prompt、context 或 workflow candidate。

## 当前能力

- MiniMax-H3 ComfyUI adapter：`/prompt` 提交、`/history/<id>` 轮询、`/view` 下载。
- 受限 context/workflow 渲染：candidate 只能修改配置 allowlist 中的字段。
- Append-only JSONL trajectory：记录 context digest、动作、结果、失败类型、评分与耗时，并脱敏常见凭据字段。
- 独立 evaluator 子进程：验证生成成功、artifact、视频解码、分辨率、帧数、亮度、黑帧和基础帧间稳定性。
- RSI loop：聚合重复失败，每个 candidate 只应用一个 mutation，完成 sanity/replay/dev 后 keep 或 drop。
- 不可变 candidate archive、原子 active 指针及 H0 → H1 → H2 lineage。

## 安装

需要 Python 3.9 或更高版本。建议使用独立环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
```

开发和测试：

```bash
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest -q
```

## 配置

默认配置在 `configs/default.yaml`，示例任务和 API-format workflow 在 `examples/`。先根据实际 ComfyUI workflow 调整 node id、input 名和模型文件名，再执行离线校验：

```bash
.venv/bin/python -m harness4h3 --config configs/default.yaml validate-config --json
```

默认 endpoint 是 `http://127.0.0.1:8188`。可在不修改配置文件的情况下覆盖：

```bash
COMFYUI_BASE_URL=http://your-comfyui-host:8188 \
  .venv/bin/python -m harness4h3 --config configs/default.yaml run --split sanity --json
```

配置中的 workflow mutation allowlist 只有 `steps`、`cfg` 和显式声明的 `stability` target。模型名、LoRA、VAE、输出路径和任意节点不能被 candidate 自行修改。

示例 MiniMax-H3 Turbo workflow 默认启用 `low_vram`，以便在 16 GB 级显卡上保留安全余量。只有完成独立显存验证后才应关闭它。

## 使用流程

建立 H0 并运行 dev 任务：

```bash
.venv/bin/python -m harness4h3 --config configs/default.yaml run --split dev
```

从 H0 的重复失败生成独立 candidate，并执行门禁：

```bash
.venv/bin/python -m harness4h3 --config configs/default.yaml evolve --json
```

查看 lineage：

```bash
.venv/bin/python -m harness4h3 --config configs/default.yaml lineage
```

对既有 artifact 重新评价，结果写入单独 JSONL，不改写原 trajectory：

```bash
.venv/bin/python -m harness4h3 --config configs/default.yaml evaluate \
  --trajectory var/trajectories.jsonl --output var/reevaluated.jsonl
```

held-out 任务只用于版本差异报告，不参与诊断或 promotion：

```bash
.venv/bin/python -m harness4h3 --config configs/default.yaml run --split heldout
```

## Evaluator 信任边界

Harness 通过 JSON stdin/stdout 调用 evaluator。若 `evaluator.command` 为空，使用内置独立 worker；也可以配置外部 VLM evaluator。外部程序必须返回：

```json
{
  "score": 0.82,
  "metrics": {"semantic_quality": 0.86, "temporal_consistency": 0.78},
  "critical_regression": false,
  "failure_type": null
}
```

非法 JSON、超时、非零退出码、缺少 score/metrics 或越界分数都采用 fail-closed 行为。Candidate policy 无法修改 evaluator 命令、基准任务或 promotion 阈值。

## 数据目录

默认运行数据位于 `var/`，不提交 Git：

```text
var/
├── outputs/Hx/<task-id>/
├── trajectories.jsonl
└── archive/
    ├── active.json
    ├── candidates/Hx.json
    └── outcomes/Hx.json
```

Candidate 文件创建后不覆盖。被 drop 的 candidate 仍保留 outcome 与 parent 关系；promotion 只原子更新 `active.json`。

## 故障恢复

- 提交前的配置、task 或 workflow 错误不会访问 ComfyUI。
- POST `/prompt` 不自动重试，避免不确定状态下重复生成；GET history 可有限重试。
- 远端执行、下载或 evaluator 失败仍会写 trajectory，并保留稳定的 failure type。
- 中断不会覆盖 parent candidate；重新运行前先检查 ComfyUI queue 和已生成 artifact。

## V1 非目标

本版本不训练 MiniMax-H3，不修改模型或 ComfyUI 源码，不做多 Agent、任意源码自修改、复杂 planner、workflow DSL、数据库、向量检索、effect model、SFT 或 RL。语义美学评分需要通过外部 evaluator 接入，不由 Harness 或 MiniMax-H3 自报。
