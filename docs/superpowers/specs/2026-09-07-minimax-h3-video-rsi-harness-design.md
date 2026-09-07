# MiniMax H3 视频 RSI Harness 设计

## 1 范围与目标

本项目实现 Harness4H3 v1。MiniMax-H3 通过现有 ComfyUI HTTP API 作为冻结的视频生成后端；Harness 负责读取任务、构造 prompt 与 workflow、执行任务、记录 trajectory、调用独立 evaluator，并根据历史结果生成和筛选下一代 Harness candidate。

本阶段验证的问题是：在不修改 MiniMax-H3 权重、模型文件和 ComfyUI 实现的前提下，Harness 能否通过历史视频任务的执行结果，逐代改进 prompt、context policy 和少量 workflow policy，并在 held-out 或 regression tasks 上得到可量化差异。

目标仓库是独立项目，不复制当前 MinMax-H3 部署仓库的历史。只复用已验证的 ComfyUI API 协议、API-format workflow 约定及输出下载方式。

## 2 硬约束

- MiniMax-H3 和 ComfyUI workflow backend 在运行中保持只读；不训练模型，不修改权重。
- 运行中的版本不得热修改自己。所有 mutation 必须产生独立 candidate，完成 sanity、replay 和 dev evaluation 后才可 promote。
- 每个 candidate 只表达一个主要 mutation 假设，以便归因。
- 最终 promotion 由与 Harness 解耦的 evaluator 决定，不能采用 Harness 或模型的自报分数。
- V1 不引入多 Agent、训练、向量数据库、知识图谱、effect model、复杂 planner、workflow DSL、消息总线、微服务或通用 RSI 平台。
- 实现优先使用 Python 标准库；仅为 YAML 配置和可选媒体分析引入必要依赖。

## 3 系统边界与架构

```text
Task manifest
    ↓
Context policy + Candidate policy
    ↓
ComfyUI MiniMax-H3 adapter
    ↓
Video artifact + execution observation
    ↓
Trajectory JSONL
    ↓
Independent evaluator process/interface
    ↓
Recurring-failure diagnosis
    ↓
One mutation → sanity → replay → dev benchmark
    ↓
Promote or drop → candidate archive and lineage
```

MiniMax-H3 是受控生成 backend，而不是 orchestration controller。Harness 主循环确定性地选择任务、渲染 API workflow、提交任务、轮询状态、下载输出、调用 evaluator 和持久化结果。模型不会决定何时结束、如何评分或是否 promotion。

## 4 代码结构

```text
Harness4H3/
├── harness4h3/
│   ├── model/minimax_h3.py
│   ├── harness/context.py
│   ├── harness/loop.py
│   ├── harness/state.py
│   ├── tools/registry.py
│   ├── memory/trajectory.py
│   ├── evaluator/evaluator.py
│   ├── self_improve/evolve.py
│   ├── archive/store.py
│   ├── config.py
│   └── cli.py
├── configs/default.yaml
├── examples/tasks.yaml
├── examples/workflow_api.json
├── tests/
├── REUSE_MATRIX.md
├── README.md
└── pyproject.toml
```

不为尚未实现的能力创建空模块。上面的文件只在其对应行为和测试一同落地时创建。

## 5 核心组件

### 5.1 MiniMax H3 adapter

Adapter 接收已渲染的 API-format workflow，调用 `POST /prompt`，保留 `prompt_id`，轮询 `GET /history/<prompt_id>`，识别成功、执行错误和超时，并下载 history 返回的输出文件。网络异常使用有界重试；无法确认提交结果时不得盲目重复提交，以避免生成重复任务。

Adapter 通过配置读取 ComfyUI base URL、超时和输出目录，不保存凭据。测试使用本地 fake HTTP server，不依赖真实 GPU 或模型文件。

### 5.2 Task 和 context policy

任务清单定义 task id、prompt、约束、seed、split 和 evaluator 期望。split 至少包括 `sanity`、`dev`、`heldout`，失败 replay 集合由 trajectory 动态派生。

Context builder 只组合当前任务、candidate 的 prompt 前后缀、当前 workflow policy 和必要约束。它不加载完整历史、完整 archive 或无关日志。最终生成的是结构化 execution request，而不是供另一个 LLM 自由解释的对话上下文。

### 5.3 Tool registry 和执行循环

V1 registry 暴露少量通用动作：提交 workflow、查询状态、下载 artifact 和调用 evaluator。每个 tool 具有 name、description、input schema 和统一的 `ToolResult`。主循环由确定性状态机控制，并设置最大步骤数、任务超时和终止状态。

每一步记录 context digest、action、tool result、verifier/evaluator feedback 和时间信息。最终状态无论成功或失败都写入 trajectory。

### 5.4 Trajectory recorder

Trajectory 采用 append-only JSONL。每条记录包含 task id、harness version、输入、步骤、artifact、最终结果、外部 score、failure type、wall time 和可获得的成本信息。大文件只保存路径和摘要，不嵌入 JSONL。

写入采用临时文件或单条原子追加策略，异常退出不能破坏已有记录。敏感环境变量、认证头和本地凭据不得进入 trajectory。

### 5.5 外部 evaluator

Evaluator 与 Harness 通过稳定的 request/response JSON 协议解耦，可由独立 Python 进程或用户配置的外部命令实现。默认 evaluator 只使用确定性、可复现的技术指标：任务是否完成、文件是否存在且可解码、分辨率、帧数、时长、黑帧/平均亮度、基础帧间稳定性、运行时间和 failure rate。

默认 evaluator 不声称判断语义美学质量。需要语义或视觉质量时，可配置外部 VLM evaluator；只要它返回相同协议，Harness 和 evolution loop 无需改动。

Evaluator 输出至少包含 overall score、分项 metrics、critical regression 标记和失败原因。Harness 仅消费评分结果，无权修改 evaluator 的实现、命令或基准任务。

### 5.6 Self improvement 和 archive

Evolution loop 从多条 trajectory 中聚合重复 failure type 或持续低分指标。V1 使用有限、可解释的 mutation catalog，而不是让模型任意改源码：

1. Prompt mutation：添加或移除一个与重复失败相关的短约束片段。
2. Context mutation：调整 prompt 片段顺序、长度预算或约束选择策略。
3. Workflow mutation：只改变 allowlist 内一个参数，例如 steps、CFG 或已声明的稳定性开关。

每轮选择一个主要假设并生成一个或少量互相独立的 candidate。candidate 保存 id、parent、generation、mutation type、结构化 patch、reason、evidence task ids、evaluation 和 metadata。

Promotion 条件全部满足才通过：sanity 全部通过；不存在 critical regression；candidate 的 dev score 大于 parent score 加 `min_delta`；regression count 不超过配置阈值。held-out 任务不参与 mutation 选择，只用于报告版本间差异。未通过的 candidate 仍归档为 dropped，不能覆盖 parent。

## 6 数据流

一次普通运行读取 active candidate 和任务，构造 workflow，调用 ComfyUI，收集 artifact 与运行 observation，调用 evaluator，最后写入 trajectory。

一次 evolve 运行读取指定 parent 的历史 trajectory，定位重复弱点，选择一个 mutation，写出独立 candidate，在隔离的运行目录中依次执行 sanity、失败任务 replay 和 dev benchmark。选择器比较 parent 与 candidate 的同任务分数后决定 promote 或 drop。promotion 通过原子更新 active-version 指针完成，历史 candidate 文件保持不可变。

## 7 错误处理和安全性

- 配置、task manifest 或 API workflow schema 无效时，在提交远程任务前失败。
- ComfyUI 返回执行错误、history 异常、超时或输出缺失时，记录稳定的 failure type，并完成 trajectory。
- evaluator 超时、退出码非零、输出不是合法 JSON 或缺少字段时，当前 task 评价失败；不得由 Harness 补造分数。
- candidate patch 超出 allowlist、同时修改多个主要类别或无法应用时，在 sanity 前拒绝。
- active candidate 更新采用原子替换；中断后 parent 仍可运行。
- 输出目录按 run id 隔离；默认不删除用户已有视频或远端 ComfyUI 文件。

## 8 命令行界面

CLI 提供以下稳定入口：

- `harness4h3 run --tasks <manifest> --split <name>`：以 active 或指定 candidate 执行任务并记录 trajectory。
- `harness4h3 evaluate --trajectory <path>`：重新调用外部 evaluator，结果另存，不改写原始记录。
- `harness4h3 evolve --tasks <manifest>`：诊断 parent、生成 candidate、执行门禁并归档结果。
- `harness4h3 lineage`：打印 parent-child、generation、状态和分数。
- `harness4h3 validate-config`：离线验证配置、任务和 workflow，不调用 ComfyUI。

所有命令用非零退出码表达失败，并支持 `--json` 机器可读摘要。

## 9 测试与验收

离线测试必须覆盖配置校验、context 裁剪、workflow 单变量 patch、tool schema、ComfyUI API 成功/失败/超时、trajectory 追加、evaluator 协议、重复失败诊断、candidate 不可变性、promotion gate 和 lineage。

端到端测试分两层：

1. fake backend 集成测试：使用本地 HTTP server 和固定 evaluator，完整执行 H0 → H1 promote 以及 candidate drop 路径，可在 CI 中复现。
2. 真实 ComfyUI smoke：在配置了可达 endpoint 时运行最小 sanity task，验证 prompt id、history、视频下载、默认 evaluator 和 trajectory；默认测试套件不要求 GPU。

第一阶段完成标准：固定 MiniMax-H3 backend 可稳定调用；批量任务可无人干预执行；每次运行均有可复现 trajectory；evaluator 与 Harness 解耦；历史失败能产生至少一个可执行 candidate；candidate 能完成 replay/benchmark 并 keep 或 drop；archive 能展示连续 lineage；held-out/regression 报告能量化版本差异。

## 10 复用审计

实现前先提交 `REUSE_MATRIX.md`。复用重点是当前 MinMax-H3 工程已经验证的 ComfyUI `/prompt`、`/history`、`/view` 协议、API-format workflow 校验、prompt id 保留和 history polling。benchmark runner 中与特定 Windows 路径、SSH 遥测、特定模型文件、手工 benchmark 表格和部署脚本耦合的代码不直接复制。

DGM、AlphaEvolve、OpenRSI、Frontis-MA1、Phi Bench 和 CAKE 只作为 archive、外部 evaluator、可复现实验和长期方向的参考；V1 不引入它们的完整实现或训练路线。

