# LLM Continuous Experiment Review Design

**Date:** 2026-09-15

## Goal

让真实 LLM 在远端 H3 长实验运行期间持续承担“实验监督员”和“结果评审员”职责，而不是只在一轮实验开始前生成一次 ExperimentPlan。系统需要在不改变 evaluator 硬判定权、不抢占训练 GPU 的前提下，周期性读取运行状态，及时发现异常，并为下一步实验提供结构化建议。

## Current gap

当前 controller 已经具备结构化 ExperimentPlan、事件记录、远端资源调度、真实训练 worker 和独立 benchmark/evaluator，但远端 campaign 的主循环在一个实验完成后才调用一次 `controller.plan()`。训练、checkpoint I/O 和 benchmark 运行期间，LLM 没有可执行的评审入口，因此 GPU1 上的 LLM 服务大部分时间空闲。

## Chosen approach

采用“事件触发 + heartbeat”的混合 reviewer：

- 训练启动、资源等待、worker 进度异常、worker 完成、benchmark 完成、评估完成和失败事件立即触发评审；
- 活跃长任务运行期间默认每 60 秒触发一次轻量结构化评审；
- heartbeat 只读取遥测和事件，不生成新的训练命令，也不直接改变 evaluator 结果；
- 评审输出只能是受 schema 限制的动作：`continue`、`stop`、`replan` 或 `review_only`；
- `stop` 只在 worker 暴露安全检查点时生效，不能删除或覆盖父模型/部分 checkpoint；
- `replan` 只影响下一次 ExperimentPlan，当前已启动的安全 worker 不被强行改写；
- 资源扩缩仍由现有 scheduler 在下一次实验开始时决定，不允许 reviewer 在正在运行的 torchrun 中注入新 GPU。

## Reviewer contract

新增一个独立的结构化 reviewer 接口，避免把评审结果伪装成 ExperimentPlan：

```python
class ReviewDecision(TypedDict):
    action: Literal["continue", "stop", "replan", "review_only"]
    reason: str
    evidence_ids: list[str]
    confidence: float
    next_review_after_s: float
    risks: list[str]
```

输入包含：

- 当前实验 ID、父模型/系统 ID、operator 和已分配 GPU；
- 当前阶段：`planning`、`waiting_for_resource`、`training`、`benchmarking`、`evaluating` 或 `failed`；
- 训练遥测：loss、gradient、optimizer steps、吞吐、GPU 利用率、显存、进程状态和最近错误；
- 当前实验预算与剩余预算；
- 最近事件原文的结构化摘要；
- 最近实验完整证据、历史失败摘要和 Pareto 前沿；
- TargetProfile 和 evaluator 的只读约束。

reviewer 可以建议下一步，但不能：

- 修改 TargetProfile、evaluator、hard gates 或质量指标；
- 伪造训练/benchmark/evaluation 证据；
- 生成 shell 命令或绕过注册 operator；
- 直接批准一个候选模型；
- 抢占 GPU0 上的 ComfyUI 或 GPU1 上的 LLM。

## Runtime flow

```text
campaign starts
  ├─ immediate review: planning/resource state
  ├─ heartbeat every 60s while long task is active
  │    └─ collect telemetry → LLM review → record decision
  ├─ immediate review: worker/benchmark/evaluator event
  └─ normal controller.plan() selects the next executable ExperimentPlan
```

评审决策的执行规则：

| Action | 当前任务 | 下一轮任务 |
|---|---|---|
| `continue` | 保持运行 | 正常进入下一阶段 |
| `review_only` | 保持运行 | 不改变计划，仅记录审查 |
| `stop` | 在安全检查点停止并保留证据 | 不自动晋级候选 |
| `replan` | 当前任务继续到安全边界或结束 | 强制重新调用 controller.plan() |

每个 reviewer 请求都写入 `controller-events.jsonl`，至少包括触发原因、阶段、输入摘要哈希、provider/model、决策、耗时和错误。LLM 不可用时不改变当前 worker 的安全行为；事件记录错误并按既有 campaign 策略处理。

## Context policy

reviewer 使用三层上下文：

1. 当前实时状态和最近 10 个事件的完整结构化字段；
2. 最近 3 个实验的完整结果；
3. 更早实验、失败原因和 Pareto 前沿的压缩摘要。

heartbeat 采用独立的 reviewer prompt 和较小输出上限。60 秒是默认可配置周期；关键事件不等待周期。默认实现每个周期最多一次 reviewer 请求，并设置独立的 `max_review_calls`，防止长实验无限消耗 controller 资源。

## GPU and resource isolation

reviewer 请求通过现有单卡 vLLM 服务运行在 GPU1。资源 scheduler 继续把 GPU0 作为 ComfyUI 保留卡，训练只从其余空闲卡中按 ExperimentPlan 的 `min_gpu_count`/`max_gpu_count` 动态选择。reviewer 本身不参与训练 GPU 的分配，也不把两个 GPU 的 LLM 推理作为默认配置。

## Failure and safety handling

- schema 校验失败：记录 `controller_review_rejected`，当前任务保持安全默认行为；
- HTTP 超时或 vLLM 不可用：记录 `controller_review_unavailable`，不伪造 review decision；
- `stop` 发生在 checkpoint 写入期间：延迟到 worker 报告安全点；
- reviewer 连续建议与硬约束冲突：丢弃建议，继续由现有 validation/evaluator 决定；
- heartbeat 与关键事件同时到达：用实验阶段和事件序列号去重，保证同一事件最多一次有效处理；
- reviewer 调用失败不能改变 `parent` 不可变、子模型 reload、冻结 tensor 和 evaluator hard gate 等现有证据门槛。

## Testing and acceptance

单元测试覆盖：

- reviewer JSON schema 接受合法动作并拒绝额外字段；
- prompt 包含当前阶段、遥测、事件摘要和 GPU 隔离信息；
- `stop`、`replan`、`continue` 的状态转换符合 contract；
- reviewer 调用计数独立于 ExperimentPlan 调用计数；
- reviewer 不可用时不会伪造成功决策。

集成测试覆盖：

- heartbeat 在长任务期间按周期触发，并在 worker 完成事件时立即触发；
- 同一事件不会产生重复有效评审；
- 评审记录可从 `controller-events.jsonl` 重放；
- reviewer 运行时 GPU 资源请求仍排除 GPU0/GPU1；
- reviewer 建议 `replan` 后，下一次真实 LLM plan 会重新读取最新观察，而不是复用旧计划。

真实验收标准：

1. vLLM provider 真实返回至少一次合法 `ReviewDecision`；
2. 训练或 benchmark 持续超过一个 heartbeat 周期时，事件流中出现至少一次 `controller_review_completed`；
3. worker、benchmark、evaluator 的真实结果仍完整产生；
4. reviewer 不能改变任何 evaluator hard gate 或 GPU 保留策略；
5. 最终 trajectory 同时包含真实 LLM plan、真实 LLM review、真实 worker 结果和 evaluator 决策。
