# Human Controller Directive Design

**Date:** 2026-09-15

## Goal

让用户可以在实验运行期间提交“下一轮优化目标”，并让真实 LLM
Controller 在下一次 `ExperimentPlan` 决策时看到、消费并落实该目标。指令
必须可审计、可恢复，并且不能绕过 TargetProfile、evaluator 或训练安全门禁。

## Current gap

Harness 已经把 `ObservationStore`、`ControllerContext`、LLM structured plan、
resource scheduler 和远端 worker 串成闭环，但没有稳定的人类输入入口。直接
向 vLLM chat endpoint 发送消息不会写入 campaign 的 observation stream，因此
不会影响正在运行的实验。

## Chosen approach

增加一个本地 CLI 入口，将用户指令写成 append-only `human_directive`
Observation。运行中的 campaign 已经在每次 Controller 调用前读取未消费的
Observation，因此不需要修改远端 worker 或引入第二个控制通道。

示例：

```bash
python -m harness4h3 directive \
  --output-root var/remote-h3-controller-20260914 \
  --text "下一轮优先降低 peak_memory，质量下降不得超过 2%"
```

指令写入后，在当前 worker/benchmark 完成后的下一个 planning boundary
生效；不对已经启动的 worker 做中途修改。

## Directive contract

新增 `harness4h3/controller/directive.py`，提供不可变的
`HumanDirective`/`submit_directive` 接口。指令至少包含：

- 唯一 `directive_id`；
- 原始文本 `instruction`，限制长度并拒绝空白输入；
- `apply_at="next_controller_plan"`；
- 创建时间、source URI 和 canonical payload SHA-256。

CLI 通过 `ObservationStore` 持久化标准 `ObservationRecord`：

- `kind="human_directive"`；
- `summary.instruction` 保存用户目标；
- `summary.directive_id`、`summary.apply_at` 用于审计和消费；
- 不保存 checkpoint、GPU 命令或聊天历史。

`RemoteCampaign._controller_context()` 将该 Observation 传给现有
`ControllerContext`；LLM prompt 通过现有 bounded observation view 读取它。
`_validate_plan_evidence()` 要求 LLM 的下一份计划消费所有新的 Observation，
因此未消费的人类指令会阻止旧 pending plan 直接执行。

## Safety boundary

人类指令可以表达下一轮的研究偏好，例如优先级、希望探索的 operator 或可接受
的软目标，但不能：

- 修改 TargetProfile 的硬约束、evaluator、quality gate 或 Pareto 定义；
- 直接指定 shell、SSH、CUDA_VISIBLE_DEVICES 或任意可执行命令；
- 让 Controller 绕过注册 operator、训练证据校验或 checkpoint 不可变性；
- 把文本目标当作已经测得的质量/硬件证据。

这些限制由现有 Controller schema、validation pipeline 和 evaluator 保持，
directive 只是只读的输入证据。

## Error handling and concurrency

- 空文本、超长文本或重复 `directive_id` 拒绝且不写入 Observation；
- 同一个 campaign/output root 下，已有 directive 通过 ObservationStore 的
  idempotent source binding 保持可重放；
- campaign 不运行时指令仍保留，下一次 `--resume` 会读取；
- worker 正在运行时不打断它，指令在下一次 Controller boundary 生效；
- 指令写入失败不改变现有 campaign 状态，也不伪造 Controller 结果。

## Testing and acceptance

单元测试覆盖：

- directive 文本校验、canonical hash 和序列化；
- CLI 写入标准 Observation，重复提交幂等；
- directive Observation 出现在 bounded Controller prompt 中；
- 旧 pending plan 在有新 directive 时不能直接复用，必须重新调用 Controller。

集成测试覆盖：

- 注入 directive 后，下一次真实/fixture Controller plan 的
  `consumed_observation_ids` 包含 directive ID；
- directive 不改变 GPU isolation、worker command 或 evaluator decision；
- campaign resume 后仍能消费未处理 directive；
- malformed directive 不会改变 campaign 或产生训练任务。

完成标准是：用户通过 CLI 提交一句目标，事件流和 Observation stream 可重放，
真实 vLLM Controller 在下一次计划请求中看到该目标，并由 Harness 按现有安全
流程执行经校验的计划。
