# Controller-Owned Elastic Resource Scheduling Design

## Goal

让远程优化 loop 在 GPU 被其他任务占用时能够持久排队，并在训练任务启动边界根据 Controller 明确允许的资源范围动态使用 2、3 或 4 张 GPU；调度器不得抢占、终止或改变其他任务。

## Ownership boundary

Controller 仍然是唯一的优化决策源。Controller 决定是否允许弹性资源、最小/最大 GPU 数、是否分布式以及资源不可用时等待还是重新诊断。Scheduler 只读取远端资源快照、维护本 campaign 的 pending plan、锁定可用 GPU、生成实际分配，并执行已经验证过的固定命令。Scheduler 不得选择优化算子、改变训练超参或将不可弹性计划自动改成另一种计划。

## Resource contract

训练计划的 `resource_request` 增加以下字段：

```json
{
  "gpu_count": 4,
  "min_gpu_count": 2,
  "max_gpu_count": 4,
  "elastic": true,
  "distributed": true,
  "exclusive": false,
  "evaluation_workers": 1,
  "on_unavailable": "wait"
}
```

`gpu_count` 是 Controller 的首选数量；`min_gpu_count` 和 `max_gpu_count` 是 Controller 授权的弹性范围。`elastic=false` 时必须满足精确的 `gpu_count`。分布式训练要求实际分配至少 2 张 GPU。CPU-only 算子仍使用 `gpu_count=0`。

## Queue and allocation behavior

1. Controller 生成并通过证据、schema 和资源契约校验的 plan。
2. Scheduler 将 plan 写入本 campaign 的 pending queue，并在每次 loop 边界及固定轮询间隔重新读取 `nvidia-smi`。
3. 若可用 GPU 数达到 Controller 授权范围，Scheduler 优先选择不超过 `max_gpu_count` 的最大可用数量，并至少满足 `min_gpu_count`；实际分配只从空闲 GPU 中选择。
4. 若资源不足且 `on_unavailable=wait`，loop 保留 pending plan，记录 `resource_queued`/`resource_retry`，等待下一次调度，不重新规划优化动作。
5. 若资源不足且 `on_unavailable=replan`，loop 只把资源事实反馈给 Controller，由 Controller 生成下一计划。
6. 训练进程启动后不热插拔 GPU；动态调整只发生在两个训练任务之间。启动命令的 `--nproc_per_node` 和 `CUDA_VISIBLE_DEVICES` 必须使用 Scheduler 的实际分配，并记录在事件流中。

## Persistence and recovery

Pending plan、Controller plan hash、资源请求、重试次数和最近一次资源快照写入 campaign state。进程重启后先恢复 pending plan，再读取新快照；不会重复执行已经完成的 plan。等待超过 Controller 的预算或 stop condition 时，loop 结束并明确记录 `resources_unavailable`。

## Observability

`controller-events.jsonl` 至少记录：

- `resource_queued`：计划进入等待队列及原因；
- `resource_retry`：重试时间、快照和仍缺少的 GPU 数；
- `resource_scheduled`：请求与实际分配的 GPU；
- `worker_started`：固定 launcher、`CUDA_VISIBLE_DEVICES` 与实际 `nproc_per_node`；
- `worker_completed` 或 `resource_wait_timeout`：结果及下一轮原因。

事件内容不得包含 SSH 密钥、token 或密码。

## Failure handling

- 已占用 GPU 上的进程一律视为外部任务，Scheduler 不发送终止信号。
- GPU 快照解析失败、资源契约不一致或实际分配不满足 Controller 下限时，不启动 worker，记录安全拒绝。
- torchrun 返回非零或 worker 结果缺失时，导入失败 Observation，并由 Controller 在下一轮决定恢复、重试或回滚。
- `exclusive=true` 保留严格语义：任何外部计算进程存在时都等待；弹性共享训练必须由 Controller 显式声明 `exclusive=false`，并仍要求分配 GPU 的显存占用低于安全阈值。

## Verification

- 单元测试覆盖精确资源请求、弹性范围、部分 GPU 可用、低于最小值排队、exclusive 安全拒绝和分配数量。
- 集成测试覆盖 pending plan 持久化、重启恢复、重试后启动命令使用实际 GPU 数，以及 `on_unavailable=replan` 只触发 Controller 反馈。
- 远程 sanity run 使用至少 3 个 loop iteration，验证计划排队/执行、worker、评测、Observation 反馈和下一次 Controller 计划；不停止远端已有任务。
