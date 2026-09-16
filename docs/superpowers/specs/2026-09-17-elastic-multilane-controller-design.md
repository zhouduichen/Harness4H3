# Elastic Multi-Lane Controller Design

## Goal

让远程本地部署的 LLM 独立驱动 MinMax-H3 的连续优化循环，并在评估、规划、训练之间动态复用 4 张 GPU。任何候选分支都必须经过资源 lease、真实 worker 结果和评估门禁后才能影响 active/Pareto lineage。

## Evidence and current gap

当前 campaign 已经能够提前生成一条 plan、启动一条 speculative worker、限制 context，并清理被拒绝 checkpoint。但它仍有三个结构性空洞：

1. `pipeline_max_inflight` 被限制为 1；当 LLM 选择 CPU-only pruning/quantize 时，评估期间没有 GPU worker 可以启动，空闲卡只能等待。
2. ComfyUI backend 的单任务等待依赖长 timeout，超时后不带 prompt id 的精确取消；队列异常会延长整个 campaign 的恢复时间。
3. vLLM launcher 的强制回收路径最终无条件 `wait` 子进程。NFS/RPC 导致子进程进入 D 状态时，launcher 本身会被拖住，动态 GPU 调度失效。

## Design

### 1. Resource lanes and leases

- **Evaluation lane**：ComfyUI 只在评估 lease 持有的 GPU 上加载模型；评估完成或超时后调用 `/free(unload_models=true, free_memory=true)`，验证队列为空和显存水位后释放 lease。
- **Controller lane**：vLLM 根据实时显存、ComfyUI lease 和 worker lease 选择 TP=1/2/4。启动、健康检查和回收都必须有界；控制器不可抢占其他任务的 GPU。
- **Training lanes**：主 worker 和最多一个独立 GPU fill worker 使用不重叠 GPU lease。每个分支拥有独立 request/config/result/checkpoint 目录，不直接修改 active model、Pareto archive 或 evaluation records。
- CPU-only operator 不伪造 GPU 负载。如果主计划是 CPU-only 且有空闲 GPU，LLM 会收到 `parallel_gpu_fill` 意图，生成一个仅允许 GPU 训练 operator 的独立候选；如果没有安全的 GPU 计划，保持空闲并记录原因。

### 2. Plan timing and lineage

训练开始前优先生成下一轮 primary plan。评估开始后，若 primary plan 已存在就立即启动；若其为 CPU-only，则异步请求一个 GPU fill plan。第二个 plan 使用新的 experiment id、同一份有界经验上下文和明确的并行意图，结果只进入 candidate pool。当前评估决定后：

- 当前候选被拒绝或 lineage 改变：取消/清理所有以该候选为 parent 的 fill 分支。
- 当前候选 accepted 或 Pareto-eligible：保留已完成分支作为 unevaluated candidate，下一轮逐个评估。
- 任何 checkpoint 文件存在都不能推断 promotion；必须有完整 result JSON、parent hash、worker success 和真实评估。

### 3. Evaluation watchdog

每个 prompt 记录 `prompt_id`、提交时间、最后一次 history 状态和 deadline。超时或 ComfyUI API 异常时，只取消该 prompt，写入 `evaluation_timeout`/`evaluation_cancelled` evidence，然后执行 idle-release；不会杀掉未归属本 campaign 的 ComfyUI 服务。默认 deadline 要覆盖当前约 10 分钟的真实 H3 任务，但不允许无限等待。

### 4. Controller watchdog

launcher 不在不可中断 D 状态子进程上无限 `wait`。TERM/KILL 只作用于本 launcher 记录的 vLLM PID；若精确 child 在 bounded grace 后仍为 D 状态，记录 orphan PID、退出本轮选择并轮询回收，不阻塞主队列，也不启动可能冲突的第二个 controller。

### 5. Memory and context

经验、recipe、失败原因和资源 telemetry 全量 append-only 保存；prompt 只发送最近、当前 lineage、失败相关和 relevance-ranked 的有限记录。checkpoint 只保留 active、Pareto rollback 集合和 in-flight branches，拒绝候选及失败分支在 durable evidence 后按 exact path 清理。

## Safety invariants

- 任意时刻同一 GPU 只能属于一个 campaign-owned lease。
- ComfyUI queue 非空、模型未释放或 lease 状态不确定时，GPU 不可分配给 Controller/worker。
- 不停止未被本 campaign 明确记录的进程。
- 不使用 `nvidia-smi -pl` 或其它改变主机功耗上限的手段；300W 只是目标/telemetry，不是可强制保证的结果。
- 分支 worker 失败、超时或 result 缺失时，主循环继续，并将失败作为下一次 LLM 的经验。

## Acceptance evidence

1. 单元/集成测试证明：CPU-only primary 会触发受限 GPU fill plan；两个分支的 lease、checkpoint 和 lineage 隔离。
2. 人为 stalled prompt 在配置 deadline 内被精确取消，ComfyUI 释放并写入 timeout evidence。
3. vLLM 启动失败或 D-state child 不再阻塞 launcher；launcher 只轮询 exact orphan PID。
4. 远程事件顺序出现 `evaluation_started`、`controller_plan_prefetch_ready`、`speculative_worker_started`，并能在评估结束后继续下一候选。
5. 远程 telemetry 报告每张卡的 utilization、power、memory；不把低功耗/低利用率样本宣称成 300W/100%。
