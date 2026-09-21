# Four-GPU Lane Packer Design

## Goal

让远程闭环在生成下一轮 Controller plan 时仍然使用所有可安全使用的 GPU：普通弹性训练保留一个 Controller lane，明确的满卡训练则先生成并持久化下一轮 plan，再短暂释放本项目 Controller 以获得四卡；评价阶段保留一个 ComfyUI lane 和一个 Controller lane，把剩余 GPU 给隔离的候选 worker；任何分布式 worker 只在完整 lease 内运行，不在 rank 中途抢卡。

## Current evidence and gap

- 当前远程 `exp_0045 / M0040` 以四进程 torchrun 持有 `[0,1,2,3]` 的完整 worker lease。GPU1 的 rank 进程仍在 NCCL/同步等待，不能被当作空闲卡复用。
- `_train_one` 对四卡请求会先释放 Controller，再拿四卡 lease，因此 plan 不能在训练计算期间生成。
- `_evaluation_worker_specs` 在有训练 worker 时把 ComfyUI 评价限制为一张卡，这是正确的安全上限，但当 Controller 规划时其余两张卡没有候选任务。
- 当前 `_start_speculative_worker` 在计划完成后无条件请求释放 Controller；因此评价、Controller 规划和候选训练不能同时形成稳定的三 lane 布局。

## Design

### 1. Elastic execution request

对 distributed worker 的执行请求增加 campaign-level 的 Controller overlap cap。启用 overlap 时，执行请求分为两个明确模式：

- 普通弹性模式：验证后的请求没有明确要求完整 GPU 集合时，`max_gpu_count` 被限制为 `total_gpu_count - controller_overlap_gpus`，默认是 3；`min_gpu_count` 保持来自 plan 且不得低于 2，请求被规范化为 elastic；
- 满卡模式：验证后的 distributed 请求显式设置 `gpu_count >= total_gpu_count` 或 `min_gpu_count >= total_gpu_count` 时，保留四卡上限，不走 overlap cap；campaign 先生成并持久化 successor plan，再通过 handoff hold 释放 Controller，最后申请四卡 worker lease；
- 原始 LLM 计划不被改写，事件和 worker result 同时记录 `planned_resource_request` 与 `effective_resource_request`。

普通模式让训练开始前可以安全释放旧的 TP=4 Controller，然后由 watcher 在 worker lease 留出的 1 张卡上启动 TP=1 Controller；满卡模式则在训练开始前完成 successor plan 和 Controller handoff。训练启动后，普通模式继续异步生成下一轮 plan。

### 2. Controller GPU reservation

Controller launcher 在 vLLM CUDA 初始化前发布一个精确的 `.controller-gpu-lease.json`，包含 launcher-owned child PID、GPU 列表、创建/过期时间和状态。RemoteResourceScheduler 读取该 marker，分配 worker 时排除 Controller 的 GPU；launcher 自己不把自己的 marker 计入候选过滤。

Controller stop/restart 的 handoff 使用 `.controller-handoff-hold.json`：campaign 先写 hold，再请求 Controller release，等 worker lease 发布后删除 hold。watcher 在 hold 存在时不重新启动 vLLM，消除 release 与 worker acquire 之间的竞态。

### 3. Evaluation overlap

评价只保留一个 ComfyUI worker（除非有多个独立 task 可以安全 fan-out）。评价 lease 建立后，Controller launcher 在剩余 GPU 中选择最小可行 TP；plan ready 后候选 worker 不再主动释放 Controller，而是用当前可用水位申请 elastic worker。单 task 的正常布局是：

```
ComfyUI: 1 GPU | Controller: 1 GPU | speculative worker: 2 GPUs
```

评价完成或 Controller 请求结束后，各 lease 独立释放。候选分支只在完整 result、parent hash 和评价门禁通过后进入 archive。

### 4. Safe expansion follow-up

第一阶段不修改已启动 torchrun 的 world size。第二阶段为真实 worker 增加 chunk checkpoint/resume 合约：只在训练 chunk 完成、所有 ranks barrier 后停止并从最新 checkpoint 以新的 world size 重启。该机制用于 `2→3→4` 扩容，不允许发送信号或修改环境变量抢占运行中的 rank。

### 5. Utilization accounting

每轮事件记录 lane allocation、每卡有效 lease 时间、GPU utilization、power、memory 和 idle reason。300W 与 100% 是目标 telemetry，不使用 `nvidia-smi -pl` 强行改变主机功耗上限，也不把等待 barrier、NFS staging 或 LLM token 生成误报成满载计算。

## Safety invariants

- 同一时间同一 GPU 只属于一个 campaign-owned lane lease。
- 分布式 worker 的任一 rank 存活时，不把该 worker 的 GPU 当作空闲卡。
- Controller release 与 worker allocation 之间由 handoff hold 保护。
- ComfyUI queue/model 未释放或 lease 不确定时，scheduler fail-closed。
- 不停止未由本 campaign 精确记录的进程，尤其不触碰旧 ComfyUI daemon。
- 被拒绝/失败分支只按 exact checkpoint path 清理，append-only experience/evaluation/telemetry 永久保留。

## Acceptance evidence

1. 纯函数测试证明四卡、三卡、评价+Controller+候选三种布局和资源上限。
2. scheduler 测试证明 live Controller lease 排除对应 GPU、过期/死亡 owner 可回收、worker lease 与 Controller lease 不重叠。
3. launcher contract 测试证明 handoff hold 阻止重启、child 退出后精确清理 Controller lease。
4. campaign 测试证明普通 overlap plan 改为 effective 3-GPU worker，明确的 full-GPU plan 保留 effective 4-GPU worker，并在获取四卡前触发 successor Controller prefetch 和 handoff；评价中候选 worker 保留 Controller lane。
5. 远程短窗口验证出现 `controller_gpu_lease`、`worker_started`、`controller_plan_prefetch_started` 的正确顺序，并报告每张卡的 lane/利用率/功耗。
