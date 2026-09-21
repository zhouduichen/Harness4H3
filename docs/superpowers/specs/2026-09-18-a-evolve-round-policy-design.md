# A-Evolve 风格远程 H3 自主优化控制面设计

## 目标

在现有 Harness4H3 真实 MiniMax-H3 闭环之上增加一个明确的 round-level
research policy 控制面，使远程部署的本地 LLM 可以在没有 Codex 长时间连接的
情况下持续迭代剪枝、量化、蒸馏、LPL、TDTM、CI-DL 等已被运行时能力探测证明
可执行的优化，同时让评价期间的 GPU 被安全地填充。

本设计借鉴 A-Evolve/A-Evolve-Training 的不变 substrate、同构 trial worker、
结构化观察、策略演化、固定 gate/reload 闭环；不引入未公开的
A-Evolve-Training 内部训练实现，也不让 LLM 改写执行安全边界。

## 不变约束

- 基础 H3 checkpoint、数据/评价切分、ComfyUI workflow、worker 入口、硬约束、
  GPU lease 和 checkpoint retention 规则由 operator/configuration 固定。
- LLM 只能产生结构化 `ExperimentPlan` 和下一轮 `RoundPolicy`，不能输出命令、
  任意路径、kill 外部进程、改变 evaluator gate 或直接分配 GPU。
- 任何优化方法只有在 live ComfyUI capability probe、worker contract 和
  operator registry 同时确认后才可进入可选 operator 集合。LPL、TDTM、CI-DL
  不存在时必须显式记录为 unavailable，不能用名称假装成功。
- 同一 GPU 同一时间只能属于一个 campaign-owned lease；外部进程 fail-closed，
  不被本项目停止。
- 300 W 和 100% utilization 是测量和调度优化目标，不通过 `nvidia-smi -pl`
  或超出温度、功耗、显存和外部进程隔离约束强行实现。

## A-Evolve 映射

```text
Solve    -> trusted H3 trial worker / speculative worker
Observe  -> experiment JSONL + evaluation + lane/power telemetry
Evolve   -> remote local LLM produces RoundPolicy and next ExperimentPlan
Gate     -> schema/resource validation + held-out benchmark + hard constraints
Reload   -> accepted recipe/candidate metadata becomes the next round substrate view
```

worker 是 memory-free 的：它只接收当前 round 的 immutable substrate 引用、
被分配的 recipe 和资源 lease，不读取完整历史。LLM 的经验来自一个有界的
`discovery_digest`，而不是把所有日志塞进上下文。

## 核心数据结构

### RoundPolicy

`RoundPolicy` 是追加写入 `round-policy.jsonl` 的小型策略快照，每轮最多一个
active 版本：

```json
{
  "schema_version": 1,
  "round_id": "R0012",
  "substrate_digest": "sha256:...",
  "search_mode": "runtime_efficiency",
  "allowed_operators": ["step_distill", "lpl"],
  "axis_budget": {"max_trials": 2, "max_gpu_hours": 8.0},
  "objective": {"quality_floor": 0.82, "latency_weight": 0.35},
  "fixed_evaluation": {"split": "heldout", "recipe_digest": "sha256:..."},
  "resource_policy": {"min_training_gpus": 2, "controller_overlap_gpus": 1},
  "stop_conditions": ["critical_regression", "no_safe_lease", "budget_exhausted"],
  "source_observation_ids": ["obs-..."],
  "created_at": "..."
}
```

Validator 必须拒绝未知字段、改变 substrate/evaluator digest、未注册 operator、
低于训练最小 GPU 数的 overlap，以及超过 round budget 的策略。`RoundPolicy`
不是 `ExperimentPlan` 的替代品；它只限制下一轮搜索空间，执行仍使用现有
`ExperimentPlan` validator。

实现上，主 Controller 的 `ExperimentPlan` 可以携带一个严格的 `round_policy`。
campaign 以当前 substrate/evaluator digest、registry 和 scheduler GPU 数再次验证，
通过后原子写入 `active-round-policy.json` 并镜像到 campaign state。并行预取只能
读取 active policy，不能改变它；没有该字段的旧计划继续按无 active policy 的兼容
路径执行。

### DiscoveryDigest

`DiscoveryDigest` 从 append-only `ExperimentRecord`、`ObservationRecord`、
evaluation 和 telemetry 生成，固定上限：

- 最近 8 条已完成实验；
- 每个 operator 最多 2 条代表性成功/失败记录；
- Pareto/frontier 最多 4 条；
- 失败模式计数最多 16 项；
- 每条 recipe/evaluation/telemetry 摘要最多 2 KiB；
- 不包含 checkpoint bytes、视频、完整 stdout 或重复的原始 JSON。

摘要同时保存 `source_observation_ids`、source digest 和生成时间，LLM 不能把
摘要中的推断当成硬证据。下一轮 plan 必须引用它实际消费的 observation ID。

### RoundGate

固定 gate 返回 `accepted/rejected/replan/waiting` 之一，并记录：

- worker result 是否真实、parent hash 是否匹配、child hash 是否存在；
- held-out quality、critical regression、latency、显存、能耗和目标 profile；
- capability、lane、lease、power/utilization 证据；
- retention outcome 和可恢复失败类型。

只有 gate 通过才会更新 active model/system 或 Pareto archive。拒绝和失败只删
除精确的 candidate checkpoint/temporary artifacts，永久保留 recipe、评价、日志、
observation 和失败经验。

## 四卡调度与评价期填充

调度器继续是唯一的 GPU 权威，LLM 只能声明期望资源。正常四卡布局按实时
waterline 选择：

```text
evaluation:  ComfyUI 1 GPU
controller:  local vLLM 1 GPU
trial:       elastic worker 2 GPUs
```

当评价任务有多个相互独立的 task 时，可以用多个 ComfyUI worker 填充空闲卡，
但不得为了提高 utilization 重复同一个 benchmark，除非 plan 明确开启统计重复。
当 trial 的最小分布式规模为 3 时采用 `ComfyUI 1 + controller 0 + trial 3`，
并在训练期间提前完成下一轮 plan；当训练需要 4 卡时，plan 必须在拿 worker
lease 之前生成并持久化，不能中途抢卡。

评价与训练不允许修改已启动 torchrun 的 world size。CPU-only 的剪枝/量化 worker
运行期间，会并行提前生成一个经过 operator filter 的 GPU-fill successor，并以
`parallel_prefetched_plan` 有界持久化；评价回调优先消费这个 plan，避免等主
successor 返回后才开始第二次 LLM 请求。未来的 2→3→4 扩容只能
在 checkpoint chunk、全 rank barrier 和新的完整 lease 边界发生。

主 Controller 的 n-way 候选中若已经包含合法的 distributed GPU candidate，CPU-only
主计划会直接复用该候选启动一个 sibling speculative worker，不再额外等待第二次
LLM 请求；该 sibling 仍使用独立 child/lease/result，并且必须经过自己的评价门禁。

所有阶段记录 `lane_allocation`、每卡 lease 时间、GPU utilization、power、
memory、idle reason。评价完成后 ComfyUI 按 `idle_release` 调用 `/free`，确认
队列为空和显存降到 waterline 后释放 lease；不需要时卸载模型/停止 campaign-owned
worker。按需 launcher 还对 benchmark lease 设置有限 TTL，SSH/campaign 崩溃时
自动退出并回收自己的子进程；既有外部 ComfyUI daemon 不由本项目停止。

## 持久化与重启

`campaign_state.json` 增加以下有界游标：

- active round policy ID/digest；
- pending/ready next plan 及其 source child/parent；
- discovery digest 的 source IDs/digest；
- pipeline stage、lane allocation 和 lease owner；
- gate 状态和 retention 状态。

每次状态更新原子写入并追加 `controller-events.jsonl`。远程 watcher 只负责在
外部 GPU 空闲且四卡通过 gate 时启动 campaign，读取 terminal result 后退出；
Codex 不承担持续监控责任。

## 实现边界

第一阶段实现以下小而完整的 vertical slice：

1. 新增独立 `RoundPolicy`/`DiscoveryDigest` 模块和 schema/unit tests。
2. 将 bounded digest 接入现有 `ControllerContext`，保留现有上下文上限。
3. 在现有 remote campaign boundary 加入 RoundPolicy 验证和 RoundGate 事件，
   不重写已验证的真实 worker。
4. 将评价期 plan prefetch 和 speculative worker 的 lane allocation 统一记录，
   确保 `ComfyUI + Controller + trial` 不重叠且下一轮 plan 可恢复。
5. 将 capability evidence 作为 plan 的必要前置条件，补齐 LPL/TDTM/CI-DL 的
   unavailable/available 分支测试。
6. 使用现有 exact-path checkpoint retention，补充 round policy 只保留 recipe
   和 digest、不保留无用全量权重的验证。

不在第一阶段引入第三方 A-Evolve runtime、修改 H3 模型结构、或实现尚未有
真实 worker contract 的优化方法。

## 验收证据

- RoundPolicy schema 拒绝越权字段、错误 digest、未注册 operator 和非法资源策略。
- DiscoveryDigest 在长 JSONL 上仍满足条数/字节上限，并能保留前一轮失败 recipe
  作为下一轮上下文引用。
- 真实/测试 Controller prompt 只收到 digest，不收到 checkpoint bytes 或无限历史。
- 评价期间事件顺序出现 `controller_plan_prefetch_started`、`lane_allocation`、
  `worker_started`，且 GPU lease 集合不相交。
- CPU-only worker 期间出现 `controller_plan_parallel_prefetch_armed`，并在评价
  开始时复用同一个 `parallel_prefetched_plan`；没有安全两卡 lease 时记录等待
  原因而不虚报满载。
- ComfyUI `/free`、lease release、worker completion 和 retention 的异常路径均
  fail-closed，不会杀外部任务。
- 远程短窗口运行报告每卡 lane、utilization、power、memory、idle reason；若
  外部进程占卡，campaign 等待而不是声称 4 卡满载。
- `tools/remote-validation-window.sh --start` 是唯一的有界实测入口：默认只读，
  需要显式解除 operator pause，达到迭代/时间上限后 graceful pause，并由
  `summarize_remote_validation.py` 输出不含模型字节的 JSON/Markdown 证据。
- `DiscoveryDigest` 同时携带最近有界 `pipeline_telemetry`；provider 只保留
  lane、underutilized GPU、prefetch 等摘要和 per-GPU power/utilization，不能把
  原始 telemetry 或 checkpoint 内容放进 LLM context。
- 现有 remote orchestration、checkpoint、ComfyUI 和 worker contract 测试保持
  全部通过。
