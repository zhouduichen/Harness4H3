# RoundPolicy 生命周期设计

## 目标

让远程本地 LLM 的有界 `RoundPolicy` 真正参与下一轮 H3 搜索，而不是只作为
可选的上下文字段存在。策略可以由主规划请求提出，但执行面仍由 Harness4H3
验证 substrate、评价 recipe、已注册 operator 和 GPU 约束。

## 设计

`ExperimentPlan` 增加可选的严格 `round_policy` 对象。批量候选中的策略只随被
选中的主计划生效；`prefetch`、`parallel_gpu_fill` 和 selector 不得激活策略。
策略通过 `RoundPolicy.from_dict` 和 `validate_round_policy` 验证：

- `substrate_digest` 必须匹配当前 target、workflow、worker contract 和允许的
  operator 集合；
- `fixed_evaluation.recipe_digest` 必须匹配当前 held-out evaluator signature；
- `allowed_operators` 必须来自当前 registry；
- 训练最小 GPU 数至少为 2，Controller overlap 与训练下限不能超过 4 卡；
- 主计划的 GPU-hour 预算不能超过策略的 round budget；
- active policy 的 operator 和训练 GPU 下限约束下一次主计划。

策略写入独立的 `active-round-policy.json`，并镜像到
`campaign_state.json`。独立 sidecar 避免主规划和并行预取同时更新 campaign
state 时丢失策略；写入使用现有原子 JSON 边界。损坏或 digest 过期的策略只会
产生 `round_policy_unavailable` 并让本次计划回到安全拒绝路径。

策略的可变执行进度单独保存在 `campaign_state.json.round_policy_progress`：
它记录 `round_id`、已完成评价的唯一 `experiment_id`、累计 worker
`gpu_hours` 和 `stop_reason`。只有候选完成评价的持久化边界才会增加
`trials_completed`；训练中的预取、重复恢复和未完成 worker 不消耗
`max_trials`。达到 `max_trials` 或 `max_gpu_hours` 后，系统原子地记录
`budget_exhausted`，清理待执行/预取游标，并拒绝新的 Controller 请求；重启后
同一 `round_id` 从该游标恢复，新 `round_id` 才会创建新预算。

## 不变约束

- 预取计划可以读取 active policy，但永远不能更新它。
- 策略不能改变 worker 命令、路径、checkpoint retention、评价 hard gate 或
  外部进程处理规则。
- 旧的 `ExperimentPlan`、旧 campaign state 和 RuleBased controller 继续可读；
  没有策略时保持现有行为。
- 远程服务同步与验证期间不启动 campaign 或 idle watcher。

## 验证

- schema 测试覆盖策略可选字段和未知字段拒绝；
- campaign 测试覆盖主计划激活策略、预取计划不能激活策略、operator/GPU/budget
  违反时拒绝；
- sidecar 与 campaign state 重启读取保持一致；
- 完整本地回归通过后再以 `REMOTE_START_CONTROLLER=0` 同步远程。
