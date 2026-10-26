# HarnessPro RSI M0 离线工程预检报告

日期：2026-10-26
状态：离线工程子门禁通过；M0 集成门禁未通过，M1/M2 不启动。

## 结论

本地 M0 预检通过了 Bundle 内容寻址、三个候选组件的隔离执行、失败尝试计费、成功任务幂等恢复、身份冲突拒绝和预算核对。14 项 RSI 定向单测通过，`git diff --check` 通过。正式 CLI 首次执行和对既有结果的再次执行均通过；第二次执行复用了已完成记录，没有再次调用任务服务。

这只是离线工程子门禁，不是完整 M0 通过，更不代表 HarnessPro RSI 能力或 Wan 研究收益。当前 seed 组件尚未证明等价于现有 Wan Block 研究 loop，ResearchTaskService 也仍是接口和离线 fixture，没有接入真实 Block worker。因此按照已批准的阶段门禁，M1 历史轨迹回放、模型调用以及 M2 GPU 对照都保持停止。

## 验收证据

- 基线 Bundle：`5d3834a8b6cf5c1879aa68345938de2ac38547668af3eeb2dbfcd56707e09930`，只含 `context.py`、`retrieval.py`、`strategy.py` 三个声明组件。
- 种子源码 SHA256：
  - `context.py`: `40def75b8025bae75e9a833f178c7e01a6d3b26eae432968cd7ecee8a6db539d`
  - `retrieval.py`: `1de3980521c85cced263d4079f19406afb5d54058cccf19ab082fecb0bd62508`
  - `strategy.py`: `ac43a378704c05744e045d5c6b156280ad23323d5fe2cfd715cde254f4b6ee3f`
- 候选组件均经 `AlgorithmSandbox` 子进程执行；测试覆盖文件访问拒绝。
- 任务 fixture 的第一次尝试故意失败，第二次完成，第三次精确重放；任务服务总计 collect 2 次、evaluate 1 次。预算 ledger 记了 2 次，重放没有增加调用或预算。
- 不同 seed 复用同一 `job_id` 会被拒绝，资源失败的分类会原样进入 trace。
- 未使用网络、模型 endpoint、Wan pipeline、GPU 或冻结划分；缺少的 token、GPU 时间、optimizer updates 和美元成本均保持 unavailable。

## 测试

命令：

```text
python3 -m pytest tests/unit/test_rsi_bundle.py tests/unit/test_rsi_trace.py tests/unit/test_rsi_budget.py tests/unit/test_rsi_runner.py tests/unit/test_rsi_m0_preflight.py -q
```

结果：`14 passed`。`git diff --check` 无输出。

正式预检输出：`var/rsi-m0/preflight.json`（本机运行产物，不纳入源码变更）。首次运行创建 fixture 证据；再次运行读取并复用相同完成记录，未重复任务服务调用。

## 尚未通过的 M0 条件

1. 将冻结 seed 接到现有 Wan Block loop，证明上下文、检索、实验选择的输入输出契约真实可用，并提供可重放基线。
2. ResearchTaskService 适配现有 Block worker，具备确定性的任务/产物身份、worker 原始结果、失败类型、更新步数与 checkpoint 重载证据；需先检查并复用已存在且指纹匹配的验证，不得仅凭 fake fixture 宣称完成。
3. 增加 Analyzer/Planner/Editor 候选补丁的生命周期和外置边界验证。当前组件 runner 只能运行组件，没有候选 patch、父子谱系、独立 evaluator 或选择回滚闭环。
4. 完成 M0 源码/实验原件的状态保全审计。现有 worktree 原本有大量脏文件；本报告未声称完成逐路径前后状态校验。

这些属于真实集成/证据缺口，不是此次 13 项 CPU 单测失败。不能通过放宽验收条件把 M0 标记为完整通过。

## 下一步

只推进上述 M0 集成缺口：先盘点可复用的 Wan Block worker 结果和指纹，再实现只调用已有固定任务服务的适配器和 seed 等价回放；在不训练、不触碰冻结集的前提下完成补充离线测试与审计。全部 M0 条件通过后，另行固定 M1 的任务清单、轨迹划分、模型/token 预算与外置评估契约，之后才开始历史回放。
