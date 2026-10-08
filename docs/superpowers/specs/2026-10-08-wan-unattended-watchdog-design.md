# Wan Campaign Unattended Watchdog Design

## Goal

让真实 Wan vLLM campaign 在 Codex/SSH 断开后继续运行，并在训练进程异常退出且没有产生最终 campaign 结果时自动重启；恢复必须沿用现有 campaign state、round checkpoint、fingerprint 和 ExperienceStore。

## Reuse boundary

不新增训练状态或实验记忆格式。现有 `wan_vllm_campaign.py` 继续拥有 round、proposal、evaluation、decision、checkpoint resume 和 ExperienceStore；watchdog 只负责远端进程生命周期。现有 `remote-campaign-service.sh` 的 PID、pause、stop、单实例和 controller 生命周期约定继续作为服务入口，增加 `REMOTE_CAMPAIGN_KIND=wan_vllm` 分支，默认 overnight 行为不变。

## Runtime flow

1. `remote-campaign-service.sh start` 在 Wan 模式下启动 `wan-vllm-campaign-watchdog.sh`，写入 service PID、日志和锁文件。
2. Watchdog 使用受信 Python、配置路径和 output root 启动现有 `wan_vllm_campaign.py`，不接受 vLLM 或 campaign artifact 提供的命令。
3. 子进程正常完成并产生 `campaign-result.json` 时 watchdog 退出，不重启。
4. 子进程异常退出且没有最终结果时，watchdog 按有限重试次数和递增退避重启同一个配置/实验根目录；campaign 自身负责从 durable state/checkpoint 恢复。
5. operator stop/pause 创建 stop marker，watchdog 不再重启，并向子进程发送 TERM；不会匹配或终止旧 smoke、其他 campaign 或 Controller。
6. 最终失败结果、达到重试上限、配置错误或 proposal/evaluation 的业务拒绝都停止自动重试，避免无限循环和重复实验。

## Safety and recovery

- PID 文件只接受命令行同时包含受信 Wan watchdog/campaign 路径的进程。
- Watchdog 使用独立 lock 防止两个 watchdog 共享一个 campaign root。
- 重启不删除任何 round artifact、checkpoint、state 或 ExperienceStore。
- 旧的 `w8-dtype-smoke` 和其他 campaign root 不在管理范围内。
- `offline_simulation=false`、parent lineage 和 30 秒 evaluator gate 仍由现有 campaign 负责。

## Verification

- shell syntax and service dispatch tests prove Wan mode does not alter overnight mode;
- watchdog test uses a fake runner that exits once without a result, then writes a result, proving bounded restart;
- stop-marker test proves no restart after operator stop;
- remote smoke verifies the watchdog PID is detached from the SSH shell, the child command uses the current commit, and the campaign root receives durable artifacts.

## Non-goals

- 不修改模型架构、训练算法、vLLM 策略空间或评价指标；
- 不自动重启已写出最终失败结果的业务 round；
- 不停止当前旧 smoke 或未经明确指定的远程任务；
- 不依赖 Codex 轮询维持实验。
