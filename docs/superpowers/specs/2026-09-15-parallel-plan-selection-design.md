# Parallel Controller Plan Selection Design

**Date:** 2026-09-15

## Goal

让真实 vLLM controller 在每次决策时批量生成 4 个候选 `ExperimentPlan`，再选择一个安全候选执行，以提高 LLM GPU 的批处理利用率和决策多样性，同时保持同一训练状态只有一个 plan 被执行。

## Chosen approach

- 保持一个 vLLM 服务和一个模型副本；通过 OpenAI-compatible `n=4` 请求生成候选。
- 候选 plan 使用较短的 completion 上限，避免 4 条序列长期占用 KV cache。
- 先解析候选 JSON；只有可解析的 `ExperimentPlan` 才进入选择阶段。
- 使用同一 vLLM 服务发起一个短 selector 请求，返回候选索引；selector 失败时回退到第一个可解析候选。
- 最终仍由现有 campaign validation、resource validation 和 evaluator 决定是否可以执行；候选生成不会绕过任何安全门槛。
- 训练、benchmark 和 checkpoint 写入仍然串行；只并行候选生成和候选评估。

## Interfaces and configuration

`OpenAICompatibleController` 增加候选数量、候选温度、候选 completion 上限和 selector completion 上限。`configs/controller.yaml` 默认使用 4 个候选；远端 vLLM launcher 默认 `MAX_NUM_SEQS=4`，并继续允许通过环境变量降回 1。

Provider 暴露最近一次候选数量、候选 request id、selector request id 和选中索引，campaign 将这些字段写入现有 controller trace/event，但只把最终 plan 交给执行器。

## Failure handling

- vLLM 返回少于 4 个 choice：使用实际返回的可解析候选，不人为补 plan。
- 单个 choice JSON 或 schema 解析失败：丢弃该 choice，继续处理其他 choice。
- 所有候选都不可解析：沿用现有 Controller error path。
- selector HTTP/JSON/索引失败：选择第一个可解析候选，并记录 fallback 原因。
- 最终选中的候选仍必须通过现有 harness validation；失败时沿用现有 rejected/replan 逻辑。

## Acceptance

- 单请求可以返回并解析多个 plan choice。
- selector 只返回合法候选索引；非法或不可用时安全回退。
- 旧的单 choice provider response 仍然可用，现有测试 controller 不需要实现 `n`。
- 真实运行配置不会启动多个 vLLM 进程，也不会并行执行多个训练 plan。
