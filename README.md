# Harness4H3

EvoGen-RSI 是长期研究框架；Harness4H3 是其 Phase I reference implementation。

```text
Fixed Strong LLM Controller
        +
H3-specific Model Optimization Harness
        +
MiniMax H3 / H3-derived Student
        +
Real Quality + Hardware Evaluation
        ↓
Autonomous Model Optimization
```

Phase I 的 Controller LLM 权重固定，Harness 只允许它产生结构化 `ExperimentPlan` 并选择已注册的模型级 Operator。模型修改必须产生不可变的 `M0000`、`M0001`… candidate，经独立质量/硬件 evaluator 验证后进入 Pareto archive。Phase I 不训练 Controller LLM、不演化 Harness、不做 kernel/compiler search，也不允许 Controller 执行 shell 或修改源码、evaluator、benchmark 与目标约束。

当前正式交付覆盖 M0–M5.5，并已实现 M6 runtime-memory validation：完全离线 Fake H3 closed loop、结构化 Ollama/OpenAI Responses 控制器、safetensors/GGUF H3 Inspector、受限本地进程执行器、固定部署变体的真实 H3 quantize operator，以及带黑帧诊断/Operator Attribution 的受控真实 benchmark。M5.5 已按独立 dev、held-out 和 multi-seed 分组完成候选复核；M6 从 M0001 NVFP4 派生 runtime branch，以峰值显存 max（而非均值）作为 16GB hard gate。真实 M6 acceptance 仍需在 RTX 5080 在线时执行，未把 TargetProfile 峰值显存约束预先改写为已满足。

## 离线闭环

安装 Python 3.9+ 环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[test]'
```

验证固定 TargetProfile：

```bash
.venv/bin/python -m harness4h3 validate \
  --target configs/targets/mobile_example.yaml --json
```

执行 Fake H3 优化：

```bash
.venv/bin/python -m harness4h3 optimize \
  --target configs/targets/mobile_example.yaml \
  --session-dir var/evogen \
  --session-id mobile_h3_fake_001 \
  --json
```

确定性结果为：

```text
M0000 (quality=.90, latency=60s, memory=12GB)
  → quantize
M0001 (quality=.891, latency=48s, memory=7.8GB)
  → step_distill
M0002 (quality=.85536, latency=26.4s, memory=5.46GB)
  → target_satisfied
```

Fake StepDistill 除规格给出的 latency `×0.55`、quality `×0.96` 外，还把工作内存设为 `×0.70`；这是为了解决规格示例中 `12 × 0.65 > 6` 与要求两步达到 6GB 目标之间的数值矛盾。该效果仅属于 fake simulator，不代表真实算法 claim。

查看模型分支、Pareto front 与 append-only experiment records：

```bash
.venv/bin/python -m harness4h3 lineage --session-dir var/evogen --json
.venv/bin/python -m harness4h3 pareto --session-dir var/evogen --json
.venv/bin/python -m harness4h3 replay --session-dir var/evogen --json
```

Inspect a local H3 checkpoint without loading its weights:

```bash
.venv/bin/python -m harness4h3 inspect \
  --checkpoint 'D:\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_nvfp4.safetensors' --json
```

The Ollama controller uses `/api/chat` with `stream=false`, `think=false`, temperature 0, and a strict ExperimentPlan schema. The OpenAI Responses controller uses `/v1/responses`, strict `text.format` JSON Schema, and `store=false`; API keys are read only from the configured environment variable.

Run the independent real H3 benchmark on a host that can access ComfyUI and the checkpoint:

```bash
.venv/bin/python -m harness4h3 benchmark \
  --checkpoint 'D:\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_nvfp4.safetensors' \
  --sampling-steps 4 --target configs/targets/rtx5080_example.yaml \
  --baseline-quality 0.991137 \
  --base-url http://100.88.143.10:8188 --reset-backend-before-run \
  --primary-intervention quantization \
  --controlled-variable sampling_steps --controlled-variable seed \
  --result var/benchmark/nvfp4.json --json
```

对于 Windows 远端路径，benchmark runner 只把路径的文件名写入 API workflow；文件实际由远端 ComfyUI 加载。`--reset-backend-before-run` 调用 ComfyUI `/free`，用于隔离模型切换时的缓存状态。

每个 session 只在初始化时读取一次 TargetProfile。`session.json` 原子保存 current model、预算和失败计数；相同 session 可在中断后恢复。每个实验目录保存 controller request/response、plan、validated plan、operator result 与 evaluation。

## Phase I 安全边界

验证顺序固定为：

```text
SchemaValidator
→ PolicyValidator
→ BudgetValidator
→ OperatorValidator
→ Executor
```

默认 fake registry 只开放 `inspect`、`quantize`、`step_distill`、`rollback`。真实 `PrebuiltQuantizeOperator` 的候选路径来自固定部署配置，LLM 只能选择 bits；`CreateStudent`、`Distill`、`Prune`、任意 shell、CUDA/Triton 和源码修改均不可执行。真实候选在 held-out benchmark 前会标记 `metrics_stale`。

Evaluator 是独立权威。Pareto 先比较 hard-constraint feasibility，再比较 quality、latency、memory、model size 与 energy；不把单一 scalar reward 当作核心排序。

## 测试

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q harness4h3 tests
```

默认测试不访问网络，不需要 GPU、ComfyUI 或 H3 checkpoint。闭环测试覆盖成功、非法 operator、非法 LLM plan、operator failure、training OOM、critical quality regression、Pareto branching、budget stop、重复失败、crash recovery 和 target reached。

## 旧版兼容

旧的冻结 ComfyUI workflow/prompt evolution 已移到 `harness4h3/legacy/workflow_evolution.py`；旧导入路径仍可用，但 EvoGen Phase I 主循环不依赖它。ComfyUI HTTP adapter 位于 `harness4h3/backends/comfyui.py`，只负责 artifact generation，不再代表 H3 model state。

旧的 `run`、`evaluate`、`evolve`、`validate-config` 与 `legacy-lineage` 命令暂时保留用于历史实验重放。新研究主线使用 `validate`、`inspect`、`optimize`、`lineage`、`pareto` 与 `replay`。

## M5 受控验收

远端 RTX 5080 的真实运行记录见 `docs/real-experiments/2026-09-09-windows-rtx5080.md`。在相同提示、种子、分辨率、scheduler、CFG、VAE、文本编码器和 LoRA、且每次切换前清理 ComfyUI 模型缓存的 20-step 对照中，NVFP4 child 相对 INT8 parent 达到 model size `-40.26%`、latency `-56.42%`，质量分数从 `0.983151` 到 `0.991137`（无回归），视频可解码且黑帧率为 `0`，因此通过 M5 acceptance gate。该记录同时保留目标 profile 的峰值显存约束结果；后续仍需在 dev/held-out 任务上复核。

## M5.5 Accepted Candidate Validation

`experiments/m5_validation.py` 提供固定 parent/child 的可复现实验入口：sanity 默认重复两次并交替 parent/child 顺序，每次切换前调用 ComfyUI `/free`；随后运行 dev、held-out 和多 seed 复核，并将均值、中位数、最小/最大值、标准差和 95% CI 写入 `var/m5-controlled/m5.5-validation.json`。本轮四个分组全部通过：sanity child 质量均值 `0.991137`、延迟均值 `90.281s`，dev/held-out/multi-seed 质量分别为 `0.988818`、`0.961489`、`0.991958`，黑帧率均为 `0`。该阶段只验证量化候选的可复现性与泛化，不把 `TargetProfile` 的峰值显存约束改写成已满足，也不提前实现 M6 runtime-memory operator。

```bash
PYTHONPATH=. .venv/bin/python experiments/m5_validation.py --sanity-repetitions 2
```

首条结构化 Design Gene 记录在 `docs/experience/design-gene-h3-nvfp4.json`，用于后续经验检索；它明确记录了缓存污染导致的黑帧诊断和 16GB 峰值显存剩余限制。

## M6 Cross-Layer Runtime Memory

`experiments/m6_runtime_memory.py` 从已验证的 `M0001` NVFP4 parent 分支出 `runtime_offload`、`vae_tiling` 和 `inference_chunking` 三个低风险 runtime operator。Controller 只接收这些 runtime operator、TargetProfile、M5.5 状态/指标和只读 Design Gene，并自主选择一个主要 intervention；权重、TargetProfile、evaluator 和 Harness Evolution 均不被修改。每个分支都执行独立 evaluator、每次运行前调用 `/free`、记录 Operator Attribution，并把结果同时写入原子 JSON evidence 与 append-only trajectory JSONL。

```bash
PYTHONPATH=. .venv/bin/python experiments/m6_runtime_memory.py \
  --controller ollama --branches controller \
  --base-url http://100.88.143.10:8188 \
  --controller-url http://100.88.143.10:11434
```

M6 只有在 dev 与 held-out 两组同时满足生成/解码有效、黑帧率为 `0`、质量下降不超过 `0.05`、保留 M5.5 的模型体积与延迟收益，且 `peak_vram_max_gb <= 16.0` 时才返回成功。缺失节点能力、远端不可达和任一峰值超限都会作为显式失败记录保留。
