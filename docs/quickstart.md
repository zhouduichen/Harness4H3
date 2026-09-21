# Quickstart

This guide separates offline software verification from commands that require
a real H3 host. Run commands from the repository root.

## 1. Install

Python 3.9 or newer is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

On Windows, use `.venv\Scripts\python.exe` instead of
`.venv/bin/python`.

## 2. Verify the repository offline

These commands require no GPU, model checkpoint, Controller service, or
ComfyUI service:

```bash
.venv/bin/python -m harness4h3 --config configs/default.yaml validate-config --json
.venv/bin/python -m harness4h3 validate \
  --target configs/targets/mobile_example.yaml --json
.venv/bin/python -m pytest -q
```

Run a simulated closed-loop protocol check:

```bash
.venv/bin/python -m harness4h3 optimize \
  --target configs/targets/mobile_example.yaml \
  --session-dir var/offline-smoke \
  --session-id offline-smoke --json
```

This uses `FakeH3Model`, fake operators, and fake evaluators. It verifies
control flow and persistence, not real H3 optimization. The result contains
separate model lineage (`Mxxxx`) and system/runtime lineage (`Sxxxx`); the
Pareto front stores evaluated model/system search points.

## 3. Inspect a real checkpoint without loading weights

Requires a local `.safetensors` or `.gguf` file, but no GPU:

```bash
.venv/bin/python -m harness4h3 inspect \
  --checkpoint /path/to/minimax-h3.safetensors \
  --model-id M0000 --sha256 --json
```

The inspector reads metadata/header information and optionally streams the
file for SHA-256. It does not run inference.

## 4. Preflight a real execution device

Run this on the machine that hosts the paths and services described by the
profile:

```bash
.venv/bin/python tools/device_preflight.py \
  --profile configs/devices/rtx5080-laptop.yaml \
  --operator benchmark \
  --result var/preflight/benchmark.json --json
```

`ready` means the declared capability, required local paths, and required
health endpoints passed. `blocked` means the experiment must not start. To
validate profile structure and local paths without network probes, add
`--skip-services`. The `--result` option persists the complete decision even
when it is blocked.

## 5. Run a real benchmark

Requires a real H3 checkpoint and a reachable ComfyUI API configured with the
workflow nodes in `examples/workflow_api.json`:

```bash
.venv/bin/python -m harness4h3 benchmark \
  --checkpoint /path/to/minimax-h3.safetensors \
  --target configs/targets/rtx5080_example.yaml \
  --base-url http://127.0.0.1:8188 \
  --split sanity --reset-backend-before-run \
  --result var/benchmark/sanity.json --json
```

Use `dev` and `heldout` only after sanity generation succeeds. Never edit an
EvaluationResult to make a failed constraint pass.

## Research entry points

The primary installed interface is `harness4h3`. Phase-specific programs are
kept as explicit research entry points:

```bash
.venv/bin/python -m research.experiments.m5_validation --help
.venv/bin/python -m research.experiments.m6_runtime_memory --help
.venv/bin/python -m research.experiments.m6_runtime_recipe --help
```

See the [research index](../research/README.md) before interpreting their
outputs as scientific evidence. M6 is an optional runtime-memory study; it is
not required for moving the Harness to another device or establishing a real
H3 baseline.

## Remote L40×4 H3 campaign

The configured SSH alias is `Jiayu-intern`. Remote checkpoints and trainer
results remain on `/data/models/MiniMax-H3`; only JSON metadata and generated
benchmark videos are copied to the local output directory.

The remote campaign defaults to the configured OpenAI-compatible Qwen
Controller (`qwen3.5-controller` on remote port `8000`). Before the first
request, verify that the service is actually listening:

```bash
ssh Jiayu-intern 'curl -fsS http://127.0.0.1:8000/v1/models'
```

To access the same server-side LLM from your own terminal, start the secure
local tunnel in one terminal:

```bash
./tools/controller-llm-tunnel.sh
```

It maps the remote `127.0.0.1:8000` to local `127.0.0.1:18000`. In another
terminal, use the OpenAI-compatible API:

```bash
curl -fsS http://127.0.0.1:18000/v1/models
curl -fsS http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-controller","messages":[{"role":"user","content":"用一句话说明当前实验下一步应检查什么。"}],"temperature":0.2,"max_tokens":128}'
```

For a persistent conversation directly on the remote server, run:

```bash
ssh Jiayu-intern
/home/intern/miniconda3/envs/comfy/bin/python \
  ~/huangjiahao/Harness4H3-rsi/tools/controller-llm-chat.py
```

The chat keeps the conversation history in memory. Use `/clear` to reset it
and `/quit` to exit; add `--thinking` only when reasoning traces are wanted.

To let the server-side LLM autonomously run repeated optimization rounds,
launch the campaign on the server itself. `--local-resources` uses the same
trusted commands and loopback services directly; it does not open a nested SSH
session or require Codex to remain connected:

```bash
cd ~/huangjiahao/Harness4H3-rsi
/home/intern/miniconda3/envs/comfy/bin/python tools/run_overnight_controller.py \
  --local-resources --controller vllm \
  --config configs/remote-l40-h3-rsi-overnight.yaml \
  --output var/remote-h3-controller-20260914 \
  --max-iterations 512 --resource-poll-interval-s 60
```

Each round is LLM plan → trusted prune/quantize/distill worker → independent
held-out benchmark → acceptance decision → next-round experience. While a
trusted worker is running, a cloned server-side LLM prepares and validates the
next plan; it is reused at the next safe boundary only if its parent and
evidence state are still current. Distributed training requests elastically
use 2–4 cards that pass the live memory waterline; single-process
structural/quantization workers use only what they need. The configured
`max_retained_checkpoints` cap keeps only a bounded active/rollback/Pareto set
of completed weights; full JSON evidence, recipes, and trajectory records stay
available to the next LLM context. Reclaimed weights are removed only from the
exact campaign `continuous/Mxxxx/` location. The runner also takes a
single-flight lock in the output root, so an accidental second launch cannot
mutate the same campaign concurrently.

The vLLM service does not use a permanently assigned GPU index. When ComfyUI
owns a benchmark lease, its launcher caps vLLM at TP=1 so at least two other
cards remain available for a distributed successor; outside evaluation it can
select TP=2/4 from the live waterline. The worker scheduler receives the
remaining cards and can expand an elastic request to every safe card; a shared
worker lease prevents the Controller launcher from selecting those cards during
an in-flight launch or restart.
The detached entry point also records an unexpected top-level process failure
and retries with bounded backoff, rebuilding its campaign object from the
persisted state; normal remote/evaluator failures are retried inside the loop.

For a detached SSH-hosted process with explicit status and stop controls, use
the configured SSH alias `Jiayu-intern`. It includes the jump host and port
`30902` from `~/.ssh/config`; using the backend IP directly can fail because
the server is only reachable through that SSH route:

```bash
ssh Jiayu-intern
cd /home/intern/huangjiahao/Harness4H3-rsi
REMOTE_CAMPAIGN_PYTHON=/home/intern/miniconda3/envs/comfy/bin/python \
  tools/remote-campaign-service.sh start
tools/remote-campaign-service.sh status
tools/remote-campaign-service.sh stop
```

To prevent both a handoff watcher and a manual operator from starting new
work, use the explicit pause gate:

```bash
tools/remote-campaign-service.sh pause
tools/remote-campaign-service.sh status  # state=paused, pid=none
tools/remote-campaign-service.sh resume # only removes the gate; it does not start a campaign
```

While the pause marker exists, `start` returns a non-zero status and the idle
autostart watcher only waits. `pause` requests a graceful stop when this
project owns a live campaign; it never kills unrelated GPU processes.

The service stores its PID, log, pipeline cursor, recipes, and bounded
checkpoint state below the campaign output root. Override
`REMOTE_CAMPAIGN_REPO_ROOT`, `REMOTE_CAMPAIGN_CONFIG`, or
`REMOTE_CAMPAIGN_OUTPUT` when using another remote checkout. `stop` writes a
one-shot graceful-stop marker and returns immediately; the campaign finishes
the current iteration and any speculative child before releasing its lock.
During a full four-card worker the service waits for the next validated plan
before taking the worker lease; during a smaller elastic worker the Controller
launcher may reuse the remaining card(s). ComfyUI is stopped only when it was
launched and owned by this campaign, after its queue/cache release succeeds.

From the local checkout, stage only the trusted runtime files before starting
or handing off the remote service:

```bash
REMOTE_PIPELINE_SOURCE_ROOT="$PWD" \
  tools/sync-remote-pipeline.sh
```

The sync command uses port `30902`, does not copy checkpoints or the local
virtual environment, and does not activate or start any remote process. After
the staging copy succeeds, launch the remote handoff watcher shown below.

When migrating an already-running legacy campaign, the staged handoff watcher
can be launched once from the remote host; it waits on the existing single
flight lock and then activates v2 automatically:

```bash
nohup env \
  REMOTE_CAMPAIGN_REPO_ROOT=/home/intern/huangjiahao/Harness4H3-rsi \
  REMOTE_PIPELINE_STAGE_ROOT=/home/intern/huangjiahao/Harness4H3-rsi/work/remote-pipeline-v2-20260918 \
  REMOTE_CAMPAIGN_OUTPUT=/home/intern/huangjiahao/Harness4H3-rsi/var/remote-h3-controller-20260914 \
  bash /home/intern/huangjiahao/Harness4H3-rsi/work/remote-pipeline-v2-20260918/tools/remote-pipeline-handoff.sh \
  >/dev/null 2>&1 < /dev/null &
```

Its handoff log is in the staging directory. It refuses activation while the
legacy supervisor or lock is live, so it can be started before the current
round finishes.

The local port can be changed without touching the remote service, for
example `CONTROLLER_LOCAL_PORT=18001 ./tools/controller-llm-tunnel.sh`.

The remote loop follows the useful control-plane split from
[A-Evolve-Training](https://arxiv.org/abs/2606.20657): the trusted worker and
evaluation recipe are an immutable substrate, each candidate is isolated by
lineage/checkpoint path, and only a bounded round policy plus a discovery
digest crosses into the next Controller context. Full JSONL experience,
observations, and audit events remain on disk as evidence; they are not copied
verbatim into every LLM prompt.

If the endpoint is unavailable, the campaign records `controller_unavailable`
and waits/retries at the next loop boundary; it never falls back to the
RuleBased controller. For this overnight entry point, use `--controller
rulebased` only for an explicit offline test.

First perform a read-only import:

```bash
.venv/bin/python -m harness4h3 import-experience \
  --remote-config configs/remote-l40-h3.yaml \
  --output var/remote-h3/experience.jsonl --json
```

Run one controlled sanity benchmark and resume it later with the same output
root:

```bash
.venv/bin/python -m harness4h3 remote-campaign \
  --remote-config configs/remote-l40-h3.yaml \
  --target configs/targets/l40x4_h3_example.yaml \
  --output-root var/remote-h3 --max-experiments 1 --split sanity --json
```

For the full configured chain:

```bash
.venv/bin/python -m harness4h3 remote-campaign \
  --remote-config configs/remote-l40-h3.yaml \
  --target configs/targets/l40x4_h3_example.yaml \
  --output-root var/remote-h3 --max-experiments 4 --json
```

`training_only_unvalidated` cannot become active; `evaluated_candidate` means
measured but not promoted; `accepted` requires training, quality, hard-gate,
and Q/L/M/E checks. The default evaluator is `structural_proxy`, so its score
does not support a semantic video-quality claim.

The remote benchmark config also controls the campaign-level ComfyUI cache:

- `idle_release` unloads ComfyUI after a completed evaluation phase and returns
  GPU0 to the scheduler when `/free` and the fresh `nvidia-smi` memory-waterline
  check both succeed.
- `warm_cache` keeps GPU0 reserved for warm-cache latency comparisons.
- `cold_cache` releases at controlled phase boundaries so the next benchmark
  reloads the model.

`idle_release` is a resource policy, not a global CPU-only mode. `quantize` and
`prune_blocks` are CPU/I/O operators because the trusted workers adopt or rewrite
checkpoint files; GPU training operators still request 2--4 elastic GPUs. After
`/free` succeeds, ComfyUI keeps only its small CUDA context and GPU0 is
shareable with the Controller or a worker; it does not reserve a standalone
model card while idle. Plan generation is batch-sampled (`n=4`) on the resident
vLLM instance, then a selector chooses one candidate for execution. While a
trusted training worker is running, the campaign also generates one
speculative next plan on a cloned Controller provider. It is persisted only
after structured validation and is reused at the next boundary when the parent
lineage and human directives are unchanged; otherwise it is discarded and
replanned from fresh evidence.

Release is fail-closed. If the queue is active, the API fails, or the memory
waterline is not met, GPU0 remains reserved and the event stream records
`comfyui_cache_release_failed`.

## Real MiniMax-H3 A1-T0 gate

The authentic path is fail-closed. Run it on the L40×4 host after the parent
checkpoint, ComfyUI, Controller, and benchmark service are available:

```bash
.venv/bin/python tools/run_real_h3_gate.py \
  --parent-checkpoint /data/models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --worker-config configs/a1-worker.l40x4-distill4.json \
  --config configs/default.yaml \
  --target configs/targets/l40x4_h3_example.yaml \
  --tasks examples/tasks.yaml \
  --base-url http://127.0.0.1:8188 \
  --output-root var/a1-real-gate \
  --baseline-quality <measured-quality> \
  --baseline-model-size-gb <measured-size-gb> \
  --baseline-latency-s <measured-latency-s> \
  --baseline-peak-memory-gb <measured-vram-gb>
```

It records `real_h3_gate.json` and exits non-zero for missing CUDA, paths,
services, or evidence. It never substitutes TinyH3, fake operators, or
simulated metrics. Passing requires authentic training evidence, independent
benchmark evaluation, and the next Controller plan in the same trajectory.

To watch the Controller loop without opening the campaign report:

```bash
.venv/bin/python -m harness4h3 controller-status \
  --output-root var/remote-h3 --follow
```

Remote training resources are Controller-authorized. To allow a distributed
plan to wait for and then use whichever 2–4 GPUs become available, run with an
elastic resource request and a retry interval:

```bash
.venv/bin/python -m harness4h3 remote-campaign \
  --remote-config configs/remote-l40-h3.yaml \
  --max-iterations 12 \
  --resource-poll-interval-s 30 \
  --resume --json
```

The scheduler never stops unrelated remote processes. A waiting plan remains
in `campaign_state.json` and can be resumed by a later invocation.
