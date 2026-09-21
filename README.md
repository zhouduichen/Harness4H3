# Harness4H3

Harness4H3 is a device-aware research harness for reproducible MiniMax H3
optimization experiments. It studies one bounded question:

> Can a fixed controller select model-level or runtime-level interventions for
> a declared device, while an independent evaluator retains only candidates
> supported by measured quality and hardware evidence?

The repository provides the experiment protocol, constrained operator
execution, model/system lineage, independent evaluation, Pareto archive, and
append-only trajectory. It also includes a shared immutable-base campaign
control plane with bounded multi-candidate review, fail-closed capability
snapshots, hard/soft gates, structured failure attribution, and an auditable
decision trace. See the [campaign control-plane guide](docs/campaign-control-plane.md).
It includes a real PyTorch TinyH3 reference trainer
for validating contracts and a fail-closed Real MiniMax-H3 adapter/worker
path. The real path still requires the declared GPU host, ComfyUI service, and
an authentic A1-T0 run before any MiniMax-H3 optimization claim is made.

## Capability status

| Capability | Evidence level | Current status |
|---|---|---|
| Offline optimization loop | Simulated and tested | Available |
| H3 checkpoint inspection | Real file metadata | Available |
| H3 generation through ComfyUI | Measured on RTX 5080 Laptop | Available |
| Quality/hardware benchmark | Measured on RTX 5080 Laptop | Available |
| Runtime-memory interventions | Measured, including rejected results | Experimental |
| TinyH3 training loop | Real PyTorch weights, gradients, resume, and lineage | Available for CPU contract testing |
| Recovery / progressive distillation | Real TinyH3 algorithm tests | Reference implementation |
| DMD2 multi-role trainer | Real TinyH3 gradients, alternating updates, EMA, and resume | Reference skeleton |
| Real MiniMax-H3 adapter | Source-grounded execution path | Implemented; host capability gate is enforced |
| Real MiniMax-H3 recovery / step distillation | Remote FSDP worker evidence | Measured on the remote L40 host; benchmark-gated |
| Real MiniMax-H3 pruning / quantization | Trusted remote workers | Implemented; every child remains benchmark-gated |

## Current gate

The CPU TinyH3 gate remains a contract test suite, not the current research
milestone. The active milestone is **Phase I — real H3 closed-loop validation**:
the remote LLM must repeatedly choose a trusted operator, produce real
forward/backward or structural checkpoint evidence, receive an independent
ComfyUI measurement, and continue from the persisted Model/System lineage.
Passing this gate is evidence that the loop runs; it is not by itself a claim
that the target profile has been improved.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q harness4h3 h3_training tools research
.venv/bin/python -m research.experiments.tiny_real_closed_loop \
  --output-root var/phase0-final-closed-loop
```

See the [Phase 0 validation evidence](research/evidence/phase0-validation-2026-09-13.md)
and the [validation plan](docs/validation-plan.md).

The current execution priority is the detached remote campaign: a local
Qwen/vLLM Controller selects the next trusted model or runtime intervention,
the scheduler allocates whatever 2–4 GPUs are safe at that boundary, and
ComfyUI is leased only for measurement before its cache is released. The
campaign records failures as experience and retries from the last valid parent;
it never falls back to RuleBased control. M6 runtime-memory work remains an
optional branch, while real H3 training and measured target feasibility are the
primary open questions.

The resident Controller normally uses the smallest feasible vLLM tensor-parallel
group (TP=1, one dynamically selected card), leaving the other cards to the
worker. It is not pinned to a GPU index: a restart re-scans live free memory and
can select another card or a larger TP group. A short-lived worker GPU lease is
shared with the launcher so a Controller restart cannot race an elastic worker.

Engineering completeness is not presented as a new optimization algorithm.
The research contribution under test is the evidence-grounded, device-aware
closed-loop protocol. See the [research index](research/README.md) for claims,
variables, experiments, negative results, and limitations.

## System under study

```text
DeviceProfile + TargetProfile + Model/System state + Budget
                         ↓
              fixed/manual Controller
                         ↓
                  ExperimentPlan / CandidateBatch
                         ↓
 immutable base → schema → review → deterministic validation
                         ↓
            ModelCandidate or StudentCandidate
                         ↓
      benchmark + canonical EvaluationRecord
                         ↓
  accept/reject/fail → archive + trajectory → next plan
```

The Controller may be Qwen through Ollama, an OpenAI-compatible provider, the
deterministic offline controller, or a manually supplied plan. For controlled
comparisons, the provider, model, prompt/schema, budget, target, tasks, and
evaluator must remain fixed.

## Quick start

Requires Python 3.9 or newer.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m harness4h3 --config configs/default.yaml validate-config --json
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/unit/test_campaign_*.py tests/unit/test_campaign_adapters.py tests/integration/test_campaign_control_plane.py
```

Windows uses `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

Run the fully offline protocol smoke test:

```bash
.venv/bin/python -m harness4h3 optimize \
  --target configs/targets/mobile_example.yaml \
  --session-dir var/offline-smoke --session-id offline-smoke --json
```

Install the optional training dependencies and run the real-weight TinyH3
reference loop on CPU:

```bash
.venv/bin/python -m pip install -e '.[test,training]'
.venv/bin/python -m research.experiments.tiny_real_closed_loop \
  --output-root var/tiny-real-closed-loop
```

This creates and independently evaluates `M0000 -> M0001 -> M0002` using
real tensors, backward passes, optimizer state, child hashes, and checkpoint
reloads. TinyH3 is deliberately small and is not MiniMax-H3 evidence.

Before using a real device, run the declarative capability preflight on that
device:

```bash
.venv/bin/python tools/device_preflight.py \
  --profile configs/devices/rtx5080-laptop.yaml \
  --operator benchmark \
  --result var/preflight/benchmark.json --json
```

The checked-in RTX 5080 profile intentionally reports
`recovery_finetune`, `prune`, and `distill` as disabled.

## Real benchmark

This command requires a host that can access the H3 checkpoint and a running
ComfyUI API. It does not train or modify the checkpoint.

```bash
.venv/bin/python -m harness4h3 benchmark \
  --checkpoint 'D:\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_nvfp4.safetensors' \
  --target configs/targets/rtx5080_example.yaml \
  --sampling-steps 4 --split sanity \
  --base-url http://127.0.0.1:8188 \
  --reset-backend-before-run \
  --result var/benchmark/nvfp4-sanity.json --json
```

Failures, rejected candidates, and unmet constraints are research outcomes and
must not be rewritten as accepted results.

## Real MiniMax-H3 closed-loop gate

Run this only on the host that owns the checkpoint, ComfyUI installation, and
four CUDA devices. The gate writes a JSON decision even when blocked; it never
falls back to TinyH3 or fake metrics:

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

The gate is passed only when the real worker produces `M0001` evidence and a
second Controller plan (`exp_0002`) is recorded after independent benchmark
evaluation. A local CPU run is expected to be blocked by the CUDA and host
preflight checks.

## Repository map

```text
harness4h3/          reusable research-preview harness implementation
tools/               device-side worker adapter and read-only preflight
configs/devices/     measured device facts and capability declarations
configs/targets/     immutable optimization objectives and hard constraints
research/experiments reproducible A0/A1/M5/M6 experiment programs
research/evidence/   real measurements, reconnaissance, and Design Genes
research/history/    completed designs, plans, and superseded research notes
docs/                active architecture, protocol, and operator guides
tests/               offline unit and integration verification
var/                 ignored local run artifacts
```

## Documentation

- [Quickstart and command tiers](docs/quickstart.md)
- [Move the harness to another device](docs/device-porting.md)
- [Optimization and evaluation protocol](docs/optimization-flow.md)
- [External operator contract](docs/operator-contract.md)
- [Architecture and trust boundaries](docs/architecture.md)
- [Experiment record schema](docs/experiment-schema.md)
- [Validation plan](docs/validation-plan.md)
- [Research questions, evidence, and limitations](research/README.md)

The current package is `Harness4H3-v0.4` research-preview. The pair lineage,
canonical evaluation boundary, real worker contract, and fail-closed A1 gate
are active protocol code; host-specific real evidence must still be produced
on the declared GPU device.

The repository also contains a manual-only [GPU contract workflow](.github/workflows/gpu-contract.yml).
It validates the real adapter, checkpoint manifest, device profile, worker
prerequisites, and benchmark schema on a self-hosted GPU runner without
starting a training worker or claiming an optimization result.
