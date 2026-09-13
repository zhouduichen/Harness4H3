# Harness4H3

Harness4H3 is a device-aware research harness for reproducible MiniMax H3
optimization experiments. It studies one bounded question:

> Can a fixed controller select model-level or runtime-level interventions for
> a declared device, while an independent evaluator retains only candidates
> supported by measured quality and hardware evidence?

The repository provides the experiment protocol, constrained operator
execution, model/system lineage, independent evaluation, Pareto archive, and
append-only trajectory. It does **not** currently provide a working H3
fine-tuning, pruning, or distillation backend.

## Capability status

| Capability | Evidence level | Current status |
|---|---|---|
| Offline optimization loop | Simulated and tested | Available |
| H3 checkpoint inspection | Real file metadata | Available |
| H3 generation through ComfyUI | Measured on RTX 5080 Laptop | Available |
| Quality/hardware benchmark | Measured on RTX 5080 Laptop | Available |
| Runtime-memory interventions | Measured, including rejected results | Experimental |
| Model-changing fine-tuning | Source-reconnaissance only | Blocked by memory/implementation |
| Real pruning/distillation | Contract only | Not implemented |

Engineering completeness is not presented as a new optimization algorithm.
The research contribution under test is the evidence-grounded, device-aware
closed-loop protocol. See the [research index](research/README.md) for claims,
variables, experiments, negative results, and limitations.

## System under study

```text
DeviceProfile + TargetProfile + ModelState + Budget
                         ↓
              fixed/manual Controller
                         ↓
                  ExperimentPlan
                         ↓
 schema → policy → budget → registered-operator validation
                         ↓
            model or runtime candidate
                         ↓
         ComfyUI benchmark + independent evaluator
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
```

Windows uses `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

Run the fully offline protocol smoke test:

```bash
.venv/bin/python -m harness4h3 optimize \
  --target configs/targets/mobile_example.yaml \
  --session-dir var/offline-smoke --session-id offline-smoke --json
```

Before using a real device, run the declarative capability preflight on that
device:

```bash
.venv/bin/python tools/device_preflight.py \
  --profile configs/devices/rtx5080-laptop.yaml \
  --operator benchmark --json
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

## Repository map

```text
harness4h3/          frozen reusable harness implementation
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

Harness behavior is frozen as `Harness4H3-v1.0`; only correctness and security
fixes belong in the core. New research should add an experiment, operator
backend, device profile, or evidence record without changing the acceptance
rules after results are observed.
