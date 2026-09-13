# Harness4H3 Research Program

This directory separates reproducible research programs and evidence from the
frozen reusable Harness implementation.

## Research question and hypothesis

**Question:** Can a bounded Controller choose device-aware H3 model/runtime
interventions and reach a feasible Pareto candidate more reliably than an
unstructured LLM-only process?

**Falsifiable hypothesis:** Given the same parent model, target device,
operator set, task splits, evaluator, Controller budget, and acceptance rules,
the Harness-mediated condition will produce more independently validated
feasible candidates per unit wall/GPU time, with fewer invalid or repeated
experiments, than the LLM-only condition.

This hypothesis has not yet been established. The repository currently
contains software-loop validation, real inference/benchmark evidence, one
measured quantization comparison, runtime-memory studies, and a documented
blocker for real model-changing training.

## What may count as a contribution

The project does not claim a new pruning, distillation, or quantization
algorithm. Its research contribution, if supported by future comparisons, is
the combination of:

- constrained LLM planning over a registered operator space;
- explicit target-device capabilities and hard constraints;
- immutable model/system lineage;
- independent artifact, quality, and hardware evaluation;
- failure-aware, append-only experience returned to later plans; and
- reproducible comparison against human recipe, fixed pipeline, random search,
  and LLM-only baselines.

These are currently an experimental-system design and research hypothesis,
not a proven performance result.

## Experimental design

| Category | Fixed or measured content |
|---|---|
| Independent variable | Search/controller condition and registered intervention |
| Controlled variables | Parent, target, tasks, seeds, workflow, sampler, evaluator, operator set, budget |
| Dependent variables | Feasible quality, latency, peak VRAM, size, energy, experiments/time-to-target, failures |
| Hard gates | Valid generation, quality floor/drop, device limits, parent immutability, child authenticity |
| Evidence units | ExperimentPlan, OperatorResult, EvaluationResult, artifacts, hashes, cost, trajectory |

The held-out split is not used to choose an intervention. Infrastructure
failure, optimization rejection, and policy/controller failure are reported
separately.

## Capability and claim matrix

| Item | Level | Claim allowed |
|---|---|---|
| Fake closed loop | Simulated | Protocol and persistence behavior only |
| Checkpoint header inspection | Real, non-executing | File identity/metadata only |
| H3 ComfyUI generation | Measured | Load/generation validity on recorded host |
| M5/M5.5 quantized comparison | Measured | Reported size/latency/quality under recorded controls |
| M6 runtime-memory branches | Measured | Reported outcomes, including VRAM rejection |
| Phase 0 TinyH3 algorithm/harness gate | CPU reference | Training, checkpoint, resume, failure, and closed-loop mechanism only |
| A1 model-changing training | Blocked | Reconnaissance and resource blocker only |
| 4×L40 distributed preflight | Configuration prepared, host unverified | Requirements and blocked preflight only |
| Pruning/distillation optimization | Not implemented | Interface intent only |
| Harness superiority over baselines | Not evaluated | No superiority claim |

## Current execution priority

Phase 0 is now the pre-GPU software gate and does not alter the real-H3 claim
boundary. The next execution step is to move the repository to the target host, replace
unverified device-profile facts with observations, run the persisted preflight,
and establish a real H3 ComfyUI sanity/dev/held-out baseline. M6 is optional
runtime-memory research and is not required for this migration. A1 remains
blocked until a source-grounded trainer performs the real forward/backward,
optimizer-step, child-save, parent-immutability, and child-reload checks.

## Experiment programs

| Program | Purpose | Execution level |
|---|---|---|
| `research.experiments.a0_model_evolution` | Bounded model-evolution campaign protocol | Offline by default; external worker optional |
| `research.experiments.a1_real_evolution` | Wire a real worker result into benchmark and next plan | Requires real trainer and ComfyUI |
| `research.experiments.m5_validation` | Parent/child reproducibility and generalization checks | Real ComfyUI |
| `research.experiments.m6_runtime_memory` | Controlled runtime-memory branches | Real ComfyUI |
| `research.experiments.m6_runtime_recipe` | Failure-aware bounded runtime campaign | Real ComfyUI + structured Controller |
| `research.experiments.power_study` | Statistical power/retry utilities | Offline analysis |

Use `python -m MODULE --help` from the repository root. Phase names are
historical experiment labels, not separate products.

## Evidence index

Real records are immutable research evidence:

- [RTX 5080 controlled H3 benchmark and M6 result](evidence/real-experiments/2026-09-09-windows-rtx5080.md)
- [A1 real-worker preflight](evidence/real-experiments/2026-09-10-a1-preflight.md)
- [Source-grounded H3 training reconnaissance](evidence/real-experiments/2026-09-11-a1-h3-training-recon.md)
- [Accepted NVFP4 Design Gene](evidence/design-genes/design-gene-h3-nvfp4.json)
- [Rejected VAE-tiling Design Gene](evidence/design-genes/design-gene-m6-vae-tiling.json)
- [Phase 0 algorithm and Harness validation](evidence/phase0-validation-2026-09-13.md)

Negative and blocked results are retained because they constrain the next
valid experiment. Files under `var/` are local run artifacts and are ignored
by Git; promote a run into `evidence/` only with its protocol, environment,
inputs, metrics, failures, and provenance.

## Current limitations

- No real H3 trainer is registered.
- The official BF16 H3 transformer is not memory-feasible on the measured RTX
  5080 Laptop host without a different loading/training strategy.
- Existing NVFP4/INT8 inference formats do not by themselves establish a
  trainable save/load roundtrip.
- Runtime studies are from one hardware class and a small task set.
- Energy/GPU-hour evidence is incomplete for several runs.
- Human recipe, fixed pipeline, random search, LLM-only, and LLM+Harness have
  not yet been compared under one powered protocol.

A [four-L40 device requirement profile](../configs/devices/l40x4-server.yaml),
[FSDP A1-T0 recipe](../configs/experiments/a1-t0-l40x4.yaml), and
[fixed worker example](../configs/a1-worker.l40x4.example.json) are prepared
for host setup. They are not execution evidence: no L40 hardware observation,
H3 forward/backward pass, optimizer step, or child checkpoint has been
recorded from that host.

## Next valid model-changing experiment

Before restarting A1, implement or adopt one source-grounded, memory-feasible
H3 `recovery_finetune` trainer and pass the isolated A1-T0 smoke gate: real
load, forward, finite loss, backward, non-zero gradient, optimizer step,
separate child save, unchanged parent, changed intended weights, and real
child reload. Only then connect it to `tools/h3_model_worker.py` and the fixed
benchmark loop.

## Research history

Completed and superseded [specifications](history/specs/) and
[implementation plans](history/plans/) are kept under `history/`. They
document why the frozen protocol exists but are not the current onboarding
path. Active usage is documented in [the repository README](../README.md),
[optimization protocol](../docs/optimization-flow.md), and
[device porting guide](../docs/device-porting.md).
