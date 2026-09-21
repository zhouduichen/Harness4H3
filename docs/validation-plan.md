# Validation Plan

## Phase 0 — Algorithm & Harness Pre-GPU Validation

The current engineering gate is a hardware-independent proof of mechanism and
evidence flow. It does not claim that MiniMax-H3 can be trained, compressed,
distilled, or improved on a real GPU.

The required closed loop is:

```text
Controller → ExperimentPlan → validation → registered worker
→ real TinyH3 forward/backward/optimizer step → child checkpoint
→ independent evaluation → archive/trajectory → next plan
```

The gate passes only when the following are demonstrated:

| Area | Required evidence |
|---|---|
| Controller and validation | A legal plan executes; malformed or unregistered plans do not execute |
| Training | Finite loss, positive gradient norm, optimizer step, and changed intended weights |
| Immutability | Parent before/after SHA-256 is equal; frozen tensors and teacher remain unchanged |
| Child checkpoint | Child path and hash differ from parent; child reloads and all floating tensors are finite |
| Algorithms | Recovery, binary progressive distillation, and DMD2 reference-role updates run on TinyH3 |
| Reproducibility | Recovery, distillation, and DMD2 resume match uninterrupted CPU runs |
| Failure handling | NaN, zero gradient, OOM, corrupt checkpoint, invalid config, and child-evidence failures retain stable types |
| Closed loop | Real worker produces and evaluates `M0000 → M0001 → M0002` with correct lineage |

Run the gate from this repository root after installing the training extra:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q harness4h3 h3_training tools research
.venv/bin/python -m research.experiments.tiny_real_closed_loop \
  --output-root var/phase0-final-closed-loop
```

The corresponding run record is [Phase 0 validation evidence](../research/evidence/phase0-validation-2026-09-13.md).

TinyH3 is a CPU reference architecture. A Phase 0 pass is not evidence for
MiniMax-H3 quality, memory, latency, energy, deployment, or L40 performance.

## Phase I — Real H3 Validation

The engineering gate includes the fully offline fake closed loop plus provider, H3 header, executor and operator-contract tests. Default tests require no network, GPU, ComfyUI or checkpoint and cover valid convergence, schema rejection, policy/operator rejection, operator failure, OOM, critical quality regression, budget exhaustion, repeated failure, crash recovery, provider schema/auth behavior, safetensors/GGUF inspection, process timeout and parent immutability.

Research claims remain gated. The real H3 gate uses the fixed Windows RTX 5080 assets in `configs/models/minimax_h3_rtx5080.yaml`. It must produce a real `M0000 → M0001` transition using a model-level operator, then measure quality and at least one of latency, peak memory or model size on an external evaluator/hardware adapter. The efficiency metric must improve and quality regression must remain within the immutable TargetProfile threshold. Exploratory failures (including black-frame output) remain recorded and are not promoted.

Later comparisons will hold controller model, target and experiment budget fixed across Human Recipe, Fixed Pipeline, Random Search, LLM Only and LLM + Harness4H3. Required search metrics are experiments/GPU-hours/wall-time to target, failed experiments, human interventions, best feasible quality and Pareto hypervolume. Hypervolume and statistical baselines are not claimed by the fake delivery.

## Phase II — Remote SSH autonomous pipeline

The remote gate is separate from local compilation: staging and unit tests do
not prove that the live four-card server has executed v2. The acceptance run
must use the server-side `--on-server` entrypoint and record, for at least one
complete training/evaluation transition:

| Area | Required evidence |
|---|---|
| Remote Controller | local vLLM preflight, structured validated plan, bounded context digest, and next-plan reuse |
| Dynamic allocation | fresh resource decision with allocated GPU indices and worker lease |
| Overlap | evaluation lease, speculative worker, and Controller prefetch events with non-conflicting indices |
| Checkpoint I/O | prefetch completion/cached result before the worker lease, atomic staged file, and bounded cache contents |
| ComfyUI lifecycle | queue idle, `/free` unload response, verified memory waterline, and released lease |
| Experience | recipe/evaluation/power evidence appended and visible to the next plan |
| Utilization | per-GPU power/utilization samples; report measured values without claiming 300 W or 100% unless observed |
| Retention | active/rollback/Pareto checkpoint set and rejected/failed cleanup records |

The current remote evidence ledger is
[remote-pipeline-utilization-2026-09-16](../research/evidence/remote-pipeline-utilization-2026-09-16.md).
Its latest state explicitly records that the legacy campaign still owns the
lock; therefore it is not a substitute for the first live v2 acceptance run.

### Bounded remote validation window

When the shared server is explicitly available, run the bounded window from
the remote checkout:

```bash
tools/remote-validation-window.sh --start \
  --max-iterations 2 --max-runtime-s 7200
```

Without `--start` the command only prints campaign status. `--start` refuses
an operator pause, an already-running campaign, or any pre-existing GPU
compute process. It samples every configured GPU, requests a graceful pause at
the iteration/time boundary, and writes `validation-report.json` and
`validation-report.md` under a timestamped `work/remote-validation-window-*`
directory. The report contains event names and references to the existing
campaign state/event log; it never copies checkpoint/model bytes.

The report is evidence only when `evidence_status` is `measured`, all four GPU
indices have samples, and the final service state is not live. Its per-GPU
power/utilization values remain measurements; 300 W and 100% are never claimed
from a short window or from missing telemetry.
