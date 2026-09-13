# Phase 0 — Algorithm & Harness Pre-GPU Validation

Date: 2026-09-13
Environment: macOS, Python 3.12.13, CPU reference path
Scope: Harness and algorithm mechanism only; no MiniMax-H3 training claim

## Result

PASS. The current repository proves a real TinyH3 training and evaluation
closed loop without requiring a GPU, ComfyUI service, or MiniMax-H3 checkpoint.

Observed results:

| Check | Result |
|---|---|
| Full test suite | `206 passed` |
| Python compilation | `compileall` passed |
| Harness config validation | valid; 3 splits and 4 tasks |
| Experiment plan validation | valid mobile target |
| Offline fake loop | `target_satisfied`; `M0000 → M0001 → M0002` |
| Real TinyH3 closed loop | `completed`; 2 experiments; 0 failures; 0 rejections; 2 staged evidence manifests |
| Real TinyH3 lineage | `M0000(4) → M0001(4) → M0002(2)` |

## Reproduction commands

Run from `Harness4H3/`:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q harness4h3 h3_training tools research
.venv/bin/python -m harness4h3 --config configs/default.yaml validate-config --json
.venv/bin/python -m harness4h3 validate --target configs/targets/mobile_example.yaml --json
.venv/bin/python -m harness4h3 optimize \
  --target configs/targets/mobile_example.yaml \
  --session-dir var/phase0-offline-smoke \
  --session-id phase0-offline-smoke --json
.venv/bin/python -m research.experiments.tiny_real_closed_loop \
  --output-root var/phase0-final-closed-loop
```

The final TinyH3 report is written to
`var/phase0-final-closed-loop/report.json`; its model checkpoints, worker logs,
experiment records, and child evidence manifests are under the same ignored
run directory.

## Evidence covered

- Trainer tests prove finite losses, non-zero gradients, gradient accumulation,
  clipping, optimizer steps, and stable NaN/zero-gradient/OOM failures.
- Recovery tests prove declared trainable scopes, frozen teacher behavior,
  finite real updates, and exact resume.
- Progressive-distillation tests prove binary `4 → 2` schedule alignment,
  weighted video/audio endpoint supervision, frozen teacher behavior, and exact
  resume.
- DMD2 tests prove critic/student alternation, frozen teacher behavior, EMA and
  sampler state restoration, and exact resume. This is a reference skeleton,
  not a MiniMax-H3 recipe.
- Child-evidence tests prove parent hash protection, changed trainable tensors,
  unchanged frozen tensors, child reload, path-collision rejection, and stable
  failure types.
- The subprocess tests prove the trusted worker contract and preserve measured
  training metrics, sidecar child manifests, and trainer failure codes through
  `h3_model_worker.py` and `ExternalScriptOperator`.

## Boundary

This pass does not establish MiniMax-H3 forward/backward compatibility,
quality improvement, 4-step distillation, L40 memory or energy reduction,
edge deployment, or any superiority over search baselines. Those remain later real-H3
acceptance gates after a source-grounded, memory-feasible adapter is available.
