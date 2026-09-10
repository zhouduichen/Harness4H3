# Autonomous Model Evolution Campaign A0

A0 is the model-level inner loop:

```text
M_t -> Controller diagnosis -> model operator -> training/distillation
    -> candidate checkpoint -> multi-fidelity evaluation -> M_(t+1)
```

It is separate from the M6 runtime-memory campaign. The registered action
space is `create_student`, `prune_blocks`, `prune_heads`, `prune_channels`,
`distill`, `step_distill`, `recovery_finetune`, and `quantize`.

Run the offline preflight with:

```bash
cd Harness4H3
PYTHONPATH=. .venv/bin/python -m harness4h3 a0-evolve \
  --controller mock \
  --target configs/targets/rtx5080_example.yaml \
  --output-root var/a0-model-evolution
```

The offline backend transforms `ModelState` deterministically and marks every
child with `offline_simulation=true`. It verifies Controller autonomy, model
lineage, budget accounting, negative/rejected candidates, and report shape; it
does not train H3 weights and cannot establish real RTX 5080 quality or VRAM
results.

The fidelity tiers are:

| Tier | Purpose |
| --- | --- |
| 0 | static schema, operator, shape, and budget checks |
| 1 | cheap structural candidate screening |
| 2 | short distillation/recovery evaluation |
| 3 | full candidate evaluation and held-out promotion gate |

A0 defaults to 24 GPU-hours, 20 experiments, one concurrent experiment, and
zero human optimization interventions. Rejected candidates are archived but
never become the next active parent. Execution failures consume the failure
budget; clean rejections do not.

For real training, pass a fixed argv worker to the same entry point, for
example `--external-operator-command python tools/h3_train_worker.py`. A0
then builds registered `ExternalScriptOperator` instances. Each external worker must receive the
immutable parent state, write a new child checkpoint inside its experiment
artifact directory, and return a validated `ModelState`; the worker's reported
GPU-hours then become the campaign's cost evidence. The evaluator/backend must
also be switched from the fake evaluator to the real benchmark before making
hardware claims.

The persisted files are `campaign.json`, `report.json`, `models/`, `pareto/`,
and `trajectories.jsonl` under the selected output root.
