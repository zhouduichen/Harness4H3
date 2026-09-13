# Harness4H3 Project Reorganization Design

Status: implemented and archived.

## Problem

Harness4H3 currently mixes three concerns at the repository top level:

1. the reusable optimization harness;
2. phase-specific A0/A1/M5/M6 research scripts; and
3. historical plans, specifications, design genes, evidence, and local run
   artifacts.

The reusable Python package is relatively small. The perceived size and
complexity come mainly from multiple historical entry points, a long README,
phase names exposed as the primary navigation, and research records presented
beside active user documentation. A new operator cannot easily tell which
files are required to move the harness to another device.

## Product position

The repository will present itself as a portable H3 optimization harness, not
as a claim of a new pruning or distillation algorithm.

Its research hypothesis is narrower:

> Under fixed quality constraints, a controller can select model-level and
> runtime-level interventions from a bounded operator set, execute them on a
> declared target device, and accept or reject candidates using independent
> measured evidence.

That hypothesis remains unproven for real model-changing H3 experiments until
a real trainer produces a loadable child checkpoint. Documentation must keep
engineering capability, simulated behavior, measured runtime optimization,
and future research claims separate.

## Goals

- Make the new-device path discoverable from the repository root.
- Keep one obvious project entry point: the `harness4h3` CLI.
- Separate reusable product code from phase-specific research material.
- Preserve every real failure, rejection, and benchmark record.
- Preserve `Harness4H3-v1.0` behavior and its frozen contracts.
- Provide one canonical device-profile template containing device facts,
  service endpoints, paths, resource limits, and operator capabilities.
- Make unsupported training capability explicit rather than implying that a
  configuration value implements a backend.
- Reduce the root README to installation, device setup, one dry run, one real
  benchmark, and links to deeper documentation.

## Non-goals

- No Harness protocol, schema, acceptance-policy, archive, trajectory, or
  evaluator redesign.
- No new orchestration abstraction or plugin framework.
- No implementation of H3 pruning, distillation, or recovery fine-tuning.
- No deletion of historical evidence or user work.
- No claim that the current RTX 5080 host can train the official BF16 H3
  transformer.
- No dependency addition solely for repository organization.

## Target repository structure

```text
Harness4H3/
├── README.md
├── pyproject.toml
├── harness4h3/                 reusable frozen package
├── tools/                      device-side worker and preflight tools
├── configs/
│   ├── devices/                one profile per execution device
│   ├── experiments/            reusable experiment defaults
│   ├── operators/              operator policy/configuration
│   └── targets/                immutable acceptance targets
├── examples/                   minimal runnable workflow/tasks
├── docs/
│   ├── quickstart.md
│   ├── device-porting.md
│   ├── optimization-flow.md
│   ├── operator-contract.md
│   └── architecture.md
├── research/
│   ├── experiments/            A0, A1, M5, M6, power-study entry points
│   ├── evidence/               real experiment records and design genes
│   └── history/                superseded plans/specifications/research notes
└── tests/
```

The internal `harness4h3/` package layout will not be reorganized in this
change. Moving its modules would create import churn without improving the
new-device workflow.

## File classification and movement

### Reusable project surface

Keep in place:

- `harness4h3/`
- `tools/h3_model_worker.py`
- `examples/workflow_api.json`
- `examples/tasks.yaml`
- `configs/operators/`
- `configs/targets/`
- `tests/`
- `pyproject.toml`

### Research experiments

Move the contents of `experiments/` to `research/experiments/`:

- `a0_model_evolution.py`
- `a1_real_evolution.py`
- `m5_validation.py`
- `m6_campaign.py`
- `m6_runtime_memory.py`
- `m6_runtime_recipe.py`
- `power_study.py`

Imports and CLI dispatch will be updated to the new module path. The two
currently modified M6 files must retain the user's complete working-tree diff
during the move. No compatibility wrapper package will be added unless an
existing test or documented public command proves it is necessary.

### Evidence

Move without changing content:

- `docs/real-experiments/` to `research/evidence/real-experiments/`
- `docs/experience/` to `research/evidence/design-genes/`

Documentation links and test fixture paths will be updated. Evidence files
remain tracked and immutable.

### Historical design material

Move completed phase material from `docs/superpowers/plans/` and
`docs/superpowers/specs/` into `research/history/plans/` and
`research/history/specs/` after the reorganization plan has been executed.
The current reorganization spec and implementation plan remain active under
`docs/superpowers/` until implementation is complete, then move with the rest
of the completed history.

Move `docs/evogen-a0-model-evolution.md`,
`docs/evogen-a1-real-model-evolution.md`, and
`docs/evogen-rsi-research.md` to `research/history/`. Their historical claims
will not remain in the primary onboarding path.

### Local artifacts

`var/`, `.venv/`, `.pytest_cache/`, `*.egg-info/`, `__pycache__/`, and
`*.pyc` remain ignored. The untracked `docs/.DS_Store` is local metadata and
will be removed from the working tree; `.DS_Store` will be added to
`.gitignore` so it does not return.

## Device profile

Create `configs/devices/rtx5080-laptop.yaml` as both the verified example and
the file operators copy for another device. A profile contains:

```text
schema version and device identity
OS, GPU count/name, VRAM, RAM, CUDA, supported dtypes
Python, worker workspace, checkpoint, artifact, and ComfyUI paths
ComfyUI and optional Controller endpoints
hard resource limits
supported checkpoint and deployment formats
per-operator enabled/disabled capability with a reason
```

The profile is declarative. It does not make an unsupported backend work.
For the current RTX 5080 profile, real inference and benchmark capability are
enabled; BF16 `recovery_finetune`, pruning, and distillation remain disabled
with the verified resource/format blocker.

The existing worker continues to consume its fixed worker configuration.
The device profile is used by a small read-only preflight command that checks
required fields, local paths, HTTP endpoints, and declared capabilities before
an experiment command is shown or run. It does not become another scheduler.

## User documentation

### Root README

The root README will contain only:

1. a two-paragraph project description and honest capability status;
2. installation;
3. `harness4h3 --help`;
4. device-profile setup;
5. an offline validation command;
6. a real benchmark command; and
7. links to active documentation and the research archive.

Phase narratives and full experiment results move out of the README.

### Active guides

- `docs/quickstart.md`: install, validate, inspect, benchmark.
- `docs/device-porting.md`: copy repository, create environment, fill device
  profile, start services, run preflight, and interpret failures.
- `docs/optimization-flow.md`: baseline, plan, validate, execute, evaluate,
  archive, and repeat; distinguish model and system candidates.
- `docs/operator-contract.md`: request/result fields, parent immutability,
  child authenticity, metrics, logs, and failure taxonomy.
- `docs/architecture.md`: concise component boundaries and frozen-core rule.

Every documented command must either run offline in tests or be explicitly
marked as requiring an external service/checkpoint.

## CLI and compatibility strategy

`harness4h3` remains the only installed console script. Current user-facing
commands remain available during this cleanup. Legacy commands stay callable
but move to a clearly labeled `Legacy/replay` section of help and docs where
possible without changing argument behavior.

Phase-specific Python scripts are research entry points, not primary product
commands. Existing CLI commands that call them will import from
`research.experiments`. Direct historical paths under `experiments/` are not
promised as a stable public API.

## Data flow after reorganization

```text
device profile + target + model state
                 ↓
          device preflight
                 ↓
 Controller or fixed/manual ExperimentPlan
                 ↓
 frozen validation pipeline → registered operator → worker
                 ↓
          model/system candidate
                 ↓
   fixed ComfyUI benchmark + independent evaluator
                 ↓
 archive + append-only trajectory + next controller context
```

The data flow is unchanged. Reorganization changes discoverability and file
ownership, not experiment semantics.

## Error handling

- Invalid device profiles fail before an experiment starts.
- Missing paths or unreachable endpoints report stable preflight fields.
- An operator declared disabled reports its configured reason.
- Existing worker, benchmark, controller, and policy failure taxonomies remain
  unchanged.
- No command silently falls back from a real worker to a fixture or simulator.

## Verification

The reorganization is accepted when:

- the full existing test suite passes;
- `python -m compileall` passes for `harness4h3`, `research`, `tools`, and
  `tests`;
- `harness4h3 --help` and all current subcommands still parse;
- offline `validate`, fake optimization, lineage, Pareto, and replay tests pass;
- device-profile preflight has one passing fixture and explicit failures for
  missing fields, unsupported operators, missing paths, and unreachable
  services;
- no tracked evidence file is deleted or modified during movement;
- the user's pre-existing changes in `m6_campaign.py` and
  `m6_runtime_recipe.py` are byte-for-byte preserved apart from their paths;
- the root README links resolve; and
- the Git working tree contains no tracked generated artifact or `.DS_Store`.

## Expected result

A new operator should be able to open the root README, copy a device profile,
run preflight, and understand whether the device can run inference, benchmark,
or real model-changing optimization. Researchers can still find all A0/A1/M5/
M6 code and evidence under `research/`, but those phase details no longer
obscure the reusable project surface.
