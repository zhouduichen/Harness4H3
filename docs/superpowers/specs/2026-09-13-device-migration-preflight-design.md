# Device Migration Preflight Design

## Goal

Make Harness4H3 portable to a new execution device by providing one clear,
evidence-producing path from environment setup to a real H3 inference baseline
and A1-T0 readiness decision. This increment prepares execution; it does not
implement a real H3 trainer and does not claim a model-changing experiment.

## Scope

In scope:

- a measured-device profile contract for the four-L40 target;
- read-only checks for profile schema, hardware, software, paths, and services;
- a fixed baseline command that can be run before any optimization;
- a bounded A1-T0 configuration and trusted worker command for a later trainer;
- one onboarding document that another operator can follow from a clean host;
- JSON evidence and log locations for ready, blocked, and failed states; and
- tests that keep the preflight decision and configuration boundaries stable.

Out of scope:

- changes to Harness4H3-v1.0 core acceptance rules or orchestration semantics;
- implementation of `recovery_finetune`, pruning, or distillation;
- enabling any model-changing capability before a real trainer is verified;
- automatic installation, driver changes, service startup, or remote execution;
- M6 runtime-memory experiments; and
- Controller, archive, trajectory, or benchmark protocol redesign.

## Architecture

The device profile is the single declaration of target-host facts and expected
capabilities. `tools/device_preflight.py` reads it without mutating the host,
probes only the checks relevant to the requested operator, and emits a stable
JSON result. The profile remains `pending` and model-changing capabilities stay
disabled until the target host supplies measured evidence.

The operational path is:

```text
device profile
  -> profile/schema check
  -> hardware + PyTorch/CUDA checks
  -> operation-scoped path checks
  -> operation-scoped service health checks
  -> ready | blocked
  -> fixed baseline benchmark
  -> evidence directory
```

The A1-T0 recipe is a separate declarative input for a future external trainer:
four-rank FSDP full-shard, BF16, head-only scope, one sample, one optimizer
step, sharded save, and explicit reload/authenticity evidence. It is not
executable by the current repository because the real trainer is not present.

## Components

### Device profile

`configs/devices/l40x4-server.yaml` records expected L40 hardware, minimum
memory, Linux/PyTorch/CUDA requirements, host paths, service endpoints, limits,
formats, and capability reasons. Unknown facts are represented as pending or
disabled, never inferred from the filename.

### Preflight

`tools/device_preflight.py` provides a read-only `preflight(profile_path,
operator, ...)` function and CLI. It reports each check by name, status, and
detail. A failed or skipped required check produces exit code 2. Hardware
requirements are opt-in through the profile so existing non-GPU profiles keep
their previous behavior.

### Baseline and A1-T0 instructions

The quickstart and device-porting guide define installation, profile editing,
preflight, baseline checkpoint identity, sanity generation, benchmark splits,
and evidence promotion. The A1-T0 recipe and worker example pin a future
four-rank invocation without pretending that invocation is currently runnable.

### Evidence

Host-generated preflight JSON belongs under `var/preflight/`. Benchmark and
training logs belong under `var/benchmark/` and the configured experiment run
directory. A research evidence record may be promoted only with the command,
inputs, environment, metrics, hashes, and failure classification. No command
edits a result to turn `blocked` or `failed` into `ready`.

## Decision states and failure handling

- `ready`: all declared checks for the requested operation passed.
- `blocked`: a prerequisite is absent, disabled, skipped, or not yet verified;
  no experiment may start.
- `failed`: a real baseline or later experiment started and returned an error;
  the original logs and taxonomy remain intact.
- `rejected`: a real candidate completed evaluation but failed a declared
  quality or hardware gate; this is optimization evidence, not infrastructure
  failure.

The preflight must not treat a skipped hardware probe as success. It must also
not check paths or services unrelated to the selected operator, so a benchmark
operator is not blocked by an uninstalled trainer and a future trainer check
does not require a running ComfyUI service unless the profile says so.

## Verification

Offline verification runs profile validation, unit tests for positive and
negative hardware observations, operation-scoped path filtering, YAML/JSON
configuration tests, and compile checks. The target-host verification is
separate: run the profile on the L40 host, persist its JSON, start ComfyUI,
run sanity, then dev/held-out baseline benchmark, and retain all logs.

The final gate for this increment is documentation and configuration
completeness plus a green offline suite. It is not A1-T0 completion. A1-T0
remains blocked until an independently verified, source-grounded trainer
performs real forward/backward/update/save/reload checks.

## Next step

On the new host, install the declared environments, replace unverified profile
facts with observations, run the preflight, and record the baseline. Only
after those facts are available should the real H3 training implementation be
selected or adopted and reviewed against the A1-T0 authenticity gate.
