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
control flow and persistence, not real H3 optimization.

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
