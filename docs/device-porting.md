# Porting Harness4H3 to Another Device

Porting has two independent parts: install the reusable Harness, then describe
what the target device can actually execute. A device profile declares facts;
it does not implement a missing training or optimization backend.

## 1. Transfer and install

Copy or clone the repository onto the target device. Keep checkpoints and run
artifacts outside Git.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m harness4h3 --config configs/default.yaml validate-config --json
```

For Windows, replace `.venv/bin/python` with
`.venv\Scripts\python.exe`.

## 2. Create the device profile

Copy `configs/devices/rtx5080-laptop.yaml` to a new, device-specific filename.
Do not edit the measured RTX 5080 record to describe another host.

Record:

- `platform`: operating system, Python, PyTorch, and CUDA versions;
- `hardware`: GPU identity/count, VRAM, RAM, and supported dtypes;
- `paths`: required local services, manifests, and artifact roots;
- `services`: base URL, health path, and operations requiring the service;
- `limits`: hard resource/concurrency limits;
- `formats`: checkpoint/deployment formats the device can consume; and
- `capabilities`: one explicit enabled/disabled decision and reason per
  operation.

Use observed values from the target host. Unknown capability is disabled until
verified.

## 3. Install device-side dependencies

The base Harness needs only the dependencies in `pyproject.toml`. Real H3
inference additionally needs the existing ComfyUI/custom-node environment and
model assets. A real model-changing experiment additionally needs a separate
trainer implementation and its dependencies.

Do not upgrade the working ComfyUI environment merely to satisfy a trainer.
Use an independent trainer environment when one exists, and point the trusted
worker configuration at its fixed executable argv.

## 4. Start external services

Start only the services required by the selected capability:

- `inference` or `benchmark`: H3 ComfyUI API;
- `controller`: Ollama or the selected structured Controller provider;
- model-changing operation: a real trainer behind
  `tools/h3_model_worker.py`.

The Controller is optional for manual/fixed plans and offline deterministic
tests. Qwen is a tested Controller choice, not a protocol requirement.

## 5. Run preflight

```bash
.venv/bin/python tools/device_preflight.py \
  --profile configs/devices/my-device.yaml \
  --operator benchmark \
  --result var/preflight/benchmark.json --json
```

Interpret the checks literally:

- `profile.schema`: required fields are structurally valid;
- `capability.NAME`: the requested operation is declared available;
- `path.NAME`: a required path exists and, when declared, has the expected
  `file` or `directory` kind;
- `service.NAME`: a required HTTP health endpoint is reachable.

Exit code `0` means ready. Exit code `2` means blocked. A profile with
`verification.status: pending`, a disabled capability, or a skipped required
probe is intentionally blocked. Fix the reported fact and record the evidence;
do not bypass the check with a fixture or by editing the result.

## 6. Establish a baseline before optimization

For a real device, record:

1. checkpoint path, size, format, and SHA-256;
2. ComfyUI workflow and all controlled generation variables;
3. sanity, dev, and held-out task manifests;
4. quality/validity metrics;
5. latency, peak VRAM, and other target metrics; and
6. software/hardware versions from the device profile.

Only then run an intervention. Parent checkpoints are read-only, every model
change creates a separate child, and all failed/rejected outcomes remain in
the trajectory.

## Transfer checklist

- Offline tests pass on the target host.
- The device profile contains measured, not assumed, values.
- Preflight returns `ready` for the intended operation.
- Model/checkpoint formats match the real loader.
- ComfyUI sanity generation succeeds before dev/held-out benchmarking.
- Controller and evaluator configurations are frozen for the run.
- Output roots have enough free disk space and are excluded from Git.
- Unsupported model-changing operators remain disabled.

## 4×L40 pending training host

The repository includes a safe, unverified prerequisite set for a future
four-GPU L40 server:

- [device requirements](../configs/devices/l40x4-server.yaml);
- [A1-T0 FSDP smoke recipe](../configs/experiments/a1-t0-l40x4.yaml); and
- [fixed Linux worker example](../configs/a1-worker.l40x4.example.json).

Prepare the host in this order:

```text
install Linux host and NVIDIA driver
→ run nvidia-smi and record the actual four-GPU topology
→ install the Python/PyTorch CUDA environment
→ place the official Diffusers H3 assets
→ place the Harness repository and ComfyUI/custom nodes
→ create the evidence directories and one-sample manifest
→ run the benchmark preflight and persist its JSON
→ start ComfyUI and verify /system_stats
→ run one real sanity baseline and persist its EvaluationResult
→ mark only measured inference capability as verified
→ rerun benchmark preflight without --skip-services
→ run dev and held-out baseline only after sanity succeeds
→ keep recovery_finetune, prune, and distill disabled
```

The first preflight may correctly return `blocked`: the checked-in L40 profile
is pending and its capabilities are disabled. That result is useful setup
evidence. Do not enable a capability until the corresponding real service or
backend has been exercised and its evidence is available.

Run the initial hardware/path preflight on the L40 host:

```bash
mkdir -p var/preflight
python tools/device_preflight.py \
  --profile configs/devices/l40x4-server.yaml \
  --operator benchmark --skip-services \
  --result var/preflight/l40x4-setup.json --json
```

On PowerShell, create the directory with
`New-Item -ItemType Directory -Force var/preflight` and use the same command
with `.venv\Scripts\python.exe`.

The profile requires four visible GPUs whose names contain `NVIDIA L40`, at
least 40 GiB reported memory per GPU, at least 128 GiB system RAM, CUDA-enabled
PyTorch with four visible devices, and all operation-specific paths. These are
requirements, not claims about a server that has not been measured.

For the real inference baseline, use the existing benchmark command after
ComfyUI sanity generation succeeds:

```bash
python -m harness4h3 benchmark \
  --checkpoint /models/MiniMax-H3/<verified-inference-checkpoint> \
  --target <approved-target-profile>.yaml \
  --base-url http://127.0.0.1:8188 \
  --split sanity --reset-backend-before-run \
  --result var/benchmark/l40x4-sanity.json --json
```

The checkpoint and target arguments above must be replaced with the measured
assets and an intentionally approved target. Do not use
`configs/targets/rtx5080_example.yaml` to make an L40 performance claim unless
that target is explicitly adopted and recorded as a controlled variable.
Run `dev` and `heldout` only after `sanity` returns valid output. Preserve the
workflow, task manifests, checkpoint hash, profile, preflight JSON, and all
EvaluationResult files together.

Ordinary DDP is not valid for this model because it replicates the complete
BF16 transformer on every GPU. The smoke recipe fixes `fsdp_full_shard`, four
ranks, BF16, head-only training, precomputed conditioning, micro-batch one,
one sample, and one optimizer step. It remains `execution_enabled: false`
until a real trainer is deployed and reviewed. The current migration work does
not deploy or implement that trainer.
