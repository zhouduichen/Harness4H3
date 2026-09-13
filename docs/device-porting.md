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
  --operator benchmark --json
```

Interpret the checks literally:

- `profile.schema`: required fields are structurally valid;
- `capability.NAME`: the requested operation is declared available;
- `path.NAME`: a required path exists on the target host;
- `service.NAME`: a required HTTP health endpoint is reachable.

Exit code `0` means ready. Exit code `2` means blocked. Fix the reported fact
or disable the capability; do not bypass the check with a fixture.

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
