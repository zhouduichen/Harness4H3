# A1 — First Real Model Evolution

A1 is the first campaign that is allowed to claim a model-changing result.
The Harness remains frozen and the Controller still emits only an
`ExperimentPlan`; the real work happens in a trusted external worker:

```text
validated parent checkpoint
  -> Controller plan
  -> h3_model_worker.py
  -> configured trainer/distiller
  -> new child checkpoint
  -> real ComfyUI dev/held-out benchmark
  -> next Controller plan
```

## Required command

Run A1 on a host where the parent checkpoint, trainer, worker config, and
ComfyUI service are all reachable. The external worker and ComfyUI should be
on the same Windows host when the child checkpoint is deployed into the
ComfyUI model directory:

```bash
PYTHONPATH=. .venv/bin/python -m harness4h3 a1-evolve \
  --parent-checkpoint 'D:/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors' \
  --worker-command 'D:/H3Training/.venv/Scripts/python.exe' 'D:/MinMax-H3/Harness4H3/tools/h3_model_worker.py' \
  --worker-config 'configs/a1-worker.json' \
  --base-url 'http://127.0.0.1:8188' \
  --controller ollama \
  --controller-url 'http://127.0.0.1:11434' \
  --baseline-quality 0.991137 \
  --baseline-model-size-gb 12.5286368 \
  --baseline-latency-s 90.28058435407002 \
  --baseline-peak-memory-gb 16.29452817 \
  --max-experiments 2 \
  --output-root var/a1-real-evolution
```

`--worker-config` is trusted local configuration. It contains the fixed
trainer argv, optional teacher-signal command/cache, and optional deployment
directory. The Controller cannot alter any of these values. See
[`configs/a1-worker.example.json`](../configs/a1-worker.example.json).

The trainer must accept the worker's `--request` and `--result` arguments. Its
result must contain a new local checkpoint and a complete `ModelState`. The
adapter stages the checkpoint in the experiment artifact directory, optionally
copies it to `deploy_model_dir`, and reports the trainer's non-zero GPU-hour
cost. Returning the parent path, a virtual URI, or a missing checkpoint is an
execution failure.

## Teacher caching

If `teacher_signal_command` and `teacher_cache_dir` are configured, the worker
keys a teacher-signal manifest by parent state, operator, and arguments. A
cache hit avoids loading the teacher again. The trainer receives the cache
metadata in `trainer_request.json`; the worker never claims that a manifest is
a valid teacher signal unless the configured signal command created it.

This supports the 16GB laptop constraint: precompute teacher signals, unload
the teacher, then train the student. Online teacher/student training should be
used only when the real worker proves that the configuration fits.

## Evidence interpretation

A1 writes the same `campaign.json`, `report.json`, model archive, Pareto
archive, and trajectory JSONL as A0. A real run has
`offline_simulation=false`, `real_worker=true`, real trainer cost, and real
benchmark summaries in each evaluation. Contract fixtures and fake evaluator
tests are not scientific A1 results. If the checkpoint, trainer, SSH access,
or ComfyUI service is unavailable, the correct result is an explicit
preflight failure rather than an offline substitute.
