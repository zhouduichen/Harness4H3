# A1 real model-evolution preflight — 2026-09-10

This is a preflight record, not a model-evolution result. No real child
checkpoint or benchmark score was produced.

## Read-only checks

- SSH to `autoresearch-5080` succeeded as `huangjiahao`.
- The configured parent checkpoint exists:
  `D:\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_nvfp4.safetensors`.
- The expected trainer and teacher-signal entry points do not exist:
  `D:\H3Training\train_worker.py` and
  `D:\H3Training\teacher_signal_worker.py`.
- The remote ComfyUI service is not listening on
  `http://127.0.0.1:8188/system_stats`.
- From the local machine, `http://100.88.143.10:8188/system_stats` also timed
  out.

## Consequence

A1 is implemented and tested, but the first real loop remains blocked before
the Controller can safely launch training. To run it, provide a real trainer
that implements the worker JSON contract, optionally a teacher-signal worker,
start ComfyUI on the same host as the deployed child checkpoint, and then run
the command in `docs/evogen-a1-real-model-evolution.md` with measured baseline
metrics. Until those prerequisites exist, only the contract fixtures may be
used, and they are explicitly non-scientific.
