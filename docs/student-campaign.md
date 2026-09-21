# Autonomous H3 Student campaign

The mainline command is a detached, resumable campaign. The local LLM returns only a JSON `StudentProposal`; it cannot return Python, shell commands, or a remote path.

```bash
./.venv/bin/python -m harness4h3 student-campaign \
  --student-config configs/student-campaign.example.yaml validate --json

./.venv/bin/python -m harness4h3 student-campaign \
  --student-config configs/student-campaign.example.yaml run \
  --detach --max-rounds 4 --json

./.venv/bin/python -m harness4h3 student-campaign \
  --student-config configs/student-campaign.example.yaml status --json
```

Before starting, deploy the current `harness4h3/student`, `harness4h3/remote`, `tools/student_*worker.py`, `tools/student_campaign_supervisor.py`, and the trusted YAML to the configured `remote.harness_root`. The detached process writes its PID, log, `resume.json`, append-only campaign events, compact experience records, and final `campaign-result.json` under `student.remote_campaign_root`.

The fixed manifest can also be created explicitly on the server:

```bash
python tools/student_build_manifest.py \
  --cache-dir /data/models/MiniMax-H3/harness4h3/cache \
  --output /home/intern/huangjiahao/Harness4H3/work/student-campaign/evaluation-manifest.json
```

All campaign components run on the server. The example points at the server-local vLLM OpenAI-compatible `/v1/chat/completions` endpoint (`qwen3.5-controller`); Ollama remains supported as an alternative server-local provider. If the configured LLM is unavailable, the supervisor records `proposal_invalid`/`campaign_error` and stops; it does not substitute a rule-based architecture. The H3 example uses the real cache geometry `5×16×16`, and the target dimensions in the YAML must match the cache used by the teacher adapter.

The remote training worker and video evaluator use `worker_device: auto`: they select a GPU with the configured free-memory budget and wait on the server when other jobs (including the controller LLM) occupy the cards. This avoids hard-coding a GPU index or treating temporary resource contention as an architecture failure.

For the effective-optimization mainline, set `quality_backend: clip_temporal` and provide a local `clip_model_path`. The supervisor first builds or reuses one fixed `evaluation-manifest.json`, then calibrates the H3 teacher on that exact manifest. Every Student candidate is evaluated on the same cases and seeds with semantic image-text similarity, adjacent-frame temporal consistency, motion, median/p95 latency, peak CUDA memory, and checkpoint size. CLIP quality runs on server CPU by default so it does not compete with the H3 VAE/Student GPU allocation; the generation latency metric excludes this evaluator overhead. The structural proxy remains available for smoke tests only; it is diagnostic evidence and must not be used as the production quality gate.

The strict gate compares each candidate to the H3 baseline and the current Student incumbent. It requires `quality / H3 quality >= quality_floor_ratio` and accepts a candidate only when it is on the Pareto frontier with either the configured reward delta or a material efficiency gain. Rejected-but-valid candidates are recorded as `rejected` and fed back to the server-local LLM; they do not consume the hard failure budget. The campaign stops after `no_improvement_patience` rounds without a frontier improvement or after `max_rounds`.

The Student worker performs meta-compiled topology validation, real H3 teacher distillation, full-precision checkpoint writing, and int8 quantization. The evaluator loads the quantized checkpoint, samples the Student latent, decodes it through the configured H3 video VAE, writes an MP4, and records decode validity, semantic/temporal quality, latency, checkpoint size, and CUDA peak memory. A decode failure or unavailable strict quality backend is never promoted.

Offline tests prove the contracts only. Real acceptance requires remote evidence containing a changed Student checkpoint, a decodable Student-generated video, hardware/quality evidence, and at least two durable proposal→train→evaluate rounds. Those conditions must not be inferred from CPU tests or a successful SSH launch alone.
