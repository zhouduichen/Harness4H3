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

All campaign components run on the server. The example points at the server-local vLLM OpenAI-compatible `/v1/chat/completions` endpoint (`qwen3.5-controller`); Ollama remains supported as an alternative server-local provider. If the configured LLM is unavailable, the supervisor records `proposal_invalid`/`campaign_error` and stops; it does not substitute a rule-based architecture. The H3 example uses the real cache geometry `5×16×16`, and the target dimensions in the YAML must match the cache used by the teacher adapter.

The Student worker performs meta-compiled topology validation, real H3 teacher distillation, full-precision checkpoint writing, and int8 quantization. The evaluator loads the quantized checkpoint, samples the Student latent, decodes it through the configured H3 video VAE, writes an MP4, and records decode validity, a clearly labelled structural quality proxy, latency, checkpoint size, and CUDA peak memory. A decode failure is never promoted.

Offline tests prove the contracts only. Real acceptance requires remote evidence containing a changed Student checkpoint, a decodable Student-generated video, hardware/quality evidence, and at least two durable proposal→train→evaluate rounds. Those conditions must not be inferred from CPU tests or a successful SSH launch alone.
