# Remote H3 Experience Import and Continuous Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable SSH-backed campaign that imports real MiniMax-H3 training evidence, evaluates each child on the remote ComfyUI/L40 device, and promotes only measured Pareto-feasible children.

**Architecture:** Keep the frozen Harness controller and core acceptance semantics intact. Add a research-layer experience store, a trusted SSH transport/tunnel, an importer for remote worker results, a remote campaign coordinator, and an energy sampler; reuse the existing H3 benchmark/evaluator through an SSH port forward and persist normalized records locally without copying 66GB checkpoints.

**Tech Stack:** Python 3.9+, existing `harness4h3` dataclasses and JSONL stores, `subprocess`/OpenSSH, `socket`, PyYAML, OpenCV evaluator, ComfyUI HTTP API, PyTorch/FSDP worker already installed on `Jiayu-intern`.

## Global Constraints

- The Controller remains fixed; imported experience is context only and no Controller fine-tuning is implemented.
- The primary host is the configured SSH alias `Jiayu-intern`; the fallback `autoresearch-5080` is not assumed reachable.
- Remote H3 artifacts stay under `/data/models/MiniMax-H3` and are referenced by path and SHA-256; checkpoint bytes are not copied during import.
- Remote worker and evaluator commands are trusted configuration, never values emitted by the Controller.
- Parent checkpoints are immutable; deployment uses a new candidate filename or an existing link that already points to the same verified file.
- A result without independent Q/L/M/E evidence is `training_only_unvalidated` and cannot become an active parent.
- Acceptance gates run before Pareto ranking; missing metrics are `None`, never zero.
- Existing user changes in `research/experiments/m6_campaign.py` and `research/experiments/m6_runtime_recipe.py` remain untouched.
- Every code task ends with a focused test and a commit containing only that task's files.

---

## File map

Create the following focused units:

- `harness4h3/memory/experience.py`: append-only normalized experience records and store.
- `harness4h3/remote/__init__.py`: public remote API exports.
- `harness4h3/remote/ssh.py`: validated remote paths, SSH commands, JSON transfer, and ComfyUI tunnel.
- `harness4h3/remote/importer.py`: normalize remote trainer result/evidence/request files.
- `harness4h3/remote/reward.py`: Q/L/M/E normalization and reward calculation.
- `harness4h3/remote/decision.py`: training-evidence and benchmark acceptance gates.
- `harness4h3/remote/power.py`: sampled remote `nvidia-smi` power integration.
- `harness4h3/remote/config.py`: validated remote campaign YAML and immutable benchmark settings.
- `research/experiments/remote_h3_closed_loop.py`: import/evaluate/train/resume coordinator.
- `examples/remote_linux_h3_workflow_api.json`: API-format Linux BF16 H3 workflow.
- `configs/remote-l40-h3.yaml`: fixed remote campaign configuration.
- `configs/targets/l40x4_h3_example.yaml`: L40×4 benchmark target profile.

Modify only these existing integration points:

- `harness4h3/memory/__init__.py`: export the experience types.
- `harness4h3/benchmark/h3.py`: accept an optional power sampler and put energy in `HardwareMetrics`.
- `harness4h3/cli.py`: add `import-experience` and `remote-campaign` commands.
- `docs/quickstart.md`, `docs/optimization-flow.md`, `docs/experiment-schema.md`: document the commands and status semantics.

---

### Task 1: Add normalized append-only experience storage

**Files:**
- Create: `harness4h3/memory/experience.py`
- Modify: `harness4h3/memory/__init__.py`
- Test: `tests/unit/test_experience.py`

**Interfaces:**
- Produces `ExperienceRecord`, `ExperienceStore`, and `experience_status()` for the importer and campaign.
- `ExperienceRecord.to_dict()` returns redacted JSON-safe metadata and `ExperienceRecord.from_dict()` round-trips schema version 1.
- `ExperienceStore.append(record)` is fsynced append-only; `ExperienceStore.read()` yields records; `ExperienceStore.source_hashes()` returns `{source_uri: source_sha256}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_experience_round_trips_and_preserves_unvalidated_status(tmp_path):
    from harness4h3.memory.experience import ExperienceRecord, ExperienceStore

    record = ExperienceRecord(
        experience_id="xp-m0006",
        source_uri="ssh://Jiayu-intern/data/m0006.json",
        source_sha256="a" * 64,
        source_kind="trainer_result",
        experiment_id="distill16-m0006",
        parent_model_id="M0005",
        child_model_id="M0006",
        operator="step_distill",
        operator_args={"target_steps": 16},
        training={"optimizer_steps": 1, "gradient_norm": 1.2},
        evaluation=None,
        decision={"status": "not_evaluated"},
        reward=None,
        status="training_only_unvalidated",
        provenance={"remote_path": "/data/m0006.json"},
        created_at="2026-09-14T00:00:00+00:00",
    )
    store = ExperienceStore(tmp_path / "experience.jsonl")
    store.append(record)
    loaded = list(store.read())
    assert loaded == [record]
    assert store.source_hashes() == {record.source_uri: record.source_sha256}
    assert loaded[0].to_dict()["status"] == "training_only_unvalidated"


def test_experience_store_rejects_duplicate_source_hash(tmp_path):
    from harness4h3.memory.experience import ExperienceRecord, ExperienceStore

    record = ExperienceRecord.minimal("xp-1", "ssh://host/result.json", "b" * 64)
    store = ExperienceStore(tmp_path / "experience.jsonl")
    store.append(record)
    store.append(record)
    assert len(list(store.read())) == 1
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest tests/unit/test_experience.py -q`

Expected: FAIL because `harness4h3.memory.experience` does not exist.

- [ ] **Step 3: Implement the record and store**

Use a frozen dataclass with these fields and defaults:

```python
@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    source_uri: str
    source_sha256: str
    source_kind: str
    experiment_id: str
    parent_model_id: Optional[str]
    child_model_id: Optional[str]
    operator: Optional[str]
    operator_args: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Optional[Mapping[str, Any]]
    decision: Mapping[str, Any]
    reward: Optional[float]
    status: str
    provenance: Mapping[str, Any]
    created_at: str
    schema_version: int = 1
```

Validate non-empty identity, a 64-character hexadecimal source hash, and the
five allowed statuses: `training_only_unvalidated`, `evaluated_candidate`,
`accepted`, `rejected`, and `failed`. Store `json.dumps(redact(asdict(record)))`
with `ensure_ascii=False`, `sort_keys=True`, newline, flush, and `os.fsync`.
Use the source URI/hash pair as the idempotency key and return without writing
when it already exists.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `pytest tests/unit/test_experience.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/memory/experience.py harness4h3/memory/__init__.py tests/unit/test_experience.py
git commit -m "feat: add normalized H3 experience store"
```

### Task 2: Implement safe SSH transport and ComfyUI tunnel

**Files:**
- Create: `harness4h3/remote/__init__.py`
- Create: `harness4h3/remote/ssh.py`
- Test: `tests/unit/test_remote_ssh.py`

**Interfaces:**
- `RemoteConfig(host, harness_root, model_root, comfyui_root, comfyui_port=8188, python="python3")`.
- `SSHClient(config, runner=subprocess.run)` exposes `run(argv)`, `read_text(path)`, `read_json(path)`, `write_json(path, value)`, `find(pattern)`, `sha256(path)`, and `ensure_model_link(model_path, model_id)`.
- `ComfyUITunnel(client)` is a context manager exposing `base_url`; it starts `ssh -N -L <free_port>:127.0.0.1:<remote_port> <host>`, waits for the local TCP port, and terminates the child process on exit.

- [ ] **Step 1: Write failing tests for root validation, quoting, link safety, and tunnel cleanup**

```python
def test_remote_path_must_stay_under_configured_root():
    from harness4h3.remote.ssh import RemoteConfig, SSHClient, RemotePathError

    client = SSHClient(RemoteConfig("host", "/srv/harness", "/srv/models", "/srv/comfy"), runner=lambda *a, **k: None)
    with pytest.raises(RemotePathError):
        client.read_text("/srv/harness/../etc/passwd")


def test_command_uses_argument_quoting_and_never_shell_true():
    calls = []
    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    from harness4h3.remote.ssh import RemoteConfig, SSHClient
    SSHClient(RemoteConfig("host", "/srv/harness", "/srv/models", "/srv/comfy"), runner=runner).run(("sha256sum", "/srv/harness/a file.json"))
    assert calls[0][1]["shell"] is False
    assert "a file.json" in " ".join(calls[0][0])


def test_existing_link_pointing_elsewhere_is_refused(tmp_path):
    from harness4h3.remote.ssh import RemoteLinkConflict
    # Exercise the pure link decision helper with same/other target values.
    from harness4h3.remote.ssh import link_action
    assert link_action("/models/M0006.safetensors", "/models/M0006.safetensors") == "keep"
    with pytest.raises(RemoteLinkConflict):
        link_action("/models/other.safetensors", "/models/M0006.safetensors")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_remote_ssh.py -q`

Expected: FAIL because the remote package and helpers do not exist.

- [ ] **Step 3: Implement transport**

Use `shlex.quote` only when composing the single remote command string passed
after the SSH host; never use a local shell. Resolve local path inputs against
the configured remote roots with `PurePosixPath` and reject `..` escapes. Cap
metadata reads at 8 MiB. `write_json` sends base64-encoded UTF-8 through a
fixed `python3 -c` decoder and writes atomically to a configured campaign path.
`find` is restricted to the configured harness root and returns newline-separated
paths. `ensure_model_link` first runs `readlink`/`test`, then creates only a
missing link with `ln -s`; it rejects an existing different target and a
deployment name equal to a parent checkpoint.

The tunnel allocates a local port with a bound socket, launches `Popen` with
`stdin=DEVNULL`, `stdout=PIPE`, `stderr=PIPE`, and polls the port for at most
10 seconds. On exit it sends `terminate`, waits two seconds, then `kill` only
if still alive.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/test_remote_ssh.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/remote/__init__.py harness4h3/remote/ssh.py tests/unit/test_remote_ssh.py
git commit -m "feat: add safe SSH transport for remote H3"
```

### Task 3: Normalize remote trainer results into experience

**Files:**
- Create: `harness4h3/remote/importer.py`
- Test: `tests/unit/test_remote_importer.py`

**Interfaces:**
- `RemoteResultImporter.import_results(files, requests=()) -> ImportSummary`.
- `ImportSummary(imported, skipped_duplicates, corrupt, records)` is immutable and JSON serializable.
- `normalize_trainer_result(source_uri, source_sha256, result, request=None, evidence=None) -> ExperienceRecord`.

- [ ] **Step 1: Write failing normalization tests**

```python
def test_successful_training_result_is_unvalidated_without_evaluation():
    from harness4h3.remote.importer import normalize_trainer_result

    record = normalize_trainer_result(
        "ssh://Jiayu-intern/data/m0006.json",
        "c" * 64,
        {
            "status": "success",
            "metrics": {"optimizer_steps": 1, "gradient_norm": 0.2, "child_sha256": "d" * 64},
            "output_state": {
                "model_id": "M0006", "parent_model_id": "M0005",
                "checkpoint_path": "/data/models/MiniMax-H3/harness4h3/distill16/M0006.safetensors",
                "algorithm_state": {"operator": "step_distill", "target_steps": 16},
            },
        },
    )
    assert record.status == "training_only_unvalidated"
    assert record.operator == "step_distill"
    assert record.reward is None


def test_failed_result_preserves_failure_and_does_not_create_child():
    from harness4h3.remote.importer import normalize_trainer_result

    record = normalize_trainer_result("ssh://host/failure.json", "e" * 64, {"status": "failed", "failure_type": "training_oom", "message": "OOM"})
    assert record.status == "failed"
    assert record.child_model_id is None
    assert record.decision["failure_type"] == "training_oom"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/unit/test_remote_importer.py -q`

Expected: FAIL because the importer is not implemented.

- [ ] **Step 3: Implement result normalization and discovery**

For successful results, copy `output_state`, `metrics`, and the algorithm
operator into separate record fields. Join a request file by child model ID
when available to recover experiment ID and operator arguments. Join the
evidence sidecar by its declared path. If no evaluation contains numeric
quality, latency, peak memory, and energy, use
`training_only_unvalidated`; never synthesize an evaluation from loss.
Map failed result status to `failed`, preserve `failure_type`, and set child
ID to `None`. Use an ID of `xp-<model_id>-<source_hash[:12]>`.

The discovery method asks `SSHClient.find` for
`trainer_result_*.json`, reads each file and its adjacent evidence/request
metadata, calculates the remote source SHA-256, and appends through
`ExperienceStore`. A malformed file adds a `corrupt` item with source and
error, then import continues. No checkpoint or video file is read during
import.

- [ ] **Step 4: Run focused tests and verify pass**

Run: `pytest tests/unit/test_remote_importer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/remote/importer.py tests/unit/test_remote_importer.py
git commit -m "feat: import remote H3 training results as experience"
```

### Task 4: Add reward calculation and evaluator-gated decisions

**Files:**
- Create: `harness4h3/remote/reward.py`
- Create: `harness4h3/remote/decision.py`
- Test: `tests/unit/test_remote_decision.py`

**Interfaces:**
- `RewardWeights(alpha, beta, gamma, delta)` validates finite non-negative weights.
- `compute_reward(quality, hardware, baseline, weights) -> RewardResult` returns normalized terms and `reward=None` if any Q/L/M/E value is missing or invalid.
- `AcceptanceInput(training_metrics, benchmark_summary, parent_summary, target, efficiency_thresholds) -> DecisionResult`.
- `decide(input) -> DecisionResult(status, accepted, violations, reward, pareto_eligible)`.

- [ ] **Step 1: Write failing tests**

```python
def test_reward_normalizes_against_parent_and_requires_energy():
    from harness4h3.remote.reward import RewardWeights, compute_reward
    result = compute_reward(
        quality=0.85,
        hardware={"latency_s": 18.0, "peak_memory_gb": 36.0, "energy_j": 90.0},
        baseline={"latency_s": 30.0, "peak_memory_gb": 42.0, "energy_j": 120.0},
        weights=RewardWeights(1.0, 0.2, 0.2, 0.2),
    )
    assert result.terms == {"Q": 0.85, "L": 0.6, "M": 36.0 / 42.0, "E": 0.75}
    assert result.reward == pytest.approx(0.85 - 0.2 * 0.6 - 0.2 * (36.0 / 42.0) - 0.2 * 0.75)
    assert compute_reward(0.85, {"latency_s": 18, "peak_memory_gb": 36}, {"latency_s": 30, "peak_memory_gb": 42, "energy_j": 120}, RewardWeights(1, 1, 1, 1)).reward is None


def test_critical_quality_failure_rejects_even_when_latency_improves():
    from harness4h3.remote.decision import AcceptanceInput, decide
    result = decide(AcceptanceInput(
        training_metrics={"optimizer_steps": 1, "gradient_norm": 1.0, "changed_trainable_tensors": 4, "unchanged_frozen_tensors": 531, "child_reloaded": True, "parent_sha256_before": "a" * 64, "parent_sha256_after": "a" * 64, "parent_sha256": "a" * 64, "child_sha256": "b" * 64},
        benchmark_summary={"quality_score": 0.4, "hardware": {"latency_s": 1, "peak_memory_gb": 1, "energy_j": 1}, "hard_gates": {"generation_valid": True, "decode_success": True, "no_critical_temporal_collapse": False}},
        parent_summary={"quality_score": 0.9, "hardware": {"latency_s": 10, "peak_memory_gb": 10, "energy_j": 10}},
        target={"max_quality_drop": 0.05}, efficiency_thresholds={"latency_s": 0.15},
    ))
    assert result.accepted is False
    assert "quality_gate" in result.violations
```

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/unit/test_remote_decision.py -q`

Expected: FAIL because the reward and decision modules are missing.

- [ ] **Step 3: Implement reward and gates**

Use exactly `Qn=Q`, `Ln=L/baseline_L`, `Mn=M/baseline_M`, `En=E/baseline_E`,
then `R=alpha*Qn-beta*Ln-gamma*Mn-delta*En`. Require positive finite
baselines. Gate training metrics for positive optimizer steps/gradient,
changed tensors, unchanged parent before/after hashes, different child hash,
and successful reload. Gate benchmark generation validity, decode success,
critical temporal state, quality drop against target, and at least one
efficiency reduction above threshold. Require numeric Q/L/M/E for
`research_grade=True`; otherwise return `evaluated_candidate` but not accepted.
Use target values as read-only inputs and do not allow a Controller plan to
change them.

- [ ] **Step 4: Run focused tests to verify pass**

Run: `pytest tests/unit/test_remote_decision.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/remote/reward.py harness4h3/remote/decision.py tests/unit/test_remote_decision.py
git commit -m "feat: gate remote H3 candidates with reward and evidence"
```

### Task 5: Add remote energy sampling to the existing benchmark

**Files:**
- Create: `harness4h3/remote/power.py`
- Modify: `harness4h3/benchmark/h3.py`
- Test: `tests/unit/test_remote_power.py`, `tests/unit/test_h3_benchmark.py`

**Interfaces:**
- `RemotePowerSampler(client, interval_s=1.0)` supports `start()`, `stop()`, and `summary() -> {"energy_j": float, "power_w_peak": float, "samples": int}`.
- `H3BenchmarkRunner.run(..., power_sampler=None)` keeps existing behavior when omitted and sets `HardwareMetrics.energy_j` from `power_sampler.summary()` when provided.

- [ ] **Step 1: Write failing tests**

```python
def test_power_sampler_integrates_sum_of_gpu_power(monkeypatch):
    from harness4h3.remote.power import integrate_power_samples
    assert integrate_power_samples([(0.0, 100.0), (1.0, 120.0), (2.0, 80.0)]) == pytest.approx(220.0)


def test_benchmark_without_power_sampler_keeps_energy_none(fake_runner):
    summary = fake_runner.run(state, tasks, target=None)
    assert summary.hardware.energy_j is None
```

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/unit/test_remote_power.py tests/unit/test_h3_benchmark.py -q`

Expected: the integration assertion or import fails because the sampler hook is absent.

- [ ] **Step 3: Implement sampler and benchmark hook**

Sample the fixed remote command `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits` through `SSHClient.run`; parse finite non-negative watts, sum all GPUs, and integrate trapezoid/left rectangles over monotonic timestamps. Sampling errors are recorded and do not fabricate zero energy. Start the sampler immediately before the benchmark loop and stop it in `finally`; put `energy_j` in `HardwareMetrics` only when at least two valid samples exist.

- [ ] **Step 4: Run focused tests to verify pass**

Run: `pytest tests/unit/test_remote_power.py tests/unit/test_h3_benchmark.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add harness4h3/remote/power.py harness4h3/benchmark/h3.py tests/unit/test_remote_power.py tests/unit/test_h3_benchmark.py
git commit -m "feat: measure remote GPU energy in H3 benchmarks"
```

### Task 6: Add the Linux BF16 H3 workflow and remote configuration

**Files:**
- Create: `examples/remote_linux_h3_workflow_api.json`
- Create: `configs/remote-l40-h3.yaml`
- Create: `configs/targets/l40x4_h3_example.yaml`
- Test: `tests/unit/test_remote_config.py`

**Interfaces:**
- The workflow exposes prompt node `139`, seed node `137`, steps node `132`, model node `135`, and saves the decoded video through node `92`.
- `configs/remote-l40-h3.yaml` declares `remote.host`, remote roots, `remote.comfyui_port`, local `runtime.experience_path`, reward weights, benchmark splits, and fixed efficiency thresholds.

- [ ] **Step 1: Write failing config tests**

```python
def test_remote_workflow_has_linux_model_and_benchmark_targets():
    workflow = json.loads(Path("examples/remote_linux_h3_workflow_api.json").read_text())
    assert workflow["135"]["inputs"]["unet_name"] == "minimax_h3_fl2va_bf16.safetensors"
    assert workflow["136"]["inputs"]["clip_name"] == "qwen3vl_32b_minimax_h3_bf16.safetensors"
    assert workflow["132"]["inputs"]["steps"] == 32
    assert workflow["139"]["inputs"]["length"] == 22


def test_remote_config_validates_host_roots_and_reward_weights():
    config = load_remote_campaign_config(Path("configs/remote-l40-h3.yaml"))
    assert config.remote.host == "Jiayu-intern"
    assert config.reward.alpha > 0
    assert config.remote.model_root == "/data/models/MiniMax-H3"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/unit/test_remote_config.py -q`

Expected: FAIL because the workflow/config loader does not exist.

- [ ] **Step 3: Implement the fixed workflow and config loader**

Build an API-format workflow using `VAELoader` for
`minimax_h3_video_vae_fp16.safetensors` and
`minimax_h3_audio_vae_fp32.safetensors`, `UNETLoader` for the BF16 model,
`CLIPLoader` for the BF16 Qwen encoder, `MiniMaxH3ImageToVideo` with 352×640
and length 22, `BasicScheduler`, `SamplerCustomAdvanced`, `VAEDecode`,
`CreateVideo`, and `SaveVideo`. Keep mutable fields limited to steps, CFG,
prompt, and seed. The loader validates absolute remote roots, `max_steps <=
32`, positive port, positive sampling interval, target path, and reward weights.
The benchmark runner must locate the configured `UNETLoader` by class type when
the legacy node `127` is absent, so the Linux workflow may use node `135` without
changing the frozen Controller contract.

- [ ] **Step 4: Run focused tests and verify pass**

Run: `pytest tests/unit/test_remote_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/remote_linux_h3_workflow_api.json configs/remote-l40-h3.yaml configs/targets/l40x4_h3_example.yaml tests/unit/test_remote_config.py
git commit -m "feat: configure Linux L40 H3 remote benchmark"
```

### Task 7: Implement the resumable remote H3 campaign

**Files:**
- Create: `research/experiments/remote_h3_closed_loop.py`
- Modify: `harness4h3/cli.py`
- Test: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- `RemoteCampaign(config, controller, ssh, evaluator_factory).run(resume=True) -> CampaignResult`.
- `CampaignResult.status`, `.current_model_id`, `.records`, `.report` are JSON serializable.
- CLI command: `python -m harness4h3 import-experience --remote-config configs/remote-l40-h3.yaml --output var/remote-h3/experience.jsonl --json`.
- CLI command: `python -m harness4h3 remote-campaign --remote-config configs/remote-l40-h3.yaml --target configs/targets/l40x4_h3_example.yaml --output-root var/remote-h3 --max-experiments 4 --json`.

- [ ] **Step 1: Write a fake SSH/ComfyUI integration test**

Define `FakeRemoteFixture` in the test with in-memory trainer results, score and
hardware maps, plus a `training_calls` counter. Define `build_campaign(root,
remote)` to construct a test `RemoteCampaign` with fake SSH discovery, a fake
benchmark runner, and the fixed Controller mock; the fixture must expose the
same result/evaluation interfaces as the production adapters.

```python
def test_campaign_imports_chain_evaluates_and_resumes_without_retraining(tmp_path):
    remote = FakeRemoteFixture(
        results=[m0005_training_result(), m0006_training_result()],
        benchmark_scores={"M0005": 0.86, "M0006": 0.85},
        benchmark_hardware={"M0005": {"latency_s": 30, "peak_memory_gb": 42, "energy_j": 120}, "M0006": {"latency_s": 18, "peak_memory_gb": 36, "energy_j": 90}},
    )
    first = build_campaign(tmp_path, remote).run(resume=False)
    assert first.records["M0006"].decision["status"] in {"accepted", "rejected"}
    assert first.report["metrics"]["M0006"]["Q"] == 0.85
    assert remote.training_calls == 0  # imported children are evaluated, not retrained
    second = build_campaign(tmp_path, remote).run(resume=True)
    assert second.report["imported_duplicates"] >= 2
    assert remote.training_calls == 0
```

- [ ] **Step 2: Run integration test to verify failure**

Run: `pytest tests/integration/test_remote_h3_closed_loop.py -q`

Expected: FAIL because `RemoteCampaign` and CLI commands do not exist.

- [ ] **Step 3: Implement import, model resolution, benchmark, and decision loop**

Implement these phases in order:

1. Instantiate `RemoteConfig`, `SSHClient`, `ExperienceStore`, `ExperimentStore`,
   `ModelStore`, and `ParetoArchive` under the local output root.
2. Import remote trainer results. Create `ModelCandidate` objects from valid
   `output_state` records; derive `M0000` from the parent checkpoint metadata
   and keep imported children ordered by parent ID/generation.
3. Evaluate any candidate lacking a complete evaluation. Open one
   `ComfyUITunnel`, instantiate `MiniMaxH3Adapter` against its local URL,
   `H3BenchmarkRunner` with the fixed Linux workflow and
   `RemotePowerSampler`, and run `sanity` first. Run `dev` and `heldout` only
   if sanity produces valid artifacts. Parent and child use identical tasks,
   seeds, CFG, sampler, and reset policy.
4. Compute the reward and decision against the immediate parent. Append an
   `ExperimentRecord` containing the Controller metadata, plan/operator,
   remote source hashes, evaluation summary, reward, decision, and Pareto
   front. Promote only accepted children in `ModelStore` and Pareto archive.
5. When no imported accepted child remains and budget allows, call the fixed
   Controller with recent normalized experience. Validate its plan against
   the existing `ValidationPipeline`. Allow only `recovery_finetune` and
   `step_distill`; write a small remote request under the campaign root and
   invoke the trusted remote worker config. Never execute a plan-provided
   command or path.
6. Import the returned result, verify the parent hash and child evidence,
   deploy a non-overwriting link, evaluate, decide, append, and checkpoint
   campaign state atomically after every experiment. On restart, source hash,
   experiment ID, and campaign state prevent duplicate successful training.

The CLI catches `RemoteError` as `remote_unavailable`, returns exit code 2 for
infrastructure failure, exit code 1 for a completed campaign with no accepted
target, and exit code 0 only for `accepted`/`completed` campaign status.

- [ ] **Step 4: Run integration test to verify pass**

Run: `pytest tests/integration/test_remote_h3_closed_loop.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research/experiments/remote_h3_closed_loop.py harness4h3/cli.py tests/integration/test_remote_h3_closed_loop.py
git commit -m "feat: add resumable remote H3 optimization campaign"
```

### Task 8: Document and verify the real server workflow

**Files:**
- Modify: `docs/quickstart.md`
- Modify: `docs/optimization-flow.md`
- Modify: `docs/experiment-schema.md`
- Test: `tests/integration/test_remote_h3_cli.py`

- [ ] **Step 1: Write CLI/documentation verification tests**

```python
def test_import_experience_cli_reports_status_without_remote_write(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "harness4h3", "import-experience", "--remote-config", "configs/remote-l40-h3.yaml", "--output", str(tmp_path / "experience.jsonl"), "--dry-run", "--json"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode in {0, 2}
    assert "training_only_unvalidated" in result.stdout or "remote_unavailable" in result.stdout
```

- [ ] **Step 2: Run the test and verify the documentation/CLI assertion fails**

Run: `pytest tests/integration/test_remote_h3_cli.py -q`

Expected: FAIL until the command and documentation are present.

- [ ] **Step 3: Document the actual server commands and claim boundary**

Add examples for read-only import, one sanity benchmark, full campaign, and
resume. Explicitly state that the existing `M0005–M0008` results are imported
as `training_only_unvalidated` until real Q/L/M/E evaluation is present, and
that `structural_proxy` quality does not support a semantic-quality paper
claim. Document where local JSONL metadata and remote artifacts live.

- [ ] **Step 4: Run the complete local test suite**

Run: `pytest -q`

Expected: PASS with all existing tests plus the new experience/remote tests.

- [ ] **Step 5: Run compile checks**

Run: `python -m compileall -q harness4h3 h3_training tools research`

Expected: exit code 0.

- [ ] **Step 6: Run read-only remote preflight and import**

Run:

```bash
python -m harness4h3 import-experience \
  --remote-config configs/remote-l40-h3.yaml \
  --output var/remote-h3/experience.jsonl --json
```

Expected: four existing training results are imported or reported as already
known; each has `training_only_unvalidated`, no child is promoted, and no
checkpoint bytes are transferred.

- [ ] **Step 7: Run a controlled real sanity benchmark**

Run:

```bash
python -m harness4h3 remote-campaign \
  --remote-config configs/remote-l40-h3.yaml \
  --target configs/targets/l40x4_h3_example.yaml \
  --output-root var/remote-h3 --max-experiments 1 --split sanity --json
```

Expected: remote ComfyUI returns a decodable candidate artifact, remote power
sampling produces numeric energy, and the record contains Q/L/M/E plus a
decision. If the Linux workflow or evaluator fails, record the stable failure
and keep the campaign resumable; do not mark the child accepted.

- [ ] **Step 8: Commit documentation and verification changes**

```bash
git add docs/quickstart.md docs/optimization-flow.md docs/experiment-schema.md tests/integration/test_remote_h3_cli.py
git commit -m "docs: describe remote H3 continuous optimization workflow"
```

## Plan self-review

- Experience import is covered by Tasks 1–3 and preserves the current
  training-only status.
- SSH execution, path safety, tunnel cleanup, and remote deployment are
  covered by Task 2 and Task 7.
- Q/L/M/E, reward normalization, missing-metric behavior, and acceptance gates
  are covered by Tasks 4–5 and consumed by Task 7.
- Existing stages are evaluated before retraining and resume is tested in
  Task 7.
- The fixed Controller boundary and unsupported-operator failure are explicit
  in Task 7 and the global constraints.
- The remote verification commands and claim boundary are covered by Task 8.
- No placeholder or deferred implementation language is used; every task has
  concrete files, interfaces, tests, commands, and expected outcomes.
