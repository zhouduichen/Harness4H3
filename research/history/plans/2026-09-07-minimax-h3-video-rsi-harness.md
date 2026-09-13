# MiniMax H3 Video RSI Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, testable Harness4H3 v1 that treats ComfyUI MiniMax-H3 as a frozen video backend and promotes isolated prompt, context, or workflow candidates only through an external evaluator.

**Architecture:** A deterministic Python state machine renders an API-format workflow, calls ComfyUI, downloads the artifact, records an append-only trajectory, and invokes an evaluator through a JSON boundary. A rule-based evolution controller diagnoses recurring failures, creates one-mutation candidates, runs sanity/replay/dev gates, and atomically promotes only candidates that beat their parent.

**Tech Stack:** Python 3.9+, standard library HTTP/subprocess/dataclasses, PyYAML, optional OpenCV media evaluation, pytest.

## Global Constraints

- MiniMax-H3 model weights, model files, and ComfyUI implementation remain frozen and read-only.
- A running candidate never modifies itself; mutations create immutable child candidates.
- Each child changes exactly one of prompt, context policy, or an allow-listed workflow setting.
- The evaluator is a separate interface/process and is the only source of promotion scores.
- V1 uses JSONL and ordinary files; no training, database, vector store, effect model, multi-agent system, workflow DSL, or source-code mutation.
- Held-out tasks never drive diagnosis or promotion.
- Offline tests do not require ComfyUI, a GPU, model files, ffmpeg, or OpenCV.

---

### Task 1: Package, configuration, and reuse audit

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `REUSE_MATRIX.md`
- Create: `harness4h3/__init__.py`
- Create: `harness4h3/config.py`
- Create: `configs/default.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: YAML files with `backend`, `workflow`, `runtime`, `evaluator`, and `evolution` mappings.
- Produces: `AppConfig`, `load_config(path)`, and `validate_config(config)`.

- [ ] **Step 1: Write configuration tests**

```python
def test_load_config_resolves_paths_relative_to_config(tmp_path):
    path = write_config(tmp_path, workflow="workflow.json")
    assert load_config(path).workflow.template == tmp_path / "workflow.json"

def test_config_rejects_non_allowlisted_workflow_key(tmp_path):
    path = write_config(tmp_path, mutable_workflow_keys=["model_name"])
    with pytest.raises(ConfigError, match="mutable workflow key"):
        load_config(path)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL because `harness4h3.config` does not exist.

- [ ] **Step 3: Implement typed config loading and validation**

```python
@dataclass(frozen=True)
class AppConfig:
    backend: BackendConfig
    workflow: WorkflowConfig
    runtime: RuntimeConfig
    evaluator: EvaluatorConfig
    evolution: EvolutionConfig
    paths: PathsConfig

def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _parse_config(raw, base=path.parent)
    validate_config(config)
    return config
```

Validation requires a non-empty backend URL, an existing API workflow, positive timeouts, known node/input targets for prompt and seed, workflow mutation keys drawn from `steps`, `cfg`, and declared stability switches, `min_delta >= 0`, and distinct writable data paths.

- [ ] **Step 4: Write the reuse matrix**

Document that `/prompt`, `/history/<id>`, and `/view` are reused conceptually from the existing MinMax-H3 deployment while Windows paths, SSH telemetry, benchmark-specific model names, and manual score columns are not copied.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS.

```bash
git add pyproject.toml .gitignore REUSE_MATRIX.md harness4h3 configs tests/test_config.py
git commit -m "feat: add Harness4H3 configuration"
```

### Task 2: ComfyUI adapter, context, state, tools, and trajectory

**Files:**
- Create: `harness4h3/model/minimax_h3.py`
- Create: `harness4h3/harness/context.py`
- Create: `harness4h3/harness/state.py`
- Create: `harness4h3/harness/loop.py`
- Create: `harness4h3/tools/registry.py`
- Create: `harness4h3/memory/trajectory.py`
- Test: `tests/test_backend.py`
- Test: `tests/test_context.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `AppConfig`, task dictionaries, candidate policy dictionaries, and an API workflow template.
- Produces: `MiniMaxH3Adapter.run(workflow, output_dir) -> BackendResult`, `build_execution_request(template, task, candidate, targets)`, `HarnessRunner.run_task(task, candidate) -> Trajectory`, and `TrajectoryStore.append/read`.

- [ ] **Step 1: Write backend and workflow-rendering tests**

```python
def test_adapter_submits_polls_and_downloads(fake_comfyui, tmp_path):
    result = MiniMaxH3Adapter(fake_comfyui.url, poll_interval=0).run(WORKFLOW, tmp_path)
    assert result.prompt_id == "prompt-1"
    assert result.artifacts[0].read_bytes() == b"video"

def test_context_changes_only_allowlisted_target():
    rendered = build_execution_request(TEMPLATE, TASK, CANDIDATE, TARGETS).workflow
    assert rendered["131"]["inputs"]["prompt"].startswith("stable subject")
    assert rendered["129"]["inputs"]["noise_seed"] == 42
    assert TEMPLATE["131"]["inputs"]["prompt"] == "original"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_backend.py tests/test_context.py -q`
Expected: FAIL because adapter and context modules do not exist.

- [ ] **Step 3: Implement the backend and context boundary**

```python
@dataclass(frozen=True)
class BackendResult:
    prompt_id: str
    artifacts: Sequence[Path]
    history: Mapping[str, Any]
    wall_time_s: float

class MiniMaxH3Adapter:
    def run(self, workflow: Mapping[str, Any], output_dir: Path) -> BackendResult:
        prompt_id = self.submit(workflow)
        history = self.wait(prompt_id)
        return self.download(prompt_id, history, output_dir)
```

The adapter uses bounded timeouts, GET retries only, quoted prompt IDs and output query values, sanitized filenames, and a stable `BackendError.failure_type`. Context rendering deep-copies the template and applies prompt, seed, and one allow-listed workflow patch.

- [ ] **Step 4: Write loop and trajectory tests**

```python
def test_loop_records_success_and_external_score(tmp_path, fake_backend, fake_evaluator):
    trajectory = make_runner(tmp_path, fake_backend, fake_evaluator).run_task(TASK, H0)
    assert trajectory.score == 0.8
    assert list(TrajectoryStore(tmp_path / "runs.jsonl").read())[0].task_id == TASK["id"]

def test_loop_records_backend_failure(tmp_path, failing_backend, fake_evaluator):
    trajectory = make_runner(tmp_path, failing_backend, fake_evaluator).run_task(TASK, H0)
    assert trajectory.failure_type == "backend_execution"
```

- [ ] **Step 5: Implement state, tool registry, loop, and JSONL store**

```python
@dataclass
class TaskState:
    task_id: str
    goal: str
    step: int = 0
    recent_history: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    done: bool = False

class HarnessRunner:
    def run_task(self, task: Task, candidate: Candidate) -> Trajectory:
        started = time.monotonic()
        request = build_execution_request(self.template, task, candidate, self.targets)
        try:
            backend_result = self.backend.run(request.workflow, self.output_dir / task.id)
            evaluation = self.evaluator.evaluate(
                evaluation_request(task, backend_result, time.monotonic() - started)
            )
            trajectory = make_trajectory(task, candidate, request, backend_result, evaluation)
        except BackendError as exc:
            trajectory = make_failed_trajectory(task, candidate, request, exc, time.monotonic() - started)
        self.trajectories.append(trajectory)
        return trajectory
```

JSONL serialization is stable, one record per line, flushes and fsyncs before return, and redacts keys matching token, password, secret, authorization, or api_key.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_backend.py tests/test_context.py tests/test_loop.py -q`
Expected: PASS.

```bash
git add harness4h3 tests
git commit -m "feat: run H3 tasks and record trajectories"
```

### Task 3: Independent evaluator

**Files:**
- Create: `harness4h3/evaluator/evaluator.py`
- Create: `harness4h3/evaluator/worker.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: evaluator request JSON containing task expectations, artifact paths, backend status, and wall time.
- Produces: `EvaluationResult(score, metrics, critical_regression, failure_type)` from an isolated subprocess or configured external command.

- [ ] **Step 1: Write protocol tests**

```python
def test_subprocess_evaluator_is_score_authority(tmp_path):
    result = SubprocessEvaluator(command=worker_command()).evaluate(valid_request(tmp_path))
    assert 0.0 <= result.score <= 1.0
    assert result.metrics["artifact_exists"] == 1.0

def test_invalid_evaluator_output_fails_closed(tmp_path):
    evaluator = SubprocessEvaluator([sys.executable, "-c", "print('not-json')"])
    with pytest.raises(EvaluatorError, match="valid JSON"):
        evaluator.evaluate(valid_request(tmp_path))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_evaluator.py -q`
Expected: FAIL because evaluator modules do not exist.

- [ ] **Step 3: Implement JSON subprocess protocol and deterministic worker**

```python
@dataclass(frozen=True)
class EvaluationResult:
    score: float
    metrics: Mapping[str, float]
    critical_regression: bool = False
    failure_type: Optional[str] = None

class SubprocessEvaluator:
    def evaluate(self, request: Mapping[str, Any]) -> EvaluationResult:
        completed = subprocess.run(
            self.command,
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise EvaluatorError(completed.stderr.strip() or "evaluator failed")
        return validate_result(json.loads(completed.stdout))
```

The worker always scores completion and artifact existence/non-empty size. When OpenCV is installed it also scores decodability, expected width/height/frame count, mean luma, black-frame ratio, and mean adjacent-frame difference. Missing optional media support is reported as an unavailable metric and never fabricated.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_evaluator.py -q`
Expected: PASS.

```bash
git add harness4h3/evaluator tests/test_evaluator.py pyproject.toml
git commit -m "feat: add independent video evaluator"
```

### Task 4: Candidate archive and evolution gates

**Files:**
- Create: `harness4h3/archive/store.py`
- Create: `harness4h3/self_improve/evolve.py`
- Test: `tests/test_archive.py`
- Test: `tests/test_evolve.py`

**Interfaces:**
- Consumes: immutable candidate JSON, parent trajectories, task splits, and a callable batch runner.
- Produces: `CandidateStore.create/get/active/promote/lineage`, `diagnose`, `propose_mutation`, and `EvolutionController.evolve`.

- [ ] **Step 1: Write archive and mutation tests**

```python
def test_candidate_is_immutable_and_lineage_is_ordered(tmp_path):
    store = CandidateStore(tmp_path)
    store.create(H0)
    store.create(H1)
    with pytest.raises(CandidateExists):
        store.create(H1)
    assert [c.id for c in store.lineage()] == ["H0", "H1"]

def test_diagnosis_requires_recurring_failure():
    assert diagnose([failed("low_luma"), failed("low_luma")]).failure_type == "low_luma"
    assert diagnose([failed("low_luma"), passed()]) is None
```

- [ ] **Step 2: Write promotion-gate tests**

```python
def test_evolution_promotes_only_better_candidate(tmp_path):
    outcome = controller(tmp_path, parent_score=.5, child_score=.7).evolve()
    assert outcome.status == "promoted"
    assert CandidateStore(tmp_path).active_id == outcome.candidate.id

def test_evolution_drops_critical_regression(tmp_path):
    outcome = controller(tmp_path, parent_score=.8, child_score=.9, critical=True).evolve()
    assert outcome.status == "dropped"
    assert CandidateStore(tmp_path).active_id == "H0"
```

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_archive.py tests/test_evolve.py -q`
Expected: FAIL because archive and evolution modules do not exist.

- [ ] **Step 4: Implement immutable archive, diagnosis, catalog mutations, and gates**

```python
def should_promote(parent_score, child_score, sanity_passed, critical, regressions, cfg):
    return (
        sanity_passed
        and not critical
        and regressions <= cfg.max_regressions
        and child_score > parent_score + cfg.min_delta
    )
```

Diagnosis uses only parent replay/dev trajectories and requires `recurrence_threshold` matching failures. The catalog maps low luma, decode/artifact failures, instability, timeout/cost, and generic low score to one prompt, context, or workflow mutation. Applying a mutation validates that exactly one major category changed. Candidate creation uses exclusive file creation; active pointer promotion uses `os.replace`.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_archive.py tests/test_evolve.py -q`
Expected: PASS.

```bash
git add harness4h3/archive harness4h3/self_improve tests
git commit -m "feat: evolve and gate harness candidates"
```

### Task 5: CLI, example assets, integration tests, and operator documentation

**Files:**
- Create: `harness4h3/cli.py`
- Create: `harness4h3/__main__.py`
- Create: `examples/tasks.yaml`
- Create: `examples/workflow_api.json`
- Create: `README.md`
- Create: `tests/test_cli.py`
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: config path plus CLI arguments.
- Produces: `harness4h3 validate-config`, `run`, `evaluate`, `evolve`, and `lineage`, each with human and `--json` output.

- [ ] **Step 1: Write CLI and fake-backend integration tests**

```python
def test_validate_config_is_offline(project_fixture):
    result = run_cli(project_fixture, "validate-config", "--json")
    assert result.returncode == 0
    assert json.loads(result.stdout)["valid"] is True

def test_fake_backend_reaches_promote_and_drop_paths(project_fixture):
    run_cli(project_fixture, "run", "--split", "dev")
    promoted = run_cli(project_fixture, "evolve", "--json")
    assert json.loads(promoted.stdout)["status"] in {"promoted", "dropped", "no_mutation"}
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_cli.py tests/test_integration.py -q`
Expected: FAIL because CLI and examples do not exist.

- [ ] **Step 3: Implement CLI and safe bootstrap behavior**

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except Harness4H3Error as exc:
        emit_error(exc, json_mode=args.json)
        return 2
```

On first use, `run` creates H0 from the configured baseline policy if the archive is empty. `evolve` never includes held-out tasks in its decision set. `evaluate` writes a separate reevaluation JSONL. `lineage` is read-only.

- [ ] **Step 4: Add runnable examples and README**

The example workflow is valid API-format JSON with generic MiniMax-H3 node IDs matching the supplied default config. README includes installation, configuration, the evaluator trust boundary, the fake/offline test command, real ComfyUI smoke command, candidate lifecycle, output layout, failure recovery, and explicit V1 non-goals.

- [ ] **Step 5: Run full verification**

Run: `python -m pytest -q`
Expected: all tests PASS.

Run: `python -m harness4h3 --config configs/default.yaml validate-config --json`
Expected: exit 0 and JSON with `"valid": true`.

Run: `python -m compileall -q harness4h3 tests`
Expected: exit 0 with no output.

- [ ] **Step 6: Commit and push**

```bash
git add README.md examples harness4h3 tests
git commit -m "feat: deliver runnable Harness4H3 v1"
git push -u origin main
```
