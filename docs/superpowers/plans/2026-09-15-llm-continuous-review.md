# LLM Continuous Experiment Review Implementation Plan

> **For agentic workers:** This plan is executed inline in the current task. It does not dispatch sub-agents or alter the existing remote campaign until the local tests pass.

**Goal:** Add a real structured LLM reviewer that reacts to campaign events and performs configurable heartbeat reviews during long remote H3 worker runs without changing evaluator authority or GPU isolation.

**Architecture:** Add a provider-neutral `ReviewDecision` contract and provider adapters alongside the existing ExperimentPlan API. `RemoteCampaign` owns a serialized review gate, builds compact read-only runtime context, and starts a daemon heartbeat monitor only while a trusted worker is running; immediate reviews are emitted at resource, worker, benchmark, evaluation, and failure boundaries. Reviewer actions are recorded and only `stop`/`replan` state flags are applied at safe campaign boundaries.

**Tech Stack:** Python 3.10+, dataclasses/typing, urllib JSON HTTP, existing vLLM OpenAI-compatible endpoint, SSHClient, ControllerEventStore, pytest.

## Global Constraints

- The reviewer cannot modify TargetProfile, evaluator hard gates, registered operators, or evidence records.
- The reviewer never requests training GPUs. The vLLM launcher chooses a feasible 1/2/4-GPU group from live free-memory and utilization samples; the ComfyUI lease reserves GPU0 only while it is needed, and the scheduler excludes all currently occupied or reserved cards.
- Reviewer calls are separate from ExperimentPlan budget accounting and are capped by `max_review_calls`.
- A reviewer failure is fail-safe: it records an unavailable/rejected event and leaves the active worker unchanged.
- The reviewer never emits shell commands, paths, or executable training configuration.
- Every accepted reviewer decision is recorded in `controller-events.jsonl` with provider, model, phase, trigger, evidence IDs, action, latency, and request ID.
- Existing uncommitted user changes must be preserved; only the files listed in each task may be changed.

---

### Task 1: Add the structured reviewer contract and vLLM adapter

**Files:**
- Create: `harness4h3/controller/reviewer.py`
- Modify: `harness4h3/controller/provider.py:17-32, 528-585`
- Test: `tests/unit/test_controller_reviewer.py`
- Test: `tests/unit/test_controller_providers.py`

**Interfaces:**
- `ReviewDecision.from_dict(raw: Mapping[str, Any]) -> ReviewDecision` validates the four allowed actions, finite confidence in `[0, 1]`, positive review delay, string evidence IDs, and no unknown fields.
- `review_json_schema() -> Mapping[str, Any]` returns a strict JSON schema with fields `action`, `reason`, `evidence_ids`, `confidence`, `next_review_after_s`, and `risks`.
- `review_prompt(request: Mapping[str, Any], schema: Mapping[str, Any]) -> str` creates a bounded prompt containing phase, trigger, runtime telemetry, recent events, target constraints, and GPU isolation rules.
- `OpenAICompatibleController.review(request: Mapping[str, Any]) -> ReviewDecision` performs one structured vLLM call using the existing endpoint/tunnel mechanism.

- [ ] **Step 1: Write failing contract tests**

```python
def test_review_decision_rejects_unknown_action_and_extra_fields():
    with pytest.raises(ValueError, match="action"):
        ReviewDecision.from_dict({"action": "approve"})
    valid = {
        "action": "continue",
        "reason": "telemetry is healthy",
        "evidence_ids": ["evt-1"],
        "confidence": 0.8,
        "next_review_after_s": 60,
        "risks": [],
    }
    with pytest.raises(ValueError, match="unknown"):
        ReviewDecision.from_dict({**valid, "extra": True})


def test_review_schema_is_strict_and_has_only_safe_actions():
    schema = review_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["action"]["enum"] == [
        "continue", "stop", "replan", "review_only"
    ]
```

Run: `.venv/bin/pytest -q tests/unit/test_controller_reviewer.py`

Expected: FAIL because the reviewer module does not exist.

- [ ] **Step 2: Implement the immutable decision and bounded prompt**

Implement `ReviewDecision` as a frozen dataclass. Validate the exact keys before coercion, reject booleans where numbers are expected, cap `reason` and each risk string at 2000 characters, and expose `to_dict()` for event logging. Implement `review_prompt()` with JSON-serialized bounded fields rather than raw command output.

- [ ] **Step 3: Add provider parsing and structured vLLM review**

Refactor the OpenAI-compatible provider’s private request helper into a generic `_chat_json(prompt, schema, schema_name, max_tokens)` while preserving the existing ExperimentPlan payload exactly. Add:

```python
def review(self, request: Mapping[str, Any]) -> ReviewDecision:
    schema = review_json_schema()
    raw = self._chat_json(review_prompt(request, schema), schema, "controller_review", 768)
    return ReviewDecision.from_dict(json.loads(self._content(raw)))
```

The request must use `temperature=0`, `stream=false`, `enable_thinking=false`, and the existing structured `response_format`. Convert transport failures to `ControllerUnavailableError` and malformed JSON/schema to `ControllerProviderError`.

- [ ] **Step 4: Add provider unit coverage**

Extend the existing HTTP test server to return a valid reviewer JSON response. Assert that the vLLM payload uses `json_schema.name == "controller_review"`, the action enum is present, the prompt contains `phase`, `trigger`, and the live GPU-isolation rule, and malformed reviewer JSON raises `ControllerProviderError`.

Run: `.venv/bin/pytest -q tests/unit/test_controller_reviewer.py tests/unit/test_controller_providers.py`

Expected: PASS.

---

### Task 2: Add configurable review runtime state and safe campaign hooks

**Files:**
- Modify: `harness4h3/remote/config.py:35-205`
- Modify: `configs/remote-l40-h3-rsi-overnight.yaml:1-65`
- Modify: `research/experiments/remote_h3_closed_loop.py:1-45, 161-212, 960-1095, 1200-1465`
- Test: `tests/unit/test_remote_config.py`
- Test: `tests/integration/test_remote_h3_closed_loop.py`

**Interfaces:**
- `RemoteCampaignConfig.review_interval_s: float` defaults to `60.0` and must be positive.
- `RemoteCampaignConfig.max_review_calls: int` defaults to `120` and must be positive.
- `RemoteCampaign._review_now(phase: str, trigger: str, payload: Mapping[str, Any]) -> Optional[ReviewDecision]` performs at most one serialized review request and appends review events.
- `RemoteCampaign._start_review_heartbeat(experiment_id: str, phase: str, context_factory: Callable[[], Mapping[str, Any]]) -> tuple[threading.Event, threading.Thread]` starts a daemon monitor that calls `_review_now` at the configured interval.
- `RemoteCampaign._collect_worker_telemetry(experiment_id: str) -> Mapping[str, Any]` uses fixed `nvidia-smi` queries and the known campaign root to return bounded GPU/process status; it never accepts a Controller-supplied command or path.

- [ ] **Step 1: Add failing config tests**

```python
def test_remote_review_defaults_are_safe(tmp_path):
    config = load_remote_campaign_config(Path("configs/remote-l40-h3-rsi-overnight.yaml"))
    assert config.review_interval_s == 60.0
    assert config.max_review_calls == 120


def test_remote_review_interval_must_be_positive(tmp_path):
    raw = yaml.safe_load(Path("configs/remote-l40-h3.yaml").read_text())
    raw["controller_review"] = {"interval_s": 0, "max_calls": 2}
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(RemoteConfigError, match="review interval"):
        load_remote_campaign_config(path)
```

Run: `.venv/bin/pytest -q tests/unit/test_remote_config.py`

Expected: FAIL because the config fields are not present.

- [ ] **Step 2: Parse review configuration and enable the overnight default**

Add `controller_review` parsing with `interval_s=60.0` and `max_calls=120`, validate both values, add the fields to `RemoteCampaignConfig`, and explicitly add this block to the overnight YAML:

```yaml
controller_review:
  interval_s: 60
  max_calls: 120
```

- [ ] **Step 3: Implement serialized review requests and bounded telemetry**

Add a `threading.Lock`, `review_calls`, `review_stop_requested`, and `review_replan_requested` to `RemoteCampaign`. `_review_now()` must:

1. return without calling the provider when `review_calls >= max_review_calls`;
2. append `controller_review_input` with trigger, phase, current model, experiment, and bounded context;
3. call `controller.review()` through the existing vLLM SSH port-forward path;
4. validate the returned `ReviewDecision` and map only `stop` and `replan` to in-memory flags;
5. append `controller_review_completed`, `controller_review_rejected`, or `controller_review_unavailable` with request ID and elapsed seconds.

For providers without `review`, append `controller_review_unavailable` and leave the worker unchanged. Telemetry failures become a bounded `telemetry_error` field, not a fake healthy value.

- [ ] **Step 4: Add event-triggered reviews at safe boundaries**

Call `_review_now()` after `resource_scheduled`, after `worker_started`, after `worker_completed`, after benchmark/evaluation completion, and in existing controller/worker failure paths. Review payloads must include only structured summaries and never raw shell commands beyond the already-redacted event tail.

If `stop` or `replan` is returned while a worker is active, record the decision but do not kill the worker. After the worker returns, `stop` prevents promotion/next plan and `replan` clears any pending plan before the next loop boundary.

- [ ] **Step 5: Start and stop heartbeat around the trusted worker**

Immediately after `worker_started`, create a `threading.Event` and daemon thread. The thread sleeps for `review_interval_s`, gathers telemetry, and calls `_review_now(phase="training", trigger="heartbeat", ...)`. In a `finally` block around `self.ssh.run(...)`, set the event and join with a bounded timeout before reading the result JSON. Never let a heartbeat exception escape into the worker result path.

- [ ] **Step 6: Apply flags only at campaign boundaries**

After worker import and after evaluation, append a review action summary to the report. If `review_stop_requested` is set, return a normal completed campaign result with `stop_reason="review_stop"`; if `review_replan_requested` is set, clear pending state and allow the next `run_loop` iteration to call the real Controller again. Existing evaluator decisions remain authoritative.

Run: `.venv/bin/pytest -q tests/unit/test_remote_config.py tests/integration/test_remote_h3_closed_loop.py`

Expected: PASS.

---

### Task 3: Add heartbeat/event integration tests and replay evidence

**Files:**
- Modify: `tests/integration/test_remote_h3_closed_loop.py`
- Modify: `tests/unit/test_controller_reviewer.py`
- Modify: `harness4h3/memory/observation.py` only if event redaction needs a narrowly scoped field rule

**Interfaces:**
- Fake reviewer records every request and returns a `ReviewDecision`.
- Fake SSH exposes deterministic `nvidia-smi` output and a worker command that completes after a short test-controlled delay.

- [ ] **Step 1: Test immediate event reviews**

Use a fake controller with `review()` and assert `resource_scheduled`, `worker_started`, `worker_completed`, and `evaluation_completed` each produce at most one reviewer request with the correct phase/trigger.

- [ ] **Step 2: Test heartbeat during a long worker**

Set `review_interval_s=0.01`, run a fake worker that blocks for `0.04` seconds, and assert at least one `heartbeat` request and one `controller_review_completed` event are recorded. Set `max_review_calls=1` and assert a second heartbeat is skipped without changing worker execution.

- [ ] **Step 3: Test fail-safe stop and replan actions**

Return `stop` and verify the worker still completes, the parent remains unchanged, no evaluator result is rewritten, and the campaign report records `review_stop`. Return `replan` and verify the pending plan is cleared only at the next safe boundary.

- [ ] **Step 4: Test provider failure and event replay**

Make `review()` raise `ControllerUnavailableError`; assert no success decision is recorded and the campaign continues with the existing safety behavior. Reload `ControllerEventStore` and assert review input/completed/unavailable events are readable as append-only records.

Run: `.venv/bin/pytest -q tests/unit/test_controller_reviewer.py tests/integration/test_remote_h3_closed_loop.py`

Expected: PASS.

---

### Task 4: Run the real dynamically placed vLLM reviewer smoke test

**Files:**
- Verify only: `var/remote-h3-controller-20260914/controller-events.jsonl`, `var/remote-h3-controller-20260914/overnight-result.json`
- No source changes are expected in this task.

- [ ] **Step 1: Run the focused local suite**

Run: `.venv/bin/pytest -q tests/unit/test_controller_reviewer.py tests/unit/test_controller_providers.py tests/unit/test_remote_config.py tests/integration/test_remote_h3_closed_loop.py`

Expected: all selected tests pass.

- [ ] **Step 2: Verify the remote single-card endpoint**

Run: `ssh Jiayu-intern "curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models"`

Expected: `qwen3.5-controller` is served; the launcher-selected GPU group is visible in its log, with no preemption of ComfyUI or unrelated jobs.

- [ ] **Step 3: Run a bounded real vLLM campaign**

Run:

```bash
.venv/bin/python tools/run_overnight_controller.py \
  --config configs/remote-l40-h3-rsi-overnight.yaml \
  --output var/remote-h3-controller-20260914 \
  --controller vllm \
  --max-iterations 1 \
  --resource-poll-interval-s 30
```

Expected: the event stream contains `controller_review_input` and either `controller_review_completed` or a recorded unavailable/rejected event, while the existing real worker/evaluator evidence remains intact. A worker request must contain only the scheduler's live allocation and must not overlap a still-occupied controller/ComfyUI card.

- [ ] **Step 4: Inspect the final evidence**

Run: `rg -n 'controller_review|worker_started|worker_completed|evaluation_completed' var/remote-h3-controller-20260914/controller-events.jsonl`

Expected: review events include provider `vllm`, model `qwen3.5-controller`, a real decision or explicit failure, and no fabricated evaluator acceptance.
