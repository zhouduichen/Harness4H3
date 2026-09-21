# Campaign Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared, fail-closed Campaign Control Plane that gives StudentCampaign and the legacy H3 loop immutable verification identity, bounded multi-candidate review, structured failures, two-stage acceptance, and an auditable event trace.

**Architecture:** Add a focused `harness4h3/campaign` domain package. It owns immutable campaign identity, event persistence, capability snapshots, candidate batches, review contracts, failure attribution, and gates; existing Student, operator, archive, and remote modules remain the execution domains and connect through explicit adapters.

**Tech Stack:** Python 3.9+, frozen dataclasses, standard-library JSONL/atomic file I/O, existing PyYAML/PyTorch/pytest stack, no new runtime dependency.

## Global Constraints

- TargetProfile, Verifier Bank, dataset manifest, evaluation recipe, controller/critic/evaluator identities, prompt version, and capability snapshot are immutable for one campaign.
- A changed verification base starts a new campaign and cannot append to or compare directly with the old campaign.
- Controller proposals are JSON/domain data only; no arbitrary Python, shell, remote command, evaluator rule, or repository path can enter the action space.
- V0–V1 deterministic validation runs before review or execution; V2 training validity is required before promotion.
- Hard constraints are evaluated before Pareto or scalar reward comparison; scalar reward never overrides a hard violation.
- `promotable` and `target_satisfied` are distinct outcomes.
- Controller, Critical agent, and evaluator identities must be pairwise distinct and persisted.
- Existing full regression must remain at least `557 passed, 2 skipped`; new tests must run without CUDA, SSH, remote LLM, or real H3 weights.
- Each task ends with a focused test command and a separate commit.

---

## File map

Create:

- `harness4h3/campaign/__init__.py`: public exports only.
- `harness4h3/campaign/base.py`: canonical JSON, actor identity, immutable CampaignBase, base mismatch errors.
- `harness4h3/campaign/events.py`: DecisionEvent and append-only trace with sequence/base validation.
- `harness4h3/campaign/capabilities.py`: real registered capability snapshot and action-space filtering.
- `harness4h3/campaign/proposals.py`: CandidateEnvelope, ProposalBatch, mutation whitelist, deterministic V0–V1 validation.
- `harness4h3/campaign/reviews.py`: Advocate/Critical/Revision contracts and bounded review pipeline.
- `harness4h3/campaign/failures.py`: typed FailureReport and deterministic-first attribution.
- `harness4h3/campaign/gates.py`: MetricEvidence, feasibility gate, Pareto comparison, stop reasons.
- `harness4h3/campaign/adapters.py`: Student/H3 adapter protocols and identity mapping.
- `tests/unit/test_campaign_base.py`, `tests/unit/test_campaign_events.py`, `tests/unit/test_campaign_capabilities.py`, `tests/unit/test_campaign_proposals.py`, `tests/unit/test_campaign_reviews.py`, `tests/unit/test_campaign_failures.py`, `tests/unit/test_campaign_gates.py`: focused contract tests.
- `tests/unit/campaign_fixtures.py`: shared base, candidate, capability, and evidence builders used by the focused tests.
- `tests/integration/test_campaign_control_plane.py`: scripted multi-candidate, review, failure, gate, and resume integration.

Modify:

- `harness4h3/student/campaign.py`: consume the shared contracts, support a proposal batch, record unified events, and stop only on the shared campaign semantics.
- `harness4h3/controller/loop.py`: emit the shared identity/event envelope through an adapter without changing existing operator execution.
- `harness4h3/student/__init__.py`: export any public batch/provider compatibility types.
- `README.md`: document the new control-plane boundary and the focused test command.

---

### Task 1: Immutable CampaignBase and append-only DecisionTrace

**Files:**

- Create: `harness4h3/campaign/base.py`
- Create: `harness4h3/campaign/events.py`
- Create: `harness4h3/campaign/__init__.py`
- Create: `tests/unit/test_campaign_base.py`
- Create: `tests/unit/test_campaign_events.py`
- Create: `tests/unit/campaign_fixtures.py`

**Interfaces:**

- Produces `canonical_json(value: Any) -> str`.
- Produces `canonical_digest(value: Any) -> str`.
- Produces frozen `ActorIdentity(provider: str, model: str, version: str)`.
- Produces frozen `CampaignBase(...)` with `payload() -> Mapping[str, Any]`, `digest -> str`, `to_dict() -> Mapping[str, Any]`, and `assert_event_base(base_digest: str) -> None`.
- Produces frozen `DecisionEvent` with `from_dict`, `to_dict`, and `event_id`.
- Produces `DecisionTrace(path: Path, base: CampaignBase)` with `append(event_type, *, round_id, experiment_id, candidate_id, parent_candidate_id, actor, payload, evidence_ids) -> DecisionEvent`, `read() -> tuple[DecisionEvent, ...]`, and `verify() -> None`.
- Raises `CampaignBaseError` for malformed identities/base data and `TraceIntegrityError` for mismatched digest, sequence, or event payload.
- Test fixture `tests/unit/campaign_fixtures.py` initially provides `make_base(**overrides) -> CampaignBase`; later tasks extend the same fixture with candidate, capability, and evidence builders.

- [ ] **Step 1: Write failing base and trace tests**

~~~python
def test_campaign_base_digest_is_stable_and_changes_when_target_changes():
    base = make_base()
    same = CampaignBase.from_dict(base.to_dict())
    changed = replace(base, target_profile={"id": "mobile-v2"})
    assert same.digest == base.digest
    assert changed.digest != base.digest

def test_actor_identities_must_be_pairwise_distinct():
    with pytest.raises(CampaignBaseError, match="pairwise"):
        make_base(critic_identity=make_base().controller_identity)

def test_event_trace_rejects_wrong_base_and_non_monotonic_sequence(tmp_path):
    base = make_base()
    trace = DecisionTrace(tmp_path / "events.jsonl", base)
    trace.append("campaign.created", round_id="R0001", experiment_id=None,
                 candidate_id=None, parent_candidate_id=None,
                 actor=base.controller_identity, payload={"ok": True},
                 evidence_ids=())
    raw = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])
    raw["base_digest"] = "sha256:wrong"
    (tmp_path / "events.jsonl").write_text(json.dumps(raw) + "\n")
    with pytest.raises(TraceIntegrityError, match="base"):
        trace.verify()

def make_base(**overrides):
    values = {
        "campaign_id": "camp_test_0001",
        "target_profile": {"id": "mobile", "constraints": {"max_latency_s": 3.0}},
        "target_profile_hash": "sha256:target",
        "verifier_bank": {"version": "v1"},
        "verifier_bank_hash": "sha256:verifier",
        "dataset_manifest_hash": "sha256:dataset",
        "evaluation_recipe_hash": "sha256:evaluation",
        "controller_identity": ActorIdentity("controller", "model-a", "1"),
        "critic_identity": ActorIdentity("critic", "model-b", "1"),
        "evaluator_identity": ActorIdentity("evaluator", "fixed-v1", "1"),
        "prompt_version": "prompt-v1",
        "capability_snapshot": {"version": "cap-v1"},
    }
    values.update(overrides)
    return CampaignBase(**values)
~~~

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_base.py tests/unit/test_campaign_events.py
~~~

Expected: collection fails with `ModuleNotFoundError: No module named 'harness4h3.campaign'`.

- [ ] **Step 3: Implement canonical identity and immutable base**

Implement the following concrete behavior:

~~~python
def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)

def canonical_digest(value):
    return "sha256:" + hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()

@dataclass(frozen=True)
class ActorIdentity:
    provider: str
    model: str
    version: str
~~~

`CampaignBase.__post_init__` must reject empty IDs, non-mapping payloads, NaN values, missing four base hashes, and non-pairwise-distinct actor identities. Its digest must be calculated from a payload that excludes any stored digest field. `from_dict` must deep-copy mappings and tuples so callers cannot mutate the base through an alias.

- [ ] **Step 4: Implement durable trace append and replay**

Each event must contain an integer sequence starting at 1, the CampaignBase digest, a payload digest, the full ID chain, actor identity, and a UTC timestamp. `append` must create the parent directory, write one JSON line, flush, call `os.fsync`, and return the event. `verify` must reject blank/malformed lines, wrong base digest, sequence gaps, duplicate event IDs, wrong payload digest, and unknown event types. `read` must call `verify` before returning events.

- [ ] **Step 5: Run focused tests and commit**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_base.py tests/unit/test_campaign_events.py
~~~

Expected: PASS.

Commit:

~~~bash
git add harness4h3/campaign tests/unit/test_campaign_base.py tests/unit/test_campaign_events.py
git commit -m "feat: add immutable campaign base and decision trace"
~~~

---

### Task 2: Capability snapshot and multi-candidate deterministic validation

**Files:**

- Create: `harness4h3/campaign/capabilities.py`
- Create: `harness4h3/campaign/proposals.py`
- Modify: `tests/unit/campaign_fixtures.py`
- Create: `tests/unit/test_campaign_capabilities.py`
- Create: `tests/unit/test_campaign_proposals.py`

**Interfaces:**

- Produces frozen `Capability(name, category, backend, schema, evidence_level, available, reason)`.
- Produces frozen `CapabilitySnapshot(capabilities: tuple[Capability, ...], digest: str)` with `available_names() -> tuple[str, ...]`.
- Produces `CapabilityRegistry.from_operator_registry(registry: OperatorRegistry, backend_status: Mapping[str, Mapping[str, Any]]) -> CapabilitySnapshot`.
- Produces frozen `CandidateEnvelope` with the fields in the spec and `digest(base: CampaignBase) -> str`.
- Produces frozen `ProposalBatch(batch_id, round_id, diagnosis, parent_selection_evidence_ids, candidates)`.
- Produces `ProposalValidationReport(errors: tuple[str, ...], candidate_errors: Mapping[str, tuple[str, ...]])` and `validate_batch(batch, *, base, snapshot, parent_ids, max_candidates=5) -> ProposalValidationReport`.
- Produces `ProposalValidationError` only for malformed data; semantic ineligibility is returned in the report.
- Extends `tests/unit/campaign_fixtures.py` with `valid_candidate`, `valid_batch`, and `valid_snapshot` helpers whose defaults use parent `M0000`, generation `1`, and three distinct candidate IDs.

- [ ] **Step 1: Write tests for capability filtering and candidate validation**

~~~python
def test_snapshot_exposes_only_available_registered_capabilities():
    registry = build_model_evolution_registry()
    snapshot = CapabilityRegistry.from_operator_registry(
        registry,
        {"dmd2": {"available": False, "reason": "backend_not_installed"}},
    )
    assert "distill" in snapshot.available_names()
    assert "dmd2" not in snapshot.available_names()
    assert snapshot.capabilities[0].evidence_level == "V2"

def test_batch_rejects_duplicate_ids_unknown_mutations_and_invalid_parent():
    batch = valid_batch(
        candidates=(
            valid_candidate("C0001", "M0000", ("training.method",)),
            valid_candidate("C0001", "M9999", ("repository.file",)),
        )
    )
    report = validate_batch(batch, base=make_base(),
                            snapshot=valid_snapshot(),
                            parent_ids={"M0000"}, max_candidates=5)
    assert "duplicate candidate_id" in report.errors
    assert "parent_candidate_id is unknown" in report.candidate_errors["C0001"]
    assert "mutation field is not registered" in report.candidate_errors["C0001"]

def test_batch_of_three_candidates_is_accepted_before_review():
    report = validate_batch(valid_batch(count=3), base=make_base(),
                            snapshot=valid_snapshot(),
                            parent_ids={"M0000"}, max_candidates=5)
    assert report.ok

def valid_candidate(candidate_id="C0001", parent_candidate_id="M0000", mutation_fields=("training.method",)):
    return CandidateEnvelope(
        candidate_id=candidate_id, parent_candidate_id=parent_candidate_id,
        generation=1, experiment_id="exp-" + candidate_id,
        proposal_digest="sha256:" + candidate_id,
        mutation_fields=tuple(mutation_fields), architecture={"family": "video_latent_dit"},
        training_recipe={"method": "velocity_distill"}, deployment_recipe={"precision": "bf16"},
        provenance={"source": "test"}, predicted_metric_delta={"quality": 0.01},
    )

def valid_batch(count=3, candidates=None):
    items = tuple(candidates or (valid_candidate("C%04d" % (index + 1)) for index in range(count)))
    return ProposalBatch("batch-test", "R0001", "latency bottleneck", ("obs-1",), items)

def valid_snapshot():
    return CapabilitySnapshot((
        Capability("training.velocity_distill", "training", "test", {}, "V2", True, ""),
        Capability("training.dmd2", "training", "test", {}, "V2", True, ""),
    ), "sha256:capability-test")
~~~

- [ ] **Step 2: Run tests to verify they fail**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_capabilities.py tests/unit/test_campaign_proposals.py
~~~

Expected: collection fails because the capability and proposal contracts do not exist.

- [ ] **Step 3: Implement capability snapshot from the real registry**

Use `OperatorRegistry.visible()` as the source of operator name, description, and schema. Map each registered operator to an evidence level: structural operators are V1, training/distillation/quantization operators are V2, and runtime/evaluation capabilities are V3 or V4 only when a trusted backend status explicitly marks them available. Never mark an operator available merely because it is registered. Include the backend status and reason in the digest.

- [ ] **Step 4: Implement ProposalBatch and deterministic report**

Validate exact candidate count bounds, unique batch/candidate/experiment IDs, parent existence, non-negative generation, allowed mutation fields, available capability names, finite predicted deltas, and base/snapshot digest binding. Reject arbitrary paths, commands, code strings, or evaluator definitions. Keep candidate-level errors separate so one invalid candidate can be rejected without discarding valid siblings.

- [ ] **Step 5: Run tests and commit**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_capabilities.py tests/unit/test_campaign_proposals.py
~~~

Expected: PASS.

Commit:

~~~bash
git add harness4h3/campaign/capabilities.py harness4h3/campaign/proposals.py tests/unit/test_campaign_capabilities.py tests/unit/test_campaign_proposals.py
git commit -m "feat: validate capability-bound candidate batches"
~~~

---

### Task 3: Bounded Advocate/Critical review contracts

**Files:**

- Create: `harness4h3/campaign/reviews.py`
- Create: `tests/unit/test_campaign_reviews.py`

**Interfaces:**

- Produces `ReviewAgent` protocol with `identity: ActorIdentity` and `review(request: Mapping[str, Any]) -> Mapping[str, Any]`.
- Produces frozen `AdvocateReport`, `CriticalReport`, and `RevisionRecord`, each with strict `from_dict/to_dict`.
- Produces frozen `CandidateReview(candidate_id, advocate, critical_rounds, revision, final_critical, approved, rejection_reasons)`.
- Produces `ReviewPipeline(advocate, critical, modifier, base, max_rounds)` with `review(candidate, context) -> CandidateReview`.
- Raises `ReviewContractError` for malformed agent output and `ReviewIdentityError` if the review actors are not distinct from the controller/evaluator identities.

- [ ] **Step 1: Write tests for strict reports and bounded review**

~~~python
def test_critical_agent_cannot_change_base_or_hard_constraints():
    pipeline = ReviewPipeline(
        advocate=ScriptedAdvocate(valid_advocate()),
        critical=ScriptedCritical({"required_revisions": ["training.method"]}),
        modifier=ScriptedModifier(valid_revision()),
        base=make_base(),
        max_rounds=1,
    )
    result = pipeline.review(valid_candidate(), {"hard_constraints": {"max_latency_s": 3}})
    assert result.approved is True
    assert result.revision.base_digest == make_base().digest
    assert result.revision.changed_fields == ("training.method",)

def test_unresolved_hard_objection_is_rejected_at_review_limit():
    pipeline = ReviewPipeline(
        advocate=ScriptedAdvocate(valid_advocate()),
        critical=AlwaysHardObjection(),
        modifier=NoopModifier(),
        base=make_base(),
        max_rounds=2,
    )
    result = pipeline.review(valid_candidate(), {})
    assert result.approved is False
    assert result.rejection_reasons == ("unresolved_hard_objection",)

def test_review_identity_must_differ_from_controller_and_evaluator():
    base = make_base()
    with pytest.raises(ReviewIdentityError, match="distinct"):
        ReviewPipeline(
            advocate=ScriptedAdvocate(valid_advocate(), identity=base.controller_identity),
            critical=ScriptedCritical(valid_critical()),
            modifier=NoopModifier(),
            base=base,
            max_rounds=1,
        )
~~~

- [ ] **Step 2: Run tests to verify they fail**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_reviews.py
~~~

Expected: collection fails because `harness4h3.campaign.reviews` does not exist.

- [ ] **Step 3: Implement strict report parsing**

Advocate parsing must require bottleneck, changed_fields, expected_metric_delta, supporting_evidence_ids, falsification_experiment, and resource_assumptions. Critical parsing must require objections, objection_categories, missing_evidence_ids, proxy_gaming_risks, target_device_risks, and required_revisions. Reject unknown fields, empty required strings, unknown objection categories, non-finite deltas, and evidence IDs not present in the request context.

- [ ] **Step 4: Implement bounded orchestration**

For each candidate call Advocate once, then Critical. If Critical reports revisions, call Modifier with the original candidate, report, and immutable base digest. Re-run deterministic proposal validation on the revision, then call Final Critical. Stop at `max_rounds`; unresolved hard objections produce `approved=False`. Review output must never include a gate decision, a new hard constraint, a verifier change, a command, or a checkpoint promotion.

- [ ] **Step 5: Run tests and commit**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_reviews.py
~~~

Expected: PASS.

Commit:

~~~bash
git add harness4h3/campaign/reviews.py tests/unit/test_campaign_reviews.py
git commit -m "feat: add bounded advocate critical review"
~~~

---

### Task 4: Structured failure attribution and two-stage gates

**Files:**

- Create: `harness4h3/campaign/failures.py`
- Create: `harness4h3/campaign/gates.py`
- Modify: `tests/unit/campaign_fixtures.py`
- Create: `tests/unit/test_campaign_failures.py`
- Create: `tests/unit/test_campaign_gates.py`

**Interfaces:**

- Produces frozen `FailureReport(stage, category, responsible_variables, evidence_ids, deterministic_fix, confidence, prohibited_changes)`.
- Produces `FailureAttributor.attribute(stage: str, result: Mapping[str, Any], evidence_ids: Sequence[str]) -> FailureReport`.
- Produces frozen `MetricEvidence(metric_name, metric_version, input_reference, value, confidence_or_validity, evidence_source, device_profile_id, hard)`.
- Produces frozen `GateDecision(feasible, promotable, target_satisfied, violations, objective_values, evidence_ids, reason)`.
- Produces `AcceptanceGate.evaluate(candidate, evidence, *, hard_constraints, objectives, min_rounds_met) -> GateDecision`.
- Produces `StopReason` values `target_satisfied`, `budget_exhausted`, `no_progress`, and `safety_or_integrity_failure`.
- Extends `tests/unit/campaign_fixtures.py` with `evidence(name, value, hard=False) -> MetricEvidence` and `all_required_evidence() -> Mapping[str, MetricEvidence]`.

- [ ] **Step 1: Write failure and gate tests**

~~~python
def test_decode_failure_is_attributed_without_llm_category():
    report = FailureAttributor().attribute(
        "generation",
        {"failure_code": "video_decode_failed", "message": "invalid mp4"},
        ("evidence-v3",),
    )
    assert report.category == "generation_decode"
    assert report.deterministic_fix
    assert "video_decode_failed" in report.prohibited_changes

def test_hard_constraint_failure_beats_better_scalar_reward():
    decision = AcceptanceGate().evaluate(
        candidate=valid_candidate(),
        evidence={
            "quality": evidence("quality", 0.99, hard=True),
            "latency": evidence("latency_s", 8.0, hard=True),
        },
        hard_constraints={"max_latency_s": 3.0},
        objectives={"quality": "maximize", "latency_s": "minimize"},
        min_rounds_met=True,
    )
    assert decision.feasible is False
    assert decision.promotable is False
    assert decision.target_satisfied is False
    assert "max_latency_s" in decision.violations

def test_promotable_candidate_does_not_imply_target_satisfied():
    decision = AcceptanceGate().evaluate(
        candidate=valid_candidate(),
        evidence=all_required_evidence(),
        hard_constraints={},
        objectives={"quality": "maximize"},
        min_rounds_met=False,
    )
    assert decision.promotable is True
    assert decision.target_satisfied is False

def evidence(name, value, hard=False):
    return MetricEvidence(name, "test-v1", "fixture", value, 1.0, "target-runtime", "mobile", hard)

def all_required_evidence():
    return {
        "quality": evidence("quality", 0.9, hard=True),
        "latency_s": evidence("latency_s", 1.0, hard=True),
        "peak_memory_gb": evidence("peak_memory_gb", 4.0, hard=True),
        "model_size_gb": evidence("model_size_gb", 3.0, hard=True),
    }
~~~

- [ ] **Step 2: Run tests to verify they fail**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_failures.py tests/unit/test_campaign_gates.py
~~~

Expected: collection fails because the failure and gate contracts do not exist.

- [ ] **Step 3: Implement deterministic-first attribution**

Map known codes to typed categories before any semantic fallback: schema/shape/hash/capability failures, missing checkpoint, worker OOM/timeout, training invalidity, video missing/decode/blank/finite-pixel failures, and missing/invalid metric evidence. Set `deterministic_fix` only for a known programmatic fix. Unknown failures must retain the raw evidence ID and use category `unclassified_experimental_failure` without inventing a fix.

- [ ] **Step 4: Implement feasibility before objective comparison**

Evaluate every hard constraint and required evidence first. Missing, invalid, or proxy-only edge evidence fails the applicable hard gate. Only if `feasible=True` calculate objective values and Pareto dominance. Set `promotable=True` only for a feasible candidate with a valid child artifact; set `target_satisfied=True` only when all target constraints, required evidence, minimum round policy, and target objective floor are satisfied.

- [ ] **Step 5: Run tests and commit**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_failures.py tests/unit/test_campaign_gates.py
~~~

Expected: PASS.

Commit:

~~~bash
git add harness4h3/campaign/failures.py harness4h3/campaign/gates.py tests/unit/test_campaign_failures.py tests/unit/test_campaign_gates.py
git commit -m "feat: add structured failure attribution and hard gates"
~~~

---

### Task 5: Adapter contracts and multi-candidate StudentCampaign integration

**Files:**

- Create: `harness4h3/campaign/adapters.py`
- Modify: `harness4h3/student/campaign.py`
- Modify: `harness4h3/student/__init__.py`
- Create: `tests/unit/test_campaign_adapters.py`
- Modify: `tests/unit/test_student_campaign.py`
- Modify: `tests/integration/test_student_campaign.py`

**Interfaces:**

- Produces `CandidateExecutor` protocol:
  - `validate(candidate, round_dir) -> Mapping[str, Any]`
  - `execute(candidate, fidelity, round_dir) -> Mapping[str, Any]`
  - `verify(candidate, execution, round_dir) -> tuple[MetricEvidence, ...]`
- Produces `StudentCampaignAdapter` that maps StudentProposal/CompileManifest/TrainingResult/StudentEvaluation to the shared envelope and evidence.
- Adds `StudentProposalBatchProvider.propose_batch(context) -> Sequence[Mapping[str, Any]]`.
- Keeps `StudentProposalProvider.propose(context)` as a compatibility method that returns one proposal and wraps it into a one-item batch only when legacy mode is explicitly enabled.
- Extends `CampaignResult` with `target_satisfied`, `stop_reason`, `candidate_decisions`, and shared event references without removing existing serialized fields.

- [ ] **Step 1: Write adapter and batch campaign tests**

~~~python
def test_student_adapter_maps_compile_train_eval_to_shared_evidence(tmp_path):
    adapter = StudentCampaignAdapter(compiler=FakeCompiler(), worker=FakeWorker(), evaluator=FakeEvaluator())
    result = adapter.run_candidate(valid_candidate(), fidelity="F1", round_dir=tmp_path)
    assert result["v1"]["graph_status"] == "compiled"
    assert result["v2"]["status"] == "success"
    assert result["v3"]["video_decodable"] is True

def test_student_campaign_validates_all_candidates_before_execution(tmp_path):
    provider = BatchProvider([proposal_payload_with_id("C0001"), invalid_payload("C0002"), proposal_payload_with_id("C0003")])
    worker = CountingWorker()
    result = make_campaign(provider, worker, tmp_path).run(max_rounds=1)
    assert worker.calls == 2
    assert result.candidate_decisions["C0002"].promotable is False
    assert result.stop_reason == "no_progress"

def test_legacy_single_proposal_provider_requires_explicit_compatibility_mode(tmp_path):
    with pytest.raises(ValueError, match="batch"):
        make_campaign(SequenceProvider([valid_payload()]), CountingWorker(), tmp_path).run(max_rounds=1)

def proposal_payload_with_id(proposal_id):
    payload = valid_payload()
    payload["proposal_id"] = proposal_id
    return payload

def invalid_payload(proposal_id):
    payload = proposal_payload_with_id(proposal_id)
    payload["architecture"]["hidden_size"] = 1000
    return payload

class BatchProvider:
    provider_name = "test"
    model_name = "batch"

    def __init__(self, payloads):
        self.payloads = list(payloads)

    def propose_batch(self, context):
        return tuple(self.payloads)

class CountingWorker:
    def __init__(self):
        self.calls = 0

    def run(self, manifest, round_dir):
        self.calls += 1
        return scripted_training_result(manifest, round_dir, self.calls)

def make_campaign(provider, worker, output_root):
    return StudentCampaign(provider, StudentCompiler(), worker, ScriptedEvaluator(),
                           output_root=output_root, target=StudentTarget(),
                           campaign_base=make_base(), review_pipeline=pass_through_review())
~~~

- [ ] **Step 2: Run tests to verify they fail**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_adapters.py tests/unit/test_student_campaign.py tests/integration/test_student_campaign.py
~~~

Expected: new adapter/batch tests fail while the existing single-proposal tests continue to pass.

- [ ] **Step 3: Implement Student adapter without moving training logic**

Use the existing `StudentCompiler`, `StudentRoundWorker`, `StudentRoundEvaluator`, `TrainingResult`, `StudentEvaluation`, and `MetricVerifierBank`. The adapter must never construct a model, run SSH, or infer quality from a free-text message. It must preserve compiler/parent/child hashes and label remote/server evidence separately from target-device evidence.

- [ ] **Step 4: Add strict batch provider parsing**

Add a JSON schema whose top-level result is an array of 3–5 StudentProposal objects. Validate every item independently, reject duplicate proposal digests, preserve parent IDs and mutation fields, and pass only the deterministic-valid subset to review/execution. A malformed batch records a proposal failure and launches no worker.

- [ ] **Step 5: Integrate shared base, review, events, and gates into StudentCampaign**

Initialize CampaignBase and DecisionTrace before round one. For each round append proposal.generated, proposal.validated, critic.completed, training.started/completed, evaluation.started/completed, gate.decided, archive.updated, and campaign.replanned/stopped events with the same IDs and base digest. Run candidates independently so one rejected candidate does not erase valid siblings. Keep the active parent unchanged until a child passes the shared Gate. Do not return success merely because any candidate is promotable.

- [ ] **Step 6: Update existing tests for compatibility and run focused integration**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_campaign_adapters.py tests/unit/test_student_campaign.py tests/integration/test_student_campaign.py
~~~

Expected: PASS, including the pre-existing failure-context and duplicate-proposal tests.

Commit:

~~~bash
git add harness4h3/campaign/adapters.py harness4h3/student/campaign.py harness4h3/student/__init__.py tests/unit/test_campaign_adapters.py tests/unit/test_student_campaign.py tests/integration/test_student_campaign.py
git commit -m "feat: integrate shared control plane with student campaign"
~~~

---

### Task 6: Scripted end-to-end control-plane integration and recovery

**Files:**

- Create: `tests/integration/test_campaign_control_plane.py`
- Modify: `harness4h3/campaign/events.py`
- Modify: `harness4h3/campaign/adapters.py`
- Modify: `harness4h3/student/campaign.py` only where the integration test exposes a contract defect.

**Interfaces:**

- The integration fixture creates a base, three candidates from two parents, scripted Advocate/Critical/Modifier agents, a worker that fails one candidate, and a verifier that returns one feasible/promotable candidate.
- The fixture must be able to stop after an event, reconstruct the trace from disk, resume without duplicating an experiment ID, and expose the second-round context with failure attribution and predicted-vs-actual deltas.

- [ ] **Step 1: Write the complete scripted integration test**

~~~python
def test_control_plane_preserves_lineage_review_evidence_and_resume(tmp_path):
    campaign = build_scripted_campaign(tmp_path)
    first = campaign.run(max_rounds=1)
    assert first.stop_reason == "no_progress"
    assert first.candidate_decisions["C0002"].failure_code == "video_decode_failed"

    resumed = build_scripted_campaign(tmp_path).run(max_rounds=2)
    assert resumed.target_satisfied is True
    assert resumed.candidate_decisions["C0003"].promotable is True

    events = DecisionTrace(tmp_path / "campaign-events.jsonl", campaign.base).read()
    assert event_types(events) == [
        "campaign.created", "proposal.generated", "proposal.validated",
        "critic.completed", "training.started", "training.completed",
        "evaluation.started", "evaluation.completed", "gate.decided",
        "archive.updated", "campaign.replanned", "campaign.stopped",
    ]
    assert lineage(events, "C0003") == ("M0002", "C0003")
    assert events_for("C0002", events)[-1].payload["failure_code"] == "video_decode_failed"

def build_scripted_campaign(tmp_path):
    return StudentCampaign(
        provider=ThreeCandidateProvider(),
        compiler=ScriptedCompiler(),
        worker=BranchingWorker(),
        evaluator=BranchingEvaluator(),
        output_root=tmp_path,
        campaign_base=make_base(),
        review_pipeline=OneObjectionThenApproveReview(),
    )

def event_types(events):
    return [event.event_type for event in events]

def events_for(candidate_id, events):
    return tuple(event for event in events if event.candidate_id == candidate_id)

def lineage(events, candidate_id):
    terminal = events_for(candidate_id, events)[-1]
    return (terminal.parent_candidate_id, terminal.candidate_id)
~~~

- [ ] **Step 2: Run the integration test and verify it fails**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/integration/test_campaign_control_plane.py
~~~

Expected: FAIL until the shared event, recovery, and adapter contracts are wired together.

- [ ] **Step 3: Implement idempotent resume**

On restart, load and verify the trace before reading resume state. If a candidate already has a terminal gate event, do not launch its worker again. If an experiment has a started event without a completed event, inspect the executor result bundle through the adapter; return the existing result when its digest matches, otherwise emit a typed recoverable failure. Never reuse an experiment ID for a different candidate.

- [ ] **Step 4: Verify multi-branch lineage and experience fields**

The integration result must retain parent_candidate_id, generation, mutation_fields, predicted_metric_delta, actual_metric_delta, advocate claims, critical objections, resolved objections, failure attribution, verifier versions, target hash, dataset hash, and novelty. The test must prove a historical non-best parent can be selected for a later child.

- [ ] **Step 5: Run integration tests and commit**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/integration/test_campaign_control_plane.py tests/integration/test_student_campaign.py
~~~

Expected: PASS.

Commit:

~~~bash
git add harness4h3/campaign/events.py harness4h3/campaign/adapters.py harness4h3/student/campaign.py tests/integration/test_campaign_control_plane.py
git commit -m "test: verify campaign control-plane recovery and lineage"
~~~

---

### Task 7: Legacy H3 adapter, documentation, and full verification

**Files:**

- Modify: `harness4h3/campaign/adapters.py`
- Modify: `harness4h3/controller/loop.py`
- Modify: `README.md`
- Create: `tests/unit/test_legacy_campaign_adapter.py`
- Create: `tests/integration/test_legacy_campaign_control_plane.py`

**Interfaces:**

- Produces `LegacyH3CampaignAdapter` that maps existing OperatorRegistry plans, ModelCandidate lineage, OperatorResult, CompositeEvaluator output, ModelStore, ParetoArchive, and ExperimentStore into the shared contracts.
- Existing `OptimizationLoop.run(...)` remains callable with its current arguments and return type; the adapter adds trace/base evidence rather than removing current fields.
- Existing `critical_regression` remains available for compatibility but is represented as one typed hard-failure evidence, not the complete video quality verifier.

- [ ] **Step 1: Write legacy adapter tests**

~~~python
def test_legacy_adapter_exposes_only_registered_operator_capabilities():
    registry = build_model_evolution_registry()
    snapshot = LegacyH3CampaignAdapter.capability_snapshot(registry)
    assert set(snapshot.available_names()) == set(registry.names())

def test_legacy_quality_regression_cannot_be_overruled_by_latency_gain(tmp_path):
    adapter = build_legacy_adapter(tmp_path)
    decision = adapter.gate(operator_result={"status": "success"},
                             evaluation={"critical_regression": True,
                                         "quality_score": 0.2,
                                         "latency_s": 0.01})
    assert decision.feasible is False
    assert decision.promotable is False

def test_existing_optimization_loop_api_and_tests_remain_compatible(tmp_path):
    result = run_existing_fake_loop(tmp_path)
    assert result.status in {"target_satisfied", "budget_exhausted", "critical_failure"}

def run_existing_fake_loop(tmp_path):
    from tests.integration.test_fake_closed_loop import make_loop, mobile_target, baseline, budget, StaticController, valid_plan
    return make_loop(
        tmp_path,
        controller=StaticController(lambda context: valid_plan(context)),
    ).run("legacy-adapter", mobile_target(), budget(), baseline())

def build_legacy_adapter(tmp_path):
    return LegacyH3CampaignAdapter(
        registry=build_model_evolution_registry(),
        output_root=tmp_path,
        target=mobile_target(),
    )
~~~

- [ ] **Step 2: Run legacy tests to establish the failing boundary**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q tests/unit/test_legacy_campaign_adapter.py tests/integration/test_legacy_campaign_control_plane.py tests/test_loop.py tests/test_archive.py tests/test_evaluator.py
~~~

Expected: the two new tests fail before the adapter is wired; existing tests remain green.

- [ ] **Step 3: Implement the legacy adapter**

Use `OperatorRegistry.names()` and `visible()` for the capability snapshot, `ModelCandidate.parent_id/generation` for lineage, `ExperimentRecord` for experiment evidence, and the existing evaluator result as V3/V4 evidence. Preserve current ModelStore active-parent behavior and append shared events around, rather than inside, the existing operator implementation.

- [ ] **Step 4: Document the operational boundary**

Update README with the control-plane flow, the distinction between proxy and edge evidence, the four allowed stop reasons, the focused campaign test command, and the fact that the current 557-pass baseline does not prove real DMD2/edge capability.

- [ ] **Step 5: Run full regression and compile checks**

Run:

~~~bash
cd Harness4H3
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q harness4h3 h3_training tools research
~~~

Expected: at least `557 passed, 2 skipped`, with no new skipped tests and no CUDA requirement added by the control plane.

- [ ] **Step 6: Commit the legacy integration and documentation**

~~~bash
git add harness4h3/campaign/adapters.py harness4h3/controller/loop.py README.md tests/unit/test_legacy_campaign_adapter.py tests/integration/test_legacy_campaign_control_plane.py
git commit -m "feat: attach legacy h3 loop to campaign control plane"
~~~

---

## Self-review checklist

- Spec coverage: Tasks 1–2 cover immutable Verification Base, capability-derived action space, candidate batch, deterministic validation, and identity hashes; Task 3 covers Advocate/Critical/Revision; Task 4 covers V0–V4 evidence, FailureAttributor, hard constraints, Pareto, and stop semantics; Tasks 5–7 cover Student/H3 adapters, Decision Trace, recovery, lineage, Experience fields, and regression evidence.
- No placeholder steps: every task names exact files, interfaces, test commands, expected outcomes, and concrete behavior.
- Type consistency: `CampaignBase`, `ActorIdentity`, `DecisionEvent`, `CandidateEnvelope`, `ProposalBatch`, `MetricEvidence`, and `GateDecision` are introduced before adapters consume them; legacy and Student adapters return the same evidence/gate shapes.
- Scope boundary: real training backends, target-device runtime, quality verifier bank, multi-fidelity promotion policy, and archive expansion remain later subprojects, but this plan makes their capability/evidence/gate seams enforceable now.
