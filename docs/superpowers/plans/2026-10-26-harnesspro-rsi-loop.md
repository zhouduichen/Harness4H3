# HarnessPro RSI Loop Implementation Plan

> **For agentic workers:** Execute tasks inline in order. Stop at M0 acceptance; M1/M2 require their own frozen protocol and data gates.

**Goal:** Build the M0 foundation for immutable harness versions, auditable research traces, bounded edits, recoverable execution, and accurate cost accounting in the Wan Block RSI loop.

**Architecture:** Add a small `harness4h3.rsi` package that wraps current provider, SkillHub, Block campaign, and worker interfaces. Store each editable harness as a content-addressed bundle; keep execution, evaluator, budget enforcement, and acceptance outside the editable bundle. This plan does not add a new training algorithm or launch Wan jobs.

**Tech Stack:** Python 3.10-compatible standard library, existing Harness4H3 provider/sandbox/Block worker, JSON/JSONL atomic persistence, existing pytest suite.

## Global Constraints

- Preserve the dirty worktree and all existing Block experiment artifacts.
- No full-checkout reset, broad cherry-pick, or editing any active remote execution snapshot.
- Candidate harness code may modify only its declared three components: prompt/context, retrieval, and proposal/experiment strategy.
- Candidate code cannot edit evaluation, budget, data split, task service, or acceptance policy.
- M0 fake/CPU checks establish engineering behavior only; they cannot satisfy Wan quality gates.
- Do not launch M1/M2 or request frozen evaluation data in this implementation stage.
- All inputs, provider calls, failures, retries, and GPU allocation are accounted; unavailable measurements remain unavailable.

---

## Repository Map

- Create `harness4h3/rsi/bundle.py`: immutable bundle identity, restricted file manifest, atomic capture and restore.
- Create `harness4h3/rsi/trace.py`: typed event validation, usage completeness, elapsed/cost ledgers, failure categories.
- Create `harness4h3/rsi/budget.py`: append-only request/GPU/optimizer budgets and idempotent reservation/finalization.
- Create `harness4h3/rsi/runner.py`: fixed task-service facade, external policy enforcement, recovery by stable job identity.
- Create `configs/rsi-harness/seed/` with the three editable seed components.
- Create focused tests under `tests/unit/test_rsi_*.py`.
- Do not modify the current WAN campaign launcher or remote configs in M0; integration follows after these contracts are proven.

## Task 1: Immutable Harness Bundles

**Files:**
- Create: `harness4h3/rsi/__init__.py`
- Create: `harness4h3/rsi/bundle.py`
- Create: `tests/unit/test_rsi_bundle.py`
- Create: `configs/rsi-harness/seed/context.py`
- Create: `configs/rsi-harness/seed/retrieval.py`
- Create: `configs/rsi-harness/seed/strategy.py`

**Interfaces:**
- `HarnessBundle.capture(source: Path, store: Path, *, parent_id: str | None, hypothesis: Mapping[str, Any], allowed_paths: Sequence[str]) -> HarnessBundle`
- `HarnessBundle.restore(store: Path, destination: Path) -> Path`
- `HarnessBundle.load(store: Path, bundle_id: str) -> HarnessBundle`
- `HarnessBundle.to_dict() -> dict[str, Any]`
- `HarnessBundle.bundle_id` is SHA256 over normalized relative paths and file bytes, excluding metadata, logs, caches and runtime state.

- [ ] Test deterministic identity, parent linkage, include/exclude behavior, and rejection of paths outside the declared bundle.
- [ ] Test that changing source after capture does not mutate the stored bundle.
- [ ] Implement capture by copying only allowlisted regular files into a temporary directory, hashing a sorted path/content stream, writing a manifest, fsyncing, then atomically renaming to the digest directory.
- [ ] Implement restore by verifying every manifest digest before copying; reject symlinks, missing files, path traversal, and digest mismatch.
- [ ] Add seed components with explicit function contracts: `build_context(task, evidence)`, `retrieve(evidence, query)`, `choose_experiment(context, candidates)`. Initial behavior should preserve the current bounded Block campaign behavior and contain no evaluator or filesystem authority.
- [ ] Run `pytest tests/unit/test_rsi_bundle.py -q`; expect all bundle identity and isolation cases to pass.

## Task 2: Trace Events and Complete Cost Accounting

**Files:**
- Create: `harness4h3/rsi/trace.py`
- Create: `harness4h3/rsi/budget.py`
- Create: `tests/unit/test_rsi_trace.py`
- Create: `tests/unit/test_rsi_budget.py`

**Interfaces:**
- `TraceEvent.create(*, run_id: str, task_id: str, bundle_id: str, stage: str, event: str, status: str, started_ns: int, ended_ns: int, payload: Mapping[str, Any], evidence_refs: Sequence[str] = ()) -> TraceEvent`
- `TraceStore.append(event: TraceEvent) -> None`; appends one JSONL record and fsyncs.
- `TraceStore.read(run_id: str) -> list[TraceEvent]`; validates schema and rejects malformed/truncated records.
- `CostLedger.summary(run_id: str) -> dict[str, Any]`; reports elapsed wall time, GPU lease time, optimizer updates, input/output tokens, observed price, and missing-cost fields separately.
- `BudgetLedger.reserve(job_id: str, limits: Mapping[str, int | float]) -> bool`; `finalize(job_id: str, actual: Mapping[str, int | float]) -> None`.

- [ ] Test nonnegative monotonic timestamps, stable run/task/bundle linkage, and required failure classes `technical`, `resource_limited`, `method_negative`, and `protocol_violation`.
- [ ] Test that missing provider usage is marked unavailable and never inferred from text length; USD remains unavailable when a price table is absent.
- [ ] Test append/fsync/read behavior and rejection of duplicate job settlement with conflicting values.
- [ ] Test that failed and retried jobs consume reserved and actual budget, exact matching duplicate finalization is idempotent, and over-budget reservations fail closed.
- [ ] Implement typed JSON serialization with schema version, atomic ledger state, append-only events, and reconciliation from events on resume.
- [ ] Run `pytest tests/unit/test_rsi_trace.py tests/unit/test_rsi_budget.py -q`; expect missing measurements to remain null/unavailable and costs to reconcile.

## Task 3: Restricted Candidate Execution and Recoverable Research Jobs

**Files:**
- Create: `harness4h3/rsi/runner.py`
- Create: `tests/unit/test_rsi_runner.py`
- Modify only if necessary: `harness4h3/sandbox/policy.py`

**Interfaces:**
- `ResearchTaskService` is a trusted protocol with `collect(task_id, bundle_id, seed) -> Mapping[str, Any]` and `evaluate(task_id, bundle_id, artifact_ref) -> Mapping[str, Any]`.
- `HarnessRunner.run_component(bundle: HarnessBundle, component: str, input_payload: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]` executes one allowlisted component in an isolated process with JSON-only input/output.
- `ResearchJobRunner.run(job_id: str, bundle_id: str, task_id: str, seed: int) -> Mapping[str, Any]` checks the budget, reuses a verified completed job with the same full identity, and writes every stage to `TraceStore`.

- [ ] Test direct API restrictions: a candidate component cannot request evaluator thresholds, raw validation/test items, unrestricted paths, dynamic imports, subprocesses, environment secrets, or GPU devices.
- [ ] Test deterministic replay of a completed job and rejection when task, seed, protocol, or bundle digest differs.
- [ ] Test real worker response validation: required artifact/hash, finite metrics, failure classification, optimizer update count, and checkpoint reload evidence.
- [ ] Implement a small stdlib-only subprocess protocol; store request/response JSON and stderr by digest; enforce timeout and output-size limits outside the candidate process.
- [ ] Add no-op/known-behavior CPU fixtures to prove one legitimate edit affects the executed component and a forbidden edit does not.
- [ ] Run `pytest tests/unit/test_rsi_runner.py -q`; expect valid JSON components to execute and all disallowed requests to fail before task-service access.

## Task 4: M0 Preflight and Seed Snapshot

**Files:**
- Create: `tools/rsi_m0_preflight.py`
- Create: `tests/unit/test_rsi_m0_preflight.py`
- Create: `docs/superpowers/reports/2026-10-26-harnesspro-rsi-m0-preflight.md`
- Modify: `harness4h3/rsi/__init__.py` only for stable exports.

**Interfaces:**
- `preflight(config_path: Path, bundle_store: Path, trace_root: Path) -> dict[str, Any]` validates local config and contracts without downloading assets or loading Wan weights.
- CLI arguments: `--config`, `--bundle-store`, `--trace-root`, `--output`; output is a JSON report with `passed`, `blockers`, source manifest, fixture evidence and budget ledger digest.

- [ ] Test that the preflight rejects missing/duplicate editable components, mutable evaluator paths inside a bundle, absent price completeness mislabeled as cost, and incomplete recovery evidence.
- [ ] Test a full offline lifecycle: capture seed, run known component fixture, interrupt/reopen stores, resume without duplicate completion, and reconcile costs.
- [ ] Implement preflight using only local files and fake task fixtures; fail if it tries to instantiate the Diffusers pipeline or contact a model endpoint.
- [ ] Run focused M0 suite: `pytest tests/unit/test_rsi_bundle.py tests/unit/test_rsi_trace.py tests/unit/test_rsi_budget.py tests/unit/test_rsi_runner.py tests/unit/test_rsi_m0_preflight.py -q`.
- [ ] Run `python tools/rsi_m0_preflight.py --config configs/rsi-harness/m0.yaml --bundle-store var/rsi-m0/bundles --trace-root var/rsi-m0/traces --output var/rsi-m0/preflight.json`; require `passed=true`, valid bundle/store digests, exact reconciliation, and no network/GPU evidence.
- [ ] Write the report with passed checks, source manifest, actual measurements, blockers and the explicit statement that M0 does not demonstrate RSI research gains.

## Task 5: Review Boundary and Handoff to M1

**Files:**
- Modify: `docs/superpowers/specs/2026-10-26-harnesspro-rsi-loop-design.md` only to record actual M0 interface changes or contract deviations.
- Modify: `docs/superpowers/reports/2026-10-26-harnesspro-rsi-m0-preflight.md` with final gate outcome.

- [ ] Inspect all new artifacts and verify the baseline bundle only contains declared source components.
- [ ] Compare git status against the initial dirty-file inventory; confirm all pre-existing paths retain their original staged/unstaged/untracked status.
- [ ] Confirm the M0 preflight never launched a Wan job, changed any remote execution snapshot, or read a frozen split.
- [ ] If every M0 gate passes, prepare M1's fixed task manifest, evidence partition hashes, request/token cap, and external evaluator contract as a separate reviewable protocol before any model calls.
- [ ] If any M0 gate fails, record the specific blocker and keep M1/M2 stopped; do not relax the gate to start the study.

## Plan Self-Review

- M0 bundle, trace, budget, task service, resumption, source inventory, and report requirements in the approved design map to Tasks 1–5.
- M1/M2 are explicitly outside this implementation batch and cannot start from an M0 engineering pass alone.
- Contracts use one spelling consistently: `bundle_id`, `run_id`, `task_id`, `job_id`, and `TraceStore`.
- All Wan training and quality gates remain in the approved design and are not weakened by this plan.
