# Real MiniMax-H3 Closed-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one authentic MiniMax-H3 recovery-finetune experiment through checkpoint verification, independent benchmark evaluation, Harness-owned continuation, and a second Controller plan.

**Architecture:** Add a dynamic `RealMiniMaxH3Adapter` for the generic training API while retaining the existing distributed ComfyUI worker for L40x4 production execution. Adapt the real benchmark to canonical `EvaluationRecord`, run the real campaign through model/system pair state, and persist a deterministic experiment fingerprint plus all training and benchmark evidence.

**Tech Stack:** Python 3.9+, PyTorch, safetensors, ComfyUI MiniMax-H3 implementation loaded dynamically, existing H3BenchmarkRunner, pytest, JSON/JSONL atomic stores.

## Global Constraints

- Never modify the parent checkpoint in place; require `parent_sha256_before == parent_sha256_after`.
- Never use training loss or operator-estimated metrics as quality, latency, memory, energy, or Pareto evidence.
- Keep `offline_simulation=false` and real-worker provenance through model, benchmark, evaluation, and trajectory records.
- Runtime-only operations preserve `model_id` and create a new `system_id`.
- Controller acceptance is advisory; evaluator feasibility and `ContinuationPolicy` are authoritative.
- CPU tests may verify contracts but cannot claim the real GPU gate passed.
- Preserve existing dirty worktree changes and old fake/TinyH3 tests.

---

### Task 1: Implement the real ComfyUI H3 adapter

**Files:**
- Create: `h3_training/adapters/real_h3.py`
- Modify: `h3_training/adapters/__init__.py`
- Test: `tests/training/test_real_h3_adapter.py`

**Interfaces:**
- Consumes: ComfyUI root, a MiniMax-H3 safetensors checkpoint, and the existing `DenoisingModelAdapter` schemas.
- Produces: `RealMiniMaxH3Adapter` with real loader, batch preparation, forward, trainable-parameter resolution, save, and reload methods.

- [ ] **Step 1: Write adapter contract tests**

Test the dynamic loader through a temporary fake ComfyUI module and real safetensors files. The fixture model must expose real `torch.nn.Parameter` objects, accept the same `model([video, audio], timestep, context, transformer_options, minimax_payload)` call, and return tensors derived from those parameters. Assert missing CUDA/ComfyUI symbols fail with `TrainingFailure`, packed cache input yields native modality tensors, prediction sign follows clean-minus-noise velocity, only declared head names resolve, and save/reload preserves state.

- [ ] **Step 2: Run the adapter tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_real_h3_adapter.py -q`

Expected: collection or import failure because `RealMiniMaxH3Adapter` does not exist.

- [ ] **Step 3: Implement the dynamic loader and H3 tensor conversions**

Implement these concrete helpers:

```python
class RealMiniMaxH3Adapter(DenoisingModelAdapter):
    def __init__(self, comfyui_root: Path, device="cuda", dtype=torch.bfloat16): ...
    def load_role(self, path: Path, trainable=False) -> ModelRole: ...
    def prepare_batch(self, raw, generator) -> PreparedBatch: ...
    def predict(self, role, noisy, timestep, conditioning) -> ModalPrediction: ...
    def resolve_trainable_parameters(self, role, policy="heads") -> Iterable[str]: ...
    def save_role(self, role, path: Path) -> Mapping[str, Any]: ...
    def reload_role(self, path: Path) -> ModelRole: ...
```

Load `metadata["config"]` from safetensors, instantiate ComfyUI `MiniMaxH3Model` with `operations.disable_weight_init`, load the complete state dict with strict key checks, and move it to the configured device. Convert the verified flat video/audio cache to `[B, C, T, H, W]` and `[B, C, T, F]`, use video shift 12/audio shift 3 for noise, pass `PackedLayout` to the native forward, negate ComfyUI raw velocity to obtain clean-minus-noise velocity, and save a metadata-preserving safetensors checkpoint. Do not add a CPU or TinyH3 fallback.

- [ ] **Step 4: Run adapter tests and the existing training contract tests**

Run: `.venv/bin/python -m pytest tests/training/test_real_h3_adapter.py tests/training/test_h3_contract.py tests/training/test_tiny_model.py -q`

Expected: all selected tests pass; CUDA-only execution remains skipped when CUDA is unavailable.

- [ ] **Step 5: Commit the adapter slice**

```bash
git add h3_training/adapters/real_h3.py h3_training/adapters/__init__.py tests/training/test_real_h3_adapter.py
git commit -m "feat: add real MiniMax H3 adapter contract"
```

---

### Task 2: Make the real worker evidence gate explicit

**Files:**
- Modify: `tools/h3_real_train_worker.py`
- Modify: `tools/h3_real_support.py`
- Modify: `harness4h3/operators/external.py`
- Test: `tests/training/test_real_h3_train_worker_contract.py`
- Test: `tests/unit/test_h3_model_worker.py`

**Interfaces:**
- Consumes: external operator requests with `recovery_finetune` and the L40x4 worker config.
- Produces: a validated worker result containing all required training evidence and a child `ModelState` with `offline_simulation=false`.

- [ ] **Step 1: Add a required-evidence validator test**

Add a test that removes each of `parent_sha256_before`, `parent_sha256_after`, `initial_loss`, `final_loss`, `gradient_norm`, `optimizer_steps`, `changed_trainable_tensors`, `unchanged_frozen_tensors`, `child_sha256`, `child_reloaded`, and `peak_vram_per_rank` from a worker result and asserts the operator rejects it as `invalid_training_evidence`.

- [ ] **Step 2: Implement one stable evidence validator**

Create a worker-side function that checks finite losses, positive gradient norm, at least one optimizer step, changed trainable tensors, unchanged frozen tensors, distinct parent/child hashes, successful reload, and real-worker provenance before publishing `status=success`. Keep the current full frozen scan or explicit byte-for-byte mode visible in evidence; never infer a pass from missing fields.

- [ ] **Step 3: Preserve evidence through `ExternalScriptOperator`**

Ensure successful operator metrics include the full worker evidence and failure results retain stable failure type/message. Reject `offline_simulation=true` from a command used by the real registry. Keep the parent digest check around the subprocess.

- [ ] **Step 4: Run worker and external-operator tests**

Run: `.venv/bin/python -m pytest tests/training/test_real_h3_train_worker_contract.py tests/unit/test_h3_model_worker.py tests/unit/test_external_operator.py -q`

Expected: all tests pass and invalid or incomplete evidence is rejected before a child is registered.

- [ ] **Step 5: Commit the evidence gate**

```bash
git add tools/h3_real_train_worker.py tools/h3_real_support.py harness4h3/operators/external.py tests/training/test_real_h3_train_worker_contract.py tests/unit/test_h3_model_worker.py
git commit -m "feat: enforce authentic H3 training evidence"
```

---

### Task 3: Canonical real benchmark evaluation for model/system pairs

**Files:**
- Modify: `research/experiments/a1_real_evolution.py`
- Modify: `harness4h3/benchmark/h3.py`
- Modify: `harness4h3/evaluator/composite.py`
- Modify: `harness4h3/controller/schemas.py`
- Test: `tests/unit/test_a1_real_evolution.py`
- Test: `tests/unit/test_h3_benchmark.py`

**Interfaces:**
- Consumes: a child `ModelState`, optional `SystemCandidate`, target, device ID, task split, and fixed benchmark recipe.
- Produces: one canonical `EvaluationRecord` with raw quality metrics, aggregate quality, latency, peak memory, model size, optional energy, validity, hard-gate violations, evaluator version, and provenance.

- [ ] **Step 1: Write tests for independent benchmark evidence**

Assert that training evidence does not make an evaluation feasible, that a benchmark with a missing/invalid task rejects the child, and that two evaluations carry distinct `(model_id, system_id, device_id, task_split, benchmark_recipe)` provenance while using the same recipe.

- [ ] **Step 2: Adapt `TieredRealBenchmarkEvaluator` to canonical records**

Accept `system=None`, `device_id`, and `task_split`; pass the system runtime recipe to the benchmark; set `quality_metrics` to raw metrics plus benchmark summary; set `hardware` only from measured sampler/benchmark values; set `validity` from artifact/decode/semantic gates; set `provenance["offline_simulation"] = False`; and calculate `search_score` only after the measured record is built.

- [ ] **Step 3: Keep raw and aggregate quality separate**

Ensure `BenchmarkSummary` retains the raw metrics and evaluator version while `quality_score` remains the aggregate. No worker loss or estimated hardware value may enter those fields.

- [ ] **Step 4: Run benchmark/evaluator tests**

Run: `.venv/bin/python -m pytest tests/unit/test_a1_real_evolution.py tests/unit/test_h3_benchmark.py tests/unit/test_evaluation_record.py -q`

Expected: all selected tests pass with canonical `EvaluationRecord` round trips.

- [ ] **Step 5: Commit the canonical benchmark slice**

```bash
git add research/experiments/a1_real_evolution.py harness4h3/benchmark/h3.py harness4h3/evaluator/composite.py harness4h3/controller/schemas.py tests/unit/test_a1_real_evolution.py tests/unit/test_h3_benchmark.py tests/unit/test_evaluation_record.py
git commit -m "feat: record canonical real H3 benchmark evaluations"
```

---

### Task 4: Run real A1 through the active model/system loop and second plan

**Files:**
- Modify: `harness4h3/controller/loop.py`
- Modify: `harness4h3/operators/model_evolution.py`
- Modify: `harness4h3/memory/experiment_store.py`
- Create: `harness4h3/memory/fingerprint.py`
- Modify: `research/experiments/a1_real_evolution.py`
- Test: `tests/integration/test_real_h3_closed_loop.py`

**Interfaces:**
- Consumes: `M0000 + S0000`, a real external operator registry, canonical benchmark evaluator, and a controller provider.
- Produces: `exp_0001 -> M0001 + S0000 -> E0001 -> continuation -> exp_0002` or a truthful rejection/failed-worker trajectory.

- [ ] **Step 1: Write the end-to-end rejection-path test**

Use a fixture external worker that writes a real child checkpoint and measured benchmark fixture, then returns a quality regression. Assert `M0001` is archived, its evaluation is `offline_simulation=false`, continuation is `REJECT`, parent remains current, and the second controller request contains the real evaluation and `exp_0002` is generated.

- [ ] **Step 2: Implement deterministic experiment fingerprints**

Add:

```python
def experiment_fingerprint(parent_model_id, parent_system_id, operator, operator_args, target_id, device_id, benchmark_recipe) -> str
```

Normalize mappings recursively with sorted keys and canonical JSON. Reject exact duplicate fingerprints before executing unless `repeat_for_statistics=true`; persist the fingerprint and repeat purpose in `ExperimentRecord`.

- [ ] **Step 3: Route the real A1 runner through pair-aware state**

Initialize model/system stores, use the external recovery-finetune registry, pass the current system into the benchmark, persist training evidence and canonical evaluation, call `ContinuationPolicy`, and build the next context from the recorded result. Do not make target satisfaction depend on the worker loss.

- [ ] **Step 4: Verify the second Controller request**

Assert context contains current model, current system, recent real evaluation, failure memory, Pareto front, device, benchmark recipe, and duplicate fingerprint. The second plan must be produced even when the first child is rejected.

- [ ] **Step 5: Run the integration slice**

Run: `.venv/bin/python -m pytest tests/integration/test_real_h3_closed_loop.py tests/unit/test_continuation.py tests/unit/test_controller_providers.py -q`

Expected: the full rejection/second-plan path passes without TinyH3 or fake metrics entering real evidence.

- [ ] **Step 6: Commit the active real-loop slice**

```bash
git add harness4h3/controller/loop.py harness4h3/operators/model_evolution.py harness4h3/memory/experiment_store.py harness4h3/memory/fingerprint.py research/experiments/a1_real_evolution.py tests/integration/test_real_h3_closed_loop.py
git commit -m "feat: connect real H3 evidence to the RSI loop"
```

---

### Task 5: Add the explicit GPU gate and finish verification

**Files:**
- Create: `tools/run_real_h3_gate.py`
- Modify: `configs/a1-worker.l40x4-distill4.json`
- Modify: `docs/quickstart.md`
- Modify: `docs/validation-plan.md`
- Test: `tests/unit/test_real_h3_gate.py`

**Interfaces:**
- Consumes: worker config, parent checkpoint, ComfyUI root/service, real task manifest, and output root.
- Produces: a persisted gate result with `passed`, prerequisite failures, training evidence path, benchmark evidence path, and trajectory path.

- [ ] **Step 1: Write gate tests**

Assert missing checkpoint, unavailable CUDA, missing ComfyUI root, missing task set, and missing benchmark endpoint each produce a non-passing JSON result with a specific prerequisite code and never claim real evidence.

- [ ] **Step 2: Implement the gate command**

The command must preflight paths/device/service, run one bounded real `recovery_finetune`, run baseline and child benchmark with the same recipe, validate all Definition-of-Done evidence, and write `real_h3_gate.json`. It must exit nonzero for any missing prerequisite or failed criterion and may never substitute fake/TinyH3 data.

- [ ] **Step 3: Document the exact real command and truthful CPU behavior**

Document the L40x4 command, expected artifact paths, required evidence, and the fact that a local CPU run only exercises rejection/contract tests. Do not label a GPU gate as passed from CI.

- [ ] **Step 4: Run complete CPU verification**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m compileall -q harness4h3 h3_training tools research && git diff --check`

Expected: all CPU tests pass; CUDA-dependent tests are explicitly skipped; compile and diff checks are clean.

- [ ] **Step 5: Run the GPU gate when external prerequisites are available**

Run: `.venv/bin/python tools/run_real_h3_gate.py --config configs/a1-worker.l40x4-distill4.json --parent-checkpoint /data/models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors --output-root var/real-h3-gate --comfyui-root /home/intern/huangjiahao/ComfyUI --tasks examples/tasks.yaml --base-url http://127.0.0.1:8188`

Expected: either `passed: true` with persisted evidence for every gate criterion, or a nonzero result naming the exact unavailable prerequisite. Never report completion without the former.

- [ ] **Step 6: Commit the gate/docs slice**

```bash
git add tools/run_real_h3_gate.py configs/a1-worker.l40x4-distill4.json docs/quickstart.md docs/validation-plan.md tests/unit/test_real_h3_gate.py
git commit -m "feat: add real H3 acceptance gate"
```
