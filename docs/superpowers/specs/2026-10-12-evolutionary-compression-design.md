# Harness4H3 EvoCompress Design

## Goal

Extend the existing Harness4H3 closed loop into an evolutionary compression
pipeline for DiT models. The system must be able to inspect a Wan2.1-T2V-1.3B
model, propose and execute compression algorithms, measure quality and system
cost, retain successful and failed experiments, and use that evidence to
generate improved algorithm candidates.

The existing H3 campaign, Wan student training campaign, evaluator, scheduler,
archive, memory stores, and controller contracts remain readable and continue
to serve their current users. The new compression path is additive and is
explicitly separate from the current Wan distillation/training path.

## Current repository boundary

The repository already provides:

- `harness4h3/operators` with a safe operator registry and H3 model-evolution
  operators;
- `harness4h3/evaluator`, `archive`, `memory`, `executor`, and `remote` modules;
- a Wan-native student training/evaluation path under
  `harness4h3/student` and `h3_training`;
- GPU leasing and remote campaign infrastructure;
- append-only `ExperienceStore` and `DesignGeneStore` implementations.

The existing `harness4h3/evolution` directory is currently documentation only.
It must not become a second implementation of the existing student campaign.
The new feature owns model-structure compression and algorithm evolution, while
the student campaign continues to own distillation and training policy search.

## Design choices

### Recommended: additive Wan compression subsystem

Add focused modules under `harness4h3/adapters/wan`,
`harness4h3/operators/wan`, `harness4h3/profiler`, `harness4h3/evolution`,
and `harness4h3/sandbox`. Reuse the existing evaluator, scheduler, archive,
memory, controller provider, and `LocalProcessExecutor` boundaries.

This keeps the H3-specific `ModelState`/operator assumptions intact while
giving Wan a real structural adapter. It also permits CPU fake-model tests when
CUDA, Diffusers, or the 1.3B checkpoint is unavailable.

### Rejected: make `operators/model_evolution.py` generic

This would minimize new files but would mix H3 validation, fake metric
simulation, and real Wan checkpoint mutation. The current operator explicitly
requires an H3 architecture, so changing it would risk existing H3 protocol
semantics and make real/fake evidence ambiguous.

### Rejected: make every algorithm a remote worker first

This would test deployment before the data model and safety contracts exist.
It would also make algorithm-generation and profiling impossible to test on
CPU. Remote execution remains a backend of the same adapter/operator contract.

## Architecture

```text
Wan checkpoint
      |
      v
WanModelAdapter -----> ModelProfile / SensitivityReport
      |                         |
      v                         v
CompressionOperator <----- AlgorithmAgent <----- SkillHub / ExperienceStore
      |
      v
Sandboxed algorithm execution
      |
      v
compressed checkpoint + CompressionResult
      |
      v
existing evaluator + scheduler + archive
      |
      v
EvaluationRecord + AlgorithmCandidate + ExperienceRecord
      |
      `----------------------> EvolutionLoop
```

### Wan adapter

`WanModelAdapter` is the only component allowed to know how a Wan checkpoint
stores transformer blocks and FFN projections. It exposes:

```python
class WanModelAdapter(Protocol):
    architecture_name: str

    def load_model(self, checkpoint: str, *, device: str = "cpu", dtype: str = "float32") -> Any: ...
    def inspect(self, model: Any) -> ModelStructure: ...
    def get_blocks(self, model: Any) -> tuple[BlockRef, ...]: ...
    def remove_blocks(self, model: Any, block_ids: Sequence[str]) -> StructuralChange: ...
    def prune_channels(self, model: Any, config: ChannelPruningConfig) -> StructuralChange: ...
    def save_model(self, model: Any, output_dir: Path) -> CheckpointArtifact: ...
    def reload_model(self, artifact: CheckpointArtifact) -> Any: ...
```

The real adapter uses optional Diffusers/Torch imports and fails closed when the
model layout is not recognized. The fake adapter uses a small Torch DiT-shaped
module with the same block/projection semantics. Both adapters run structural
validation after mutation: forward shapes, block count, projection dimensions,
finite parameters, and save/reload identity.

### Compression operators

`BlockPruningOperator` removes selected transformer blocks and records the exact
IDs, original count, retained count, and checkpoint digest. It must reject an
empty selection, duplicate IDs, out-of-range IDs, and removal of every block.

`FFNChannelPruningOperator` applies one channel mask to the FFN input
projection and the matching output projection. It records the original width,
retained width, mask digest, and alignment checks. It must reject invalid masks,
zero retained channels, and projections whose dimensions do not match.

Both operators return a `CompressionResult` rather than pretending a model
state is measured before evaluation:

```python
@dataclass(frozen=True)
class CompressionResult:
    status: Literal["success", "failed"]
    parent_checkpoint_sha256: str
    child_checkpoint: Optional[CheckpointArtifact]
    structural_change: Mapping[str, Any]
    failure_type: Optional[str]
    message: str
    wall_time_s: float
```

### Profiling

`profiler` consumes the adapter and a loaded model. It produces JSON-safe
records:

- per-block parameter count, estimated FLOPs, forward latency, and peak-memory
  delta;
- block sensitivity measured by leave-one-block-out output distance;
- FFN channel contribution and channel sensitivity;
- timestep buckets for high-noise, middle, and low-noise inference;
- model/checkpoint/config digests and profiling environment.

Profiling must support a deterministic fake forward function and an injected
real inference function. It must never claim real latency or memory when the
measurement backend is unavailable; the report marks unavailable fields
explicitly.

### Algorithm candidates and sandbox

An `AlgorithmCandidate` contains an ID, parent ID, source code, declared
entrypoint, policy metadata, source digest, and lifecycle status. The source
must implement a bounded function:

```python
def propose(profile: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ...
```

The function returns a serializable compression plan, not a shell command,
checkpoint path, device assignment, or arbitrary Python callable. The local
sandbox writes source and input JSON into a per-run directory and invokes the
existing no-shell executor with a fixed runner. Before execution it rejects:

- syntax errors and missing `propose`;
- imports outside an allowlist (`math`, `statistics`, `json`, `collections`);
- filesystem, process, network, dynamic-import, and bytecode escape patterns;
- output that is not a bounded JSON compression plan;
- timeout, non-zero exit, or output-size violations.

The sandbox is a safety boundary, not a security guarantee against a hostile
kernel. Real deployment still requires the existing remote host isolation.

### Agent, Skill Hub, and evolution loop

`AlgorithmAgent` receives a compact profile, current target constraints, and
retrieved experience. It uses the existing structured controller provider when
available and falls back to a deterministic seed algorithm in offline mode.
The agent response is normalized into source plus metadata and is validated by
the sandbox before it can run.

`AlgorithmStore` persists candidate source and immutable metadata under an
experiment root. `SkillHub` indexes accepted, rejected, and failed
`ExperienceRecord`/`DesignGene` entries and retrieves only records matching the
architecture, operator family, and device/evaluator context.

`EvolutionLoop` performs one bounded round:

1. load the active parent checkpoint;
2. build or load a profile;
3. retrieve relevant skills and failed fingerprints;
4. ask the agent for an algorithm candidate;
5. validate and run the candidate in the sandbox;
6. execute the resulting compression plan through the Wan adapter/operator;
7. evaluate the child with the existing evaluator boundary;
8. archive the child only after valid checkpoint and evaluation evidence;
9. append an experience record and update algorithm status;
10. select the next parent with explicit exploit/explore policy.

The loop is restartable. It never promotes a failed, unevaluated, offline-only,
or structurally invalid child. Search-policy evolution is represented as a
separate `SearchPolicy` object and is enabled only after the fixed algorithm
loop has produced enough completed rounds to compare policies.

## CLI and configuration

Add explicit commands without changing existing commands:

```text
harness4h3 profile-wan --config <file> --output <dir>
harness4h3 compress-wan --config <file> --algorithm <file> --output <dir>
harness4h3 evolve-compression --config <file> --rounds <n> --output <dir>
```

Configuration declares architecture, checkpoint, adapter backend (`fake` or
`diffusers`), evaluator, sandbox limits, budget, and search policy. Paths and
devices are bound by trusted local configuration; generated algorithm source
cannot override them.

## Error and evidence policy

Every failure has a stable type such as `unsupported_model_layout`,
`invalid_block_selection`, `projection_shape_mismatch`, `sandbox_policy`,
`algorithm_output_invalid`, `checkpoint_reload_failed`, `evaluation_failed`,
or `budget_exhausted`. Failure details are persisted without promoting the
candidate. Offline fake metrics are labeled `offline_simulation=true` and are
never used as real Wan evidence.

## Verification gates

Phase 1 passes when a fake Wan model can be profiled, block-pruned, channel-
pruned, saved, reloaded, evaluated, and archived in one test run; invalid
mutations fail closed; and the existing full test suite remains green.

Phase 2 passes when the profile JSON contains layer, sensitivity, channel, and
timestep sections with explicit measurement provenance.

Phase 3 passes when a deterministic algorithm candidate can be generated,
rejected/accepted by the sandbox, and executed into a bounded compression plan.

Phase 4 passes when two or more algorithm versions form a persisted parent
lineage, failed experiments are retrievable, and an accepted candidate is
selected only from valid evaluation evidence.

Phase 5 passes when exploit/explore allocation is persisted and a controlled
comparison can report experiments, GPU cost, and final objective values.

Real Wan2.1-1.3B GPU smoke is an additional deployment gate. Its absence does
not turn fake results into real-model claims.
