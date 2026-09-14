# Remote H3 Experience Import and Continuous Optimization Design

## Goal

Turn the real MiniMax-H3 training work already running on the four-L40 SSH
server into a resumable Harness campaign. The campaign imports existing
training evidence as experience, runs real child training on the server,
benchmarks each child through the server's ComfyUI instance, computes
quality/latency/peak-memory/energy evidence, and only promotes an accepted
child to the next parent.

The Controller remains fixed. Its context may include imported experience, but
Controller fine-tuning is explicitly out of scope for this milestone.

## Current environment and constraints

The primary reachable host is the SSH alias `Jiayu-intern`, running four
NVIDIA L40 GPUs. The relevant paths are:

- H3 model store: `/data/models/MiniMax-H3`;
- experiment artifacts: `/data/models/MiniMax-H3/harness4h3`;
- ComfyUI root: `/home/intern/huangjiahao/ComfyUI`;
- ComfyUI API: `http://127.0.0.1:8188` on the remote host;
- real worker repository: `/home/intern/huangjiahao/Harness4H3`.

The worker has produced `M0005`, `M0006`, `M0007`, and `M0008`, but these
results currently contain training evidence only. Their quality and hardware
generation metrics are stale or absent. The importer must preserve this fact;
it must never infer video quality from training loss.

The remote repository is dirty. The local implementation must not overwrite,
reset, or commit remote working-tree changes. Checkpoints are approximately
66GB each, so normal operation transfers metadata only and leaves checkpoint
and video artifacts on the server.

## Architecture

The local process is the campaign coordinator. It uses a fixed, allowlisted
SSH configuration to read remote metadata, write small request files under a
campaign directory, launch a trusted remote worker, and create an SSH local
port forward to the remote ComfyUI API. Existing local Harness benchmark and
evaluator components are reused through that tunnel; generated artifacts are
downloaded only for the current benchmark record.

```text
remote JSON/evidence/logs
          │ SSH read-only import
          ▼
local ExperienceStore (metadata + remote artifact references)
          │ fixed Controller context
          ▼
ExperimentPlan → SSH trusted worker → child checkpoint on server
                                      │
                                      ▼
                         ComfyUI SSH tunnel benchmark
                                      │
                                      ▼
                  Q/L/M/E → gates → reward + Pareto + record
                                      │
                         accepted child becomes parent
```

No Controller-generated shell command, path, evaluator, target, or worker
configuration is executable. Remote commands come only from trusted campaign
configuration and fixed implementation code.

## Experience schema and import

Add an append-only `ExperienceStore` with schema version 1. Each record stores:

- stable `experience_id` and source URI;
- source file SHA-256 and import timestamp;
- experiment/model lineage (`experiment_id`, `parent_model_id`,
  `child_model_id`);
- operator and operator arguments;
- complete training metrics and child evidence references;
- optional independent evaluation, hardware metrics, decision, and reward;
- status and provenance.

Statuses are explicit:

- `training_only_unvalidated`: training succeeded but no independent quality
  evaluation is present;
- `evaluated_candidate`: evaluation exists but acceptance gates are incomplete;
- `accepted`, `rejected`, or `failed` only when the corresponding gate/worker
  result is recorded.

The importer discovers trusted `trainer_result_*.json` files under the remote
H3 harness root, joins them with request/evidence files when available, and
normalizes them without copying checkpoint bytes. Deduplication uses the
remote source path plus source SHA-256. A changed file is a new import, never
an in-place rewrite. Importing records never changes the active model pointer.

The existing `M0005–M0008` records therefore become usable historical context,
but remain non-promotable until benchmark evidence is added.

## Benchmark and continuous campaign

The campaign has one fixed evaluation recipe and a resumable state file. It
uses the existing sanity/dev/held-out task splits and a Linux MiniMax-H3 API
workflow with the server's BF16 model and text/VAE assets. For each model:

1. verify the remote checkpoint exists and its hash is stable;
2. create an isolated, non-overwriting model link in ComfyUI's diffusion-model
   directory;
3. call `/free` before switching model state;
4. run fixed prompts, seeds, resolution, frame count, sampler, CFG, and step
   count;
5. collect decoded-artifact checks and the configured independent quality
   evaluator result;
6. sample ComfyUI system stats and remote GPU power during generation;
7. persist the complete benchmark summary before making a promotion decision.

Existing imported stages are evaluated in lineage order (`M0005`, then
`M0006`, `M0007`, `M0008`) rather than retrained. After the last accepted
stage, the fixed Controller may propose the next registered operation. Only
`recovery_finetune` and `step_distill` are enabled for the current real worker;
unsupported operators fail closed.

The campaign is single-flight: at most one training or ComfyUI benchmark runs
at a time. A resume uses the append-only store and campaign checkpoint to skip
completed source hashes and never repeats a successful remote training job
merely because the local process stopped.

## Metrics, reward, and acceptance

The independent benchmark reports:

- `Q`: evaluator quality score in `[0, 1]`, with evaluator scope recorded;
- `L`: mean successful-task wall latency in seconds;
- `M`: peak GPU memory in GiB;
- `E`: integrated GPU energy in joules from sampled power.

The configured reward weights are fixed for a campaign. To preserve the
requested equation while making units comparable, the stored normalized terms
are:

```text
Qn = Q
Ln = L / baseline_L
Mn = M / baseline_M
En = E / baseline_E
R  = alpha*Qn - beta*Ln - gamma*Mn - delta*En
```

`R` is null when any required term is unavailable. Missing energy, missing
quality, invalid artifacts, or a stale metric cannot be treated as zero.

Acceptance is evaluated before Pareto ranking:

1. child checkpoint evidence is valid: optimizer update, positive gradient,
   changed trainable tensors, unchanged parent/frozen tensors, child hash, and
   reload all pass;
2. generation is valid and decodable for every required task;
3. quality drop from the same parent is within the immutable target limit;
4. no critical failure or black-frame/temporal-collapse gate is present;
5. at least one declared efficiency metric improves by the configured minimum;
6. Q, L, M, and E are all present for a research-grade accepted result.

The candidate is `rejected` when it is measured but fails a declared gate, and
`failed` when training or infrastructure fails before a valid evaluation. A
rejected or failed result is retained as experience and cannot become a
parent. Pareto ranking maximizes Q and minimizes L/M/E over feasible accepted
candidates.

The default built-in evaluator remains a structural video evaluator. Its scope
is recorded as `structural_proxy`; it may support engineering smoke runs but
does not justify a semantic-quality paper claim. A configured semantic or
composite evaluator is required for a research-grade acceptance report.

## Failure handling and safety

- SSH connection/authentication errors are `remote_unavailable` and do not
  become quality rejection.
- Missing, malformed, or hash-mismatched remote JSON is `experience_corrupt`;
  the importer skips only that record and reports the exact source.
- Training timeout, OOM, distributed failure, parent mutation, child reload
  failure, and benchmark timeout retain stable failure types.
- Remote paths must resolve under configured roots. Model deployment refuses
  to overwrite a parent or an existing link pointing elsewhere.
- Local writes use append-only JSONL or atomic replacement. Existing user
  changes and all remote artifacts are preserved.

## Verification

Unit tests cover record normalization, hash-based deduplication, remote command
quoting, model-link safety, reward nullability/normalization, and acceptance
gates. Integration tests mock SSH and ComfyUI to cover import → plan → remote
worker result → benchmark → accept/reject → resume.

The server verification sequence is:

1. import the four existing training results;
2. benchmark parent/candidate pairs with one controlled sanity run;
3. run the full fixed dev/held-out comparison for the first candidate;
4. confirm the stored record contains Q/L/M/E, decision, source hashes, and
   remote artifact references;
5. resume the campaign and confirm no duplicate training/import occurs;
6. only then allow the next distillation stage to run.

This milestone proves the Harness loop and evidence discipline. It does not
claim that any current child is accepted until the server produces the
required independent quality and hardware measurements.
