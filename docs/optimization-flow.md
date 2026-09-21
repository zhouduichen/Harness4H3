# Optimization and Evaluation Protocol

## Research question

Harness4H3 tests whether bounded planning plus independent measurement can
find feasible H3 model/runtime candidates for a declared device. It does not
assume that an LLM plan is correct and does not use Controller confidence as
an evaluation metric.

## Experimental variables

Independent variables are the registered operator and its validated
arguments. Depending on the study, this may be a model intervention
(quantization, pruning, distillation, recovery fine-tuning) or a runtime
intervention (offload, VAE tiling, lifecycle, cache, chunking).

Controlled variables include the parent checkpoint, prompt/task split, seed,
resolution, duration, frame rate, sampler, steps, CFG, text encoder, VAE,
LoRA, cache-reset policy, Controller model/prompt, evaluator, target, and
budget unless the protocol explicitly names one as the intervention.

Dependent variables include generation validity, quality score, model size,
latency, peak VRAM, energy when available, wall time, GPU hours, failed
experiments, and experiments-to-target.

## Closed-loop sequence

1. **Declare device and target.** The DeviceProfile describes capabilities;
   the immutable TargetProfile becomes the Controller's explicit goal: success
   criteria, hard constraints, objective definitions, priority order, and stop
   conditions.
2. **Measure the parent baseline.** Record checkpoint identity and benchmark
   under the controlled generation recipe.
3. **Build Controller context.** Include model/system state, remaining budget,
   registered operator schemas, recent experiments/failures, campaign summary,
   relevant Design Genes, and the current Pareto front.
4. **Propose one ExperimentPlan.** The plan states diagnosis, hypothesis,
   operator, arguments, expected effects, risks, budget, acceptance criteria,
   and stop conditions.
5. **Validate before execution.** Apply schema, policy, budget, operator, and
   declared-cost gates in fixed order.
6. **Execute the registered intervention.** No arbitrary Controller command is
   executed. A model-changing success creates a new immutable checkpoint; a
   runtime-only success references the unchanged model and a new SystemState.
7. **Benchmark independently.** Run sanity first, then dev and held-out when
   valid. Capture artifacts, quality, latency, peak VRAM, and failures.
8. **Apply continuation policy.** The evaluator and TargetProfile determine
   validity and feasibility; the fixed Harness policy returns `reject`,
   `exploratory_keep`, `pareto_keep`, or `final_accept`. A rejected candidate
   is scientifically valid negative evidence; a failed experiment is
   classified separately.
9. **Persist and continue in a loop.** Append the complete trajectory, update
   lineage and Pareto state if justified, return measured evidence to the next
   Controller call, and stop only at the goal, budget bound, critical failure,
   or absence of a validated plan.

The remote campaign follows the same ownership boundary. Imported checkpoints
are historical evidence and may be benchmarked in lineage order, but a new
training or runtime intervention can only be launched from a Controller
`ExperimentPlan` that passed the fixed validation pipeline. The campaign
records the Controller context, plan, and validation result; it does not invent
an operator or rewrite a rejected plan. The quality evaluator (or a configured
quality-review subagent behind that evaluator interface) reports evidence back
to the Controller and never executes model-changing commands itself.

## Candidate types

`ModelCandidate` represents a checkpoint lineage such as `M0000 → M0001`; the
parent never changes. `SystemCandidate` uses an `Sxxxx` lineage such as
`S0000 → S0001` and references a model through `model_ref`. A runtime-only
branch changes only the system lineage, so it cannot create a fake model child.
The evaluated search point is the pair `(model_id, system_id)`.

## Acceptance discipline

Feasibility gates are evaluated before Pareto ranking. Invalid output,
critical quality regression, missing metrics, or a hard hardware violation
cannot be compensated by improvement in another metric. Acceptance rules are
declared before observing the candidate.

Repeated comparisons must report run order, repetitions, aggregate statistics,
and cache/reset policy. Infrastructure failures are not optimization
rejections, and an unevaluated timeout supports no performance conclusion.

## Controller comparisons

The protocol is not tied to Qwen. Supported Controller adapters include a
deterministic offline controller, Ollama-hosted structured models, and an
OpenAI Responses-compatible endpoint. To compare Controllers scientifically,
hold the operator set, context schema, target, budget, task splits, worker,
evaluator, and initial state fixed; report Controller calls, invalid plans,
experiments, failures, wall time/GPU hours, and best feasible candidate.

## Remote experience loop

For the Linux L40×4 deployment, `harness4h3 remote-campaign` runs this fixed
sequence: read trainer-result JSON and SHA-256 metadata; build the immutable
lineage; evaluate the child through ComfyUI; sample system stats and
`nvidia-smi` power; calculate Q/L/M/E and
`R = αQ − βL/Lparent − γM/Mparent − δE/Eparent`; then apply gates before
promotion and Pareto update. SSH mode uses localhost tunnels; with
`--local-resources`, the same trusted path runs directly on the training
server and needs no Codex connection. Checkpoints are never copied during
import.

The Controller service is fixed within a campaign. Imported experiences are
passed through its bounded context as evidence; no Controller weights or
prompt policy are optimized. The autonomous overnight entry point repeatedly
invokes the real server-side LLM and persists every boundary, so a disconnected
client does not stop the loop.
For the remote L40×4 campaign, the default Controller is the explicitly
configured OpenAI-compatible Qwen/vLLM service; its `/v1/models` endpoint and
one structured `ExperimentPlan` probe must succeed before planning. An
unavailable service is a recorded stop/wait condition, never a silent
RuleBased fallback.
The continuous loop performs this Controller preflight before importing history
or reconciling checkpoints. While vLLM is waiting for a safe GPU group, it
records one bounded dependency event per poll interval and sleeps; it does not
repeat a full evaluation/training cycle or grow the Controller prompt with
duplicate unavailable traces. Once the service is ready, the normal cycle
resumes from the persisted campaign state.
The prompt keeps the complete list of new observation IDs, the current measured
state, and a small recent/retrieved slice of experiment recipes; it omits raw
media payloads and duplicate operator schemas. A schema-valid plan that fails a
Harness evidence or safety rule is recorded and retried at the next bounded
loop iteration rather than ending the autonomous run.
`remote-campaign` calls the Controller once per loop iteration. The Controller
selects among the enabled operators, while trusted configuration selects the
executable and launcher. The current remote worker supports real recovery,
binary step distillation, frozen-parent output distillation, and structural
block pruning. It uses a deterministic cached latent sample for training, and
the default `structural_proxy` evaluator checks media validity, motion, and
stability rather than semantic prompt fidelity. A semantic-quality claim
requires configuring an independent semantic evaluator.

Each iteration persists a goal event and an append-only Observation stream.
Historical trainer results and newly measured Q/L/M/E, hard gates, video
references, checkpoint hashes, and GPU telemetry are replayed into the next
Controller context. A plan must consume every new Observation and cite its
diagnosis evidence; the validator rejects unsupported or unconsumed evidence.
The plan also declares a preferred GPU count, an explicit elastic range when
allowed, distributed/exclusive mode, evaluation workers, and wait/replan
behavior. If the request is elastic, the scheduler may launch with any safe
2–4-GPU allocation inside that Controller-declared range; otherwise the exact
count is required. A waiting plan is persisted and retried at a later loop
boundary. The scheduler maps the request to actual GPUs and emits the fixed
worker command, `CUDA_VISIBLE_DEVICES`, and actual `nproc_per_node` in
`controller-events.jsonl`. The Controller launcher itself may use 1/2/4 cards
selected from live capacity; training cards are whatever 2–4 cards remain
above the waterline, not fixed GPU IDs. ComfyUI is released at idle boundaries,
and rejected child weights are deleted only under the exact campaign
`continuous/Mxxxx/` path; recipes, evidence, and experience remain. The
scheduler never preempts unrelated jobs.

### Idle ComfyUI sharing during overlap

On the shared four-card host, a resident primary ComfyUI process may coexist
with a TP1 Controller only when its queue is empty and its measured CUDA
allocation is at or below `COMFY_MAX_IDLE_USED_MIB` (2 GiB by
default). The launcher permits only that exact configured ComfyUI PID; any
other compute PID keeps the GPU unavailable. A campaign-owned ComfyUI lease
always wins and blocks the card, even if the resident process appears idle.

This gives the intended overlap layout when the primary ComfyUI has unloaded
its model: `GPU0 = idle ComfyUI context + Controller`, `GPU2 = held-out
evaluator`, and `GPU1/GPU3 = elastic training worker`. If the queue becomes
active or ComfyUI crosses the memory waterline while Controller is sharing the
card, the launcher terminates only its own vLLM child and returns the card to
the queue. Among same-size candidates, the launcher prefers this shareable
primary ComfyUI GPU even if another card has a small free-memory advantage.
This is a measured packing policy, not a claim that every card is always at
300 W or 100% utilization.
