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
   the immutable TargetProfile defines constraints and priorities.
2. **Measure the parent baseline.** Record checkpoint identity and benchmark
   under the controlled generation recipe.
3. **Build Controller context.** Include normalized state, remaining budget,
   registered operator schemas, recent experiments/failures, relevant Design
   Genes, and the current Pareto front.
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
8. **Accept, reject, or fail.** The evaluator and TargetProfile determine the
   result. A rejected candidate is scientifically valid negative evidence; a
   failed experiment is classified separately.
9. **Persist and continue.** Append the complete trajectory, update lineage
   and Pareto state if justified, then return measured evidence to the next
   Controller call.

The remote campaign follows the same ownership boundary. Imported checkpoints
are historical evidence and may be benchmarked in lineage order, but a new
training or runtime intervention can only be launched from a Controller
`ExperimentPlan` that passed the fixed validation pipeline. The campaign
records the Controller context, plan, and validation result; it does not invent
an operator or rewrite a rejected plan. The quality evaluator (or a configured
quality-review subagent behind that evaluator interface) reports evidence back
to the Controller and never executes model-changing commands itself.

## Candidate types

`ModelCandidate` represents a checkpoint lineage such as
`M0000 → M0001`. The parent never changes. `SystemCandidate` represents a
runtime recipe applied to a model reference; it must not duplicate or relabel
the underlying checkpoint.

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
sequence: read trainer-result JSON and SHA-256 metadata over SSH; build the
immutable lineage; evaluate the child through a localhost SSH tunnel to remote
ComfyUI; sample system stats and `nvidia-smi` power; calculate Q/L/M/E and
`R = αQ − βL/Lparent − γM/Mparent − δE/Eparent`; then apply gates before
promotion and Pareto update. Checkpoints are never copied during import.

The Controller is fixed. Imported experiences are passed through its existing
context as evidence; no Controller weights or prompt policy are optimized.
Every bounded run still records the Controller's next validated plan in the
campaign report; the `--max-experiments` limit controls whether that plan may
be executed in the current run, not whether the Controller is bypassed.
The current remote worker uses a deterministic cached latent sample for
training, and the default `structural_proxy` evaluator checks media validity,
motion, and stability rather than semantic prompt fidelity. A semantic-quality
claim requires configuring an independent semantic evaluator.
