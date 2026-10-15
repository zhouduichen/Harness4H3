# TICA-Prune v2 Algorithm and RSI Verification Plan

> **Execution note:** implement each task in this worktree, run its focused tests before moving on, and keep unrelated Student/H3 changes outside the branch.

## Global constraints

- Work only in `codex/tica-prune-v2`, based on the existing `codex/tica-prune` implementation.
- Do not add a second `AlgorithmStore`, `Memory`, `Sandbox`, or `Evaluator`; extend the existing paths.
- Do not start a new real Wan GPU experiment. Use CPU, Fake, and compatibility tests only.
- Preserve historical experiment decisions and reports; the new protocol applies only to future evaluations.
- Keep changes limited to TICA search, evaluation accounting, EvolutionLoop integration, profiling definitions, tests, and directly related documentation.

## Task 1: Audit and establish a test baseline

- Inspect the current TICA search, calibration, evaluator, operator, EvolutionLoop, Agent, SkillHub, configuration, and profiling code.
- Identify the current decision/data flow and existing reusable budget/store/evaluator abstractions.
- Run the focused TICA unit and integration tests before editing and record the baseline result.

## Task 2: Implement real stateful TICA search and budget accounting

- Add explicit search-state and candidate identity/digest data to the existing TICA search path.
- Implement genuinely distinct greedy and beam search transitions over compression decisions; make strategy, beam width, risk, activation, interaction, time weights, and evaluation budgets affect choices.
- Add shared budget/audit accounting so fast, full, and interaction evaluations consume the configured limits, with no hidden overrun.
- Measure Block-Block and Block-FFN interactions from the existing profiling/evaluation interfaces and preserve evidence in candidate records.
- Keep missing/invalid measurements missing; never infer a zero CUDA-memory result as an improvement.

## Task 3: Unify Agent and TICA decisions with EvolutionLoop promotion

- Let AlgorithmAgent proposals modify search rules as well as numeric parameters, while retaining sandbox and trusted-operator boundaries.
- Route TICA-selected candidates and Agent-produced plans through one final compression-plan representation.
- Persist `selected_candidate_id`, `selected_plan_digest`, and final-plan digest checks, and reject mismatches before promotion.
- Keep historical decisions immutable and use the new protocol only for new rounds.

## Task 4: Correct metrics, baselines, and evaluation splits

- Separate target Transformer parameter/weight metrics from complete Pipeline logical size, deduplicating hard links, symlinks, and shared weights.
- Verify real Wan FFN profiling and make device/monitoring failures explicit rather than zero-valued.
- Define Calibration/Validation/Test roles in the Evaluation Recipe and prevent test data from search or promotion decisions.
- Make baseline ablations materially different and ensure each is evaluated under the same accounting and metric definitions.
- Document the metric definitions and split rules in the existing recipe/reporting path.

## Task 5: Add regression coverage and validate compatibility

- Add CPU/Fake tests for greedy and beam behavior, parameter/rule effects, budget exhaustion, interactions, candidate identity/digests, plan unification, and no test leakage.
- Add promotion/compatibility tests for quality floors, hardware regressions, missing metrics, Pareto dominance, and multiple simultaneous improvements.
- Run focused tests, Python compilation, and the CPU full test suite; run only low-cost smoke checks if compatibility coverage leaves a targeted gap.

## Task 6: Review evidence, commit, and report

- Review the four historical Wan experiment records without rerunning them, separating observed evidence from hypotheses about RSI versus Agent + Memory.
- Summarize what the current evidence can and cannot establish about continual algorithm improvement and identify the minimum missing experiment.
- Commit only task-related files on `codex/tica-prune-v2` and push the branch if the repository remote is available.
