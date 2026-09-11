# EvoGen-RSI Research Scope

Phase I asks whether a fixed strong LLM, an H3-specific restricted Harness and real evaluation can automatically discover an H3-derived student that improves hardware efficiency without exceeding quality loss constraints. The Phase-I Harness is now frozen as `Harness4H3-v1.0`; its protocol, evaluator, archive, trajectory and acceptance semantics are fixed except for correctness/security bugfixes. The current research variable is the H3-derived model/runtime system, not the Harness. It does not claim recursive self-improvement.

The active inner loop is:

```text
fixed Harness v1.0
  → Controller diagnoses current state
  → Controller chooses one registered experiment tool
  → real execution and independent verification
  → state/action/result experience
  → next Controller decision
```

M6 runtime recipe continuation is therefore an autonomous model/system
optimization campaign. A rejected `vae_tiling` hypothesis is feedback for the
next action, not a reason to redesign the execution environment.

Phase II begins only after stable real execution, multiple operators and tens of meaningful trajectories. It may derive Experiment Summaries and Design Genes, retrieve hardware-conditioned experience, adjust operator priors, parent selection and budget allocation, and compare `Harness_k` under the same fixed LLM and benchmark. Evaluator thresholds remain outside the evolution boundary. Until that phase is explicitly opened, no accumulated result may silently change the frozen Harness.

Phase III may use accumulated state/action/result/reward data for controller post-training. Controller SFT/RL, multi-agent systems, automatic source modification, kernel generation, compiler/IR co-evolution, online edge adaptation and a general AutoML platform are outside the current repository scope.
