# Reuse Matrix

| Source | Reused in EvoGen Phase I | Adaptation | Deferred / not reused |
|---|---|---|---|
| Harness4H3 v0 | Append-only/redacted JSONL, evaluator isolation, candidate lineage, atomic active pointer, ToolRegistry design, offline fake testing | Video-task trajectory becomes model-optimization `ExperimentRecord`; `H*` workflow candidates are separated from `M0000+` model candidates | Prompt/workflow mutation is frozen under `legacy/` and is not in the Phase I loop |
| Existing MiniMax-H3 ComfyUI client | `/prompt`, `/history/<id>`, `/view`, bounded polling and isolated artifacts | Moved to `backends/comfyui.py`; it is an artifact/quality adapter rather than ModelState | Deployment credentials, Windows paths and manual benchmark columns |
| REEF | Harness/skill evolution concepts | Documentation and Phase II extension boundary only | Phase I runtime dependency |
| AlphaEvolve | Candidate/evaluator/archive/search separation | Immutable `ModelStore` plus external Pareto decision data | Program mutation and LLM ensemble |
| Frontis / OpenMLE | Execution-grounded AI4AI trajectories and experience | Append-only experiment records with plan, result, evaluation, failure and cost | Learned search policy before real data exists |
| HAQ | Hardware-in-the-loop, target-specific optimization | `TargetProfile`, hard constraints and normalized hardware metrics | Hardware surrogate in the first delivery |
| MobileVD / MobileWan | Domain recipe/operator inspiration | Future CreateStudent, Distill, Prune and runtime operators | Vendored training framework |
| DMD / DMD2 / Progressive Distillation | Step-distillation operator family | Deterministic fake StepDistill protocol in M1 | Real training implementation until teacher/student pipeline exists |
| CAKE / Phi-Bench | Future kernel/infra evaluation and reproducible runner/evaluator boundaries | Offline integration-test philosophy | Kernel/compiler search in Phase I |

No external implementation is vendored. The first delivery reuses infrastructure boundaries while replacing the old prompt/workflow-evolution research target.
