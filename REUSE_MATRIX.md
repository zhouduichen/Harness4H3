# Reuse Matrix

| Source | Reuse | Adaptation in Harness4H3 | Not reused |
|---|---|---|---|
| Existing MinMax-H3 ComfyUI client | `/prompt`, `/history/<id>`, `/view`, API-format workflow validation, prompt ID retention, history polling | Small standard-library adapter with bounded timeouts and isolated output directories | Deployment scripts, credentials, machine paths |
| Existing MiniMax-H3 benchmark runners | Workflow node/input targeting and deterministic seed injection | Targets are configuration, not hard-coded model-specific constants | Windows SSH telemetry, fixed model names, benchmark prompt set, manual score columns |
| DGM | Parent-child candidates, generations, lineage | Immutable JSON candidate archive and atomic active pointer | SWE-bench flow and unrestricted repository self-modification |
| AlphaEvolve | External comparable evaluation and promotion gate | JSON subprocess evaluator plus score/min-delta/regression rules | LLM ensemble and program database |
| OpenRSI and Frontis-MA1 | Experience-driven diagnosis and small mutations | Repeated failure aggregation and finite mutation catalog | SFT, RL, learned operators, model training |
| Phi Bench | Runner/evaluator separation and reproducible fixtures | Fake ComfyUI integration tests and held-out split | Its benchmark tasks |
| CAKE | Failure-to-infrastructure feedback as a future direction | Documentation only in V1 | Compiler, IR, kernel and runtime co-evolution |

V1 writes all new MiniMax-H3-specific behavior locally. No external project is vendored or used as the system foundation.

