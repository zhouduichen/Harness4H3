# Harness Evolution (EvoCompress)

This package contains the additive algorithm-evolution path for Wan-shaped
DiT compression. It is separate from the existing Wan student training
campaign: student training owns distillation policy search, while this package
owns structural compression, generated algorithm code, and compression
experience.

The current runtime pieces are:

- `AlgorithmAgent`: deterministic offline generation or a bounded structured
  provider response;
- `AlgorithmSandbox`: AST-checked, no-shell execution of `propose(profile,
  history)`;
- `AlgorithmStore`: immutable source lineage and experiment metadata;
- `SkillHub`: architecture/operator-aware retrieval from append-only memory;
- `EvolutionLoop`: profile → generate → sandbox → compress → evaluate → record.

`evolve-compression` can use `controller.provider: deterministic` for offline
tests or `ollama`, `vllm`, `openai_compatible`, and `openai` for a structured
JSON LLM request. The provider receives profile/history context and can only
return `source`, `hypothesis`, and the already-bound `parent_id`; the sandbox
remains the execution gate.

CPU fake verification:

```bash
.venv/bin/python -m harness4h3 profile-wan \
  --config configs/evolutionary-compression.fake.yaml --json
.venv/bin/python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.fake.yaml --rounds 2 --json
```

All fake results are marked `offline_simulation=true`. Real Wan2.1-T2V-1.3B
execution uses the optional Diffusers adapter and must be run on a host with
the checkpoint, CUDA, and the matching Diffusers dependency set.
