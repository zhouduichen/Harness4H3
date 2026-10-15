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

## TICA-Prune CPU research baseline

TICA is enabled only by the separate `tica_prune: true` configuration flag;
the legacy path and its output roots are unchanged. The reproducible Fake
configuration is `configs/evolutionary-compression.tica.fake.yaml`:

```bash
uv run --with torch --with pytest python -m harness4h3 evolve-compression \
  --config configs/evolutionary-compression.tica.fake.yaml --rounds 2 --json
```

The first algorithm is stored as `origin=human_baseline` and
`not_agent_discovered=true`. Later rounds can revise its source through
`AlgorithmAgent`; validation compares parent and child under the same fixed
calibration manifest, source-independent model parent, and explicit search
budget. Each TICA round writes `tica-profile.json`, `tica-search.json`, and
the usual algorithm/model evidence. `tica_prune` is classified as an
algorithm rule over the existing Block/FFN structural closure, not as a new
primitive search-space operator.

The current sparse-attention runtime interface is intentionally disabled. It
records an interface-only proposal and cannot mutate weights or claim GPU
speedup. Real Wan use requires an explicit cached-input manifest; a missing
manifest is rejected rather than replaced by random probes.
