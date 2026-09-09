# Phase I Validation Plan

The engineering gate for delivery one is the fully offline fake closed loop. Tests require no network, GPU, ComfyUI or checkpoint and cover valid convergence, schema rejection, policy/operator rejection, operator failure, OOM, critical quality regression, budget exhaustion, repeated failure, branch-preserving Pareto behavior, crash recovery and an already-satisfied target.

Research claims remain gated. Delivery two must produce a real `M0000 → M0001` transition using a real model-level operator, then measure quality and at least one of latency, peak memory or model size on an external evaluator/hardware adapter. The efficiency metric must improve and quality regression must remain within the immutable TargetProfile threshold.

Later comparisons will hold controller model, target and experiment budget fixed across Human Recipe, Fixed Pipeline, Random Search, LLM Only and LLM + Harness4H3. Required search metrics are experiments/GPU-hours/wall-time to target, failed experiments, human interventions, best feasible quality and Pareto hypervolume. Hypervolume and statistical baselines are not claimed by the fake delivery.
