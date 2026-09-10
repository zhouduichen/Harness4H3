# Harness4H3 v1.0 Freeze and Autonomous Optimization Campaign

## Decision

Harness4H3 Phase I is now a frozen optimization environment identified as
`Harness4H3-v1.0`. Its Controller protocol, state/schema contracts, evaluator,
archive, trajectory format, acceptance policy, and TargetProfile semantics are
fixed for the current research campaign. Changes to those components are
limited to correctness and security bug fixes; a research result must not be
attributed to an untracked Harness change.

The package and every new M6 evidence/trajectory record expose the same
version, `frozen` status, and `bugfix_only` change policy. Existing archive
compatibility is retained for historical replay, but no new Harness Evolution
is part of the Phase-I campaign.

## Optimization boundary

The object being optimized is the H3-derived model/runtime system. Runtime
operators are experiment tools, not Harness evolution: adding or selecting a
capability-guarded operator is allowed only when it preserves the existing
execution contracts and records its result through the normal evidence path.
The Controller remains the only proposer of the next experiment; people do not
preselect the next runtime intervention during a campaign.

## Inner-loop campaign

`experiments/m6_runtime_recipe.py` is the bounded autonomous campaign entry
point. Each iteration passes the fixed TargetProfile, current state, available
operator schemas, prior experiment results, rejected Design Genes, and budget
to the Controller. It then executes exactly one legal operator, validates dev
and held-out splits, records the state/action/result tuple, and feeds the
failure or accepted candidate into the next iteration.

The campaign stops only when both acceptance splits satisfy the strict
TargetProfile, the iteration/failure budget is exhausted, or an explicit
capability/Controller failure reaches the configured failure limit. No failure
is converted into a successful feasibility claim.

## Research phases

The current work is **autonomous model/system optimization** under a fixed
Harness and fixed Controller. Harness Evolution remains a later phase that may
learn operator priors, retrieval, or budget allocation from accumulated
trajectories. Controller post-training is later still. Until those transitions
are explicitly started, the core Harness is not a research variable.

## Verification

The freeze metadata is unit-tested, runtime recipe trajectories carry
`Harness4H3-v1.0`, and the existing offline suite must remain green. Real M6
acceptance still requires generation/decode validity, zero black-frame rate,
quality-drop limits, retained M5.5 efficiency, and `peak_vram_max_gb <= 16.0`
on both dev and held-out splits.
