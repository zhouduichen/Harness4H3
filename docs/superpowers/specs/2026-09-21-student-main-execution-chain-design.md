# Student Main Execution Chain Design

## Goal

Make the existing Harness4H3 Student campaign control plane the default path for
both `student-campaign run` and the detached supervisor. A real run must leave
durable evidence for the complete chain:

```text
multi-candidate -> Advocate/Critical -> algorithm dispatch -> parent checkpoint
inheritance -> multi-fidelity -> semantic verifier -> hard/Pareto gate
-> archive/experience
```

The work is an integration repair. It does not add a second campaign engine,
new storage system, or new horizontal module family.

## Fail-closed boundary

The default Student run never falls back to the legacy single-proposal path.
Offline fake workers remain available to unit and integration tests when they
are injected explicitly. A production CLI or detached run fails with a typed
result if the configured remote worker, semantic evaluator, parent checkpoint,
or algorithm dispatch cannot produce evidence.

`structural_proxy` is not an acceptable production semantic verifier. The
default Student campaign requires `clip_temporal` and a fixed evaluation
manifest; validation may still inspect a config without contacting the remote
host.

## Existing components and changes

1. `StudentCampaign` receives a persisted `CampaignBase`, a real
   `CapabilitySnapshot`, and a `ReviewPipeline` by default from the CLI and
   detached supervisor. The immutable base binds target, teacher/cache
   manifest, verifier bank, evaluation recipe, actor identities, and supported
   algorithms.
2. The existing `ReviewPipeline` is used with deterministic trusted Student
   Advocate/Critical/Modifier agents. Their reports are appended to the
   existing `DecisionTrace`; no candidate reaches the worker without these
   reports.
3. `StudentCampaignAdapter` passes the current parent checkpoint and fidelity
   to the existing worker protocol. The first round is rooted at the trusted H3
   teacher; after `parent.selected`, the selected Student child becomes the
   next round parent. Within a candidate, a successful lower-fidelity child is
   the parent of the next fidelity.
4. `StudentTrainWorker` dispatches `velocity_distill` and `dmd2` to the
   existing `h3_training` algorithm implementations through a narrow
   Student-shaped adapter. The worker records the selected algorithm,
   dispatch status, fidelity, parent hash, inherited tensor count, and child
   evidence. It never reports the old generic MSE loop as a production success.
5. `StudentCampaignAdapter.verify` maps the existing evaluator and
   `MetricVerifierBank` records into hard semantic/decodability evidence and
   continuous quality, latency, memory, size, and optional energy evidence.
   The existing `AcceptanceGate` evaluates hard constraints first. The control
   loop then applies `pareto_dominates` across feasible candidates before
   selecting a parent.
6. Existing archive and experience JSONL files remain the sinks. Their records
   gain the actual checkpoint hashes, parent identity, fidelity history,
   semantic evidence, gate result, and algorithm dispatch evidence needed to
   replay the chain.

## Data flow and recovery

The CLI and supervisor build the same campaign object. The base is written
atomically at `<output>/campaign-base.json`; an existing base is loaded and
must have the same digest. Each round writes `proposal.generated`,
`proposal.validated`, `critic.completed`, `training.started/completed`,
`evaluation.started/completed`, `gate.decided`, `archive.updated`, and
`parent.selected` events to the existing base-bound decision trace.

`parent.selected` includes the selected checkpoint path and SHA-256. On resume,
the campaign reconstructs both candidate lineage and parent checkpoint from
that event. A missing or inconsistent parent is a hard campaign failure, never
an implicit reset to the teacher.

Fidelity budgets are deterministic (`F1`, `F2`, `F3` by default) and are
derived from the trusted full-step budget. Every fidelity launch has its own
directory and result record. Only the final candidate decision can promote or
retain a child; intermediate children are marked in-flight until the next
fidelity succeeds.

## Verification and acceptance

The implementation is accepted only when tests and a scripted evidence run
prove all of the following:

- default entrypoint construction injects control-plane state and cannot use
  the legacy path;
- one round produces at least three candidates and durable Advocate/Critical
  reports;
- the worker records an executed existing algorithm name, not a generic loop;
- a second round and a higher fidelity receive the preceding child checkpoint
  hash as their parent;
- semantic verifier evidence is present and structural proxy evidence alone is
  rejected by the production gate;
- hard constraints reject an invalid artifact before Pareto selection;
- Pareto selection chooses among feasible candidates and archives both
  promoted and rejected candidates;
- experience records contain predicted/actual evidence, parent/child hashes,
  fidelity, semantic evidence, and gate outcome;
- the existing full test suite remains green, with fake/CPU tests clearly
  marked as contract evidence rather than remote-model evidence.

