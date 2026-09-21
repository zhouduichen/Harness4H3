# Metric Verifier Bank

The Student campaign separates two kinds of evidence:

1. Hard verifiers decide whether an artifact is safe to consider at all:
   generation validity, decode success, finite pixels, and non-empty video.
2. Continuous metric verifiers measure trade-offs between otherwise valid
   candidates. A poor metric cannot hide a hard-verifier failure.

The first bank is implemented by
`harness4h3.student.metrics.MetricVerifierBank` and emits:

| Verifier | Evidence | Direction | Default reference |
| --- | --- | --- | --- |
| `V_quality` | `quality.score` | maximize | `1.0` |
| `V_latency` | `hardware.latency_s` converted to ms | minimize | `100ms` |
| `V_memory` | `hardware.peak_memory_gb` | minimize | `8GB` |
| `V_energy` | `hardware.energy_j` | minimize | `12J` |
| `V_size` | `hardware.model_size_gb` or checkpoint bytes | minimize | `4.7GB` |

The raw bank is normalized against the H3 teacher baseline for campaign decisions. Quality is a ratio (`Q_student / Q_H3`); latency, memory, size, and optional energy are efficiency ratios where lower is better. The bounded log deltas are combined as:

```text
R = αQ − βL − γM − δE − εS
```

The default weights are `α=.60`, `β=.20`, `γ=.10`, `δ=.05`, and `ε=.05` when energy is available. The strict policy also enforces a quality floor, a minimum reward improvement or material efficiency improvement, and a maximum per-metric regression. Thus a faster but visibly worse video cannot win by latency alone. Valid candidates that fail this gate are retained as rejection evidence and sent back to the LLM for the next architecture/training proposal.

`V_energy` is optional because the current remote worker does not always have
power telemetry. Missing energy is recorded explicitly and omitted from the
finite reward; it is never inferred. Required metrics remain fail-closed for
reward computation.

The JSON result keeps all raw values, normalized values, units, references,
sources, missing metrics, invalid metrics, reward terms, and the scalar reward.
This lets the LLM receive both the interpretable vector and the compact scalar
objective in the next campaign context. `structural_proxy` is intentionally
labeled as a smoke-test diagnostic; production runs use the fixed-manifest
`clip_temporal` backend and fail closed when its local model or dependencies
are unavailable.
