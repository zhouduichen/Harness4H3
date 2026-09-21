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

The normalized terms are combined as:

```text
R = αQ − βL − γM − δE − εS
```

`V_energy` is optional because the current remote worker does not always have
power telemetry. Missing energy is recorded explicitly and omitted from the
finite reward; it is never inferred. Required metrics remain fail-closed for
reward computation.

The JSON result keeps all raw values, normalized values, units, references,
sources, missing metrics, invalid metrics, reward terms, and the scalar reward.
This lets the LLM receive both the interpretable vector and the compact scalar
objective in the next campaign context.
