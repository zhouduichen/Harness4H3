from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.profiler import (
    CalibrationCorpus,
    CalibrationSampleSpec,
    FakeWanCalibrationProvider,
    ProfileReport,
)


def test_fake_calibration_is_deterministic_and_cpu_only():
    provider = FakeWanCalibrationProvider(hidden_size=8)
    spec = CalibrationSampleSpec("a fixed prompt", 17, (1, 8), (0.9, 0.5, 0.1), "s0")
    first = provider.build(spec)
    second = provider.build(spec)
    assert first.provenance["offline_simulation"] is True
    assert first.provenance["device"] == "cpu"
    assert torch.equal(first.payload, second.payload)


def test_calibration_corpus_digest_and_profile_report_fields():
    provider = FakeWanCalibrationProvider(hidden_size=8)
    corpus = CalibrationCorpus(
        (CalibrationSampleSpec("prompt", 3, (2, 8), (0.9, 0.5, 0.1), "s0"),),
        provider,
    )
    payload = corpus.to_dict()
    assert payload["digest"] == corpus.digest()
    assert payload["provenance"]["offline_simulation"] is True

    report = ProfileReport(
        model={"architecture": "WanFakeDiT"},
        layers=(),
        block_sensitivity=(),
        channel_sensitivity=(),
        timestep_sensitivity={
            "high_noise": {},
            "middle": {},
            "low_noise": {},
        },
        provenance={"offline_simulation": True},
    )
    restored = ProfileReport.from_dict(report.to_dict())
    assert restored.schema_version == 2
    assert restored.calibration == {}


def test_profile_report_reads_schema_v1_payload():
    report = ProfileReport.from_dict(
        {
            "schema_version": 1,
            "model": {"architecture": "WanFakeDiT"},
            "layers": [],
            "block_sensitivity": [],
            "channel_sensitivity": [],
            "timestep_sensitivity": {"high_noise": {}, "middle": {}, "low_noise": {}},
            "provenance": {"offline_simulation": True},
        }
    )
    assert report.schema_version == 1
    assert report.calibration == {}
