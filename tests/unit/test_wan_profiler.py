from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.profiler import ProfileReport, profile_model


def test_profile_report_contains_required_sections_and_explicit_provenance(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    report = profile_model(adapter, adapter.load_model("fake://baseline"), seed=7)
    payload = report.to_dict()
    assert set(payload) >= {"model", "layers", "block_sensitivity", "channel_sensitivity", "timestep_sensitivity", "provenance"}
    assert set(payload["timestep_sensitivity"]) == {"high_noise", "middle", "low_noise"}
    assert payload["provenance"]["offline_simulation"] is True
    assert len(payload["layers"]) == 3
    report.write_json(tmp_path / "profile.json")
    assert ProfileReport.read_json(tmp_path / "profile.json").to_dict() == payload


def test_missing_real_measurements_are_explicitly_unavailable():
    adapter = FakeWanDiTAdapter()
    report = profile_model(adapter, adapter.load_model("fake://baseline"), measure_latency=False)
    assert report.provenance["latency_available"] is False
    assert all(item["measurement"]["memory_available"] is False for item in report.layers)
    assert all(item["available"] is False for item in report.timestep_sensitivity.values())

