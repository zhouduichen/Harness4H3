from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.profiler import ProfileReport, profile_layers, profile_model


def test_profile_report_contains_required_sections_and_explicit_provenance(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    report = profile_model(adapter, adapter.load_model("fake://baseline"), seed=7)
    payload = report.to_dict()
    assert set(payload) >= {"model", "layers", "block_sensitivity", "channel_sensitivity", "timestep_sensitivity", "provenance"}
    assert set(payload["timestep_sensitivity"]) == {"high_noise", "middle", "low_noise"}
    assert payload["provenance"]["offline_simulation"] is True
    assert len(payload["layers"]) == 3
    assert all(item["estimated_flops"] > 0 for item in payload["layers"])
    report.write_json(tmp_path / "profile.json")
    assert ProfileReport.read_json(tmp_path / "profile.json").to_dict() == payload


def test_missing_real_measurements_are_explicitly_unavailable():
    adapter = FakeWanDiTAdapter()
    report = profile_model(adapter, adapter.load_model("fake://baseline"), measure_latency=False)
    assert report.provenance["latency_available"] is False
    assert all(item["measurement"]["memory_available"] is False for item in report.layers)
    assert all(item["available"] is True for item in report.timestep_sensitivity.values())


def test_layer_latency_is_measured_on_the_corresponding_block_not_whole_model():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline")
    original = model.blocks[0].forward

    def slower(value):
        time.sleep(0.004)
        return original(value)

    model.blocks[0].forward = slower
    records = profile_layers(adapter, model, latency_warmup=0, latency_repeats=2)
    values = {item["block_id"]: item["latency_ms"] for item in records}
    assert values["block.0"] is not None
    assert values["block.1"] is not None
    assert values["block.0"] > values["block.1"]
    assert all("whole_model" not in str(item["measurement"]["latency_method"]) for item in records)
