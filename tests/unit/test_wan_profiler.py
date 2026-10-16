from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.profiler import ProfileReport, measure_head_sensitivity, profile_layers, profile_model


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


def test_head_sensitivity_uses_attention_output_ablation_when_sequence_sample_is_available():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline")
    sample = adapter.profile_sample(model, seed=11)

    records = measure_head_sensitivity(adapter, model, sample=sample)

    assert records
    assert all(item["method"] == "attention_output_ablation" for item in records)
    assert all(item["available"] is True for item in records)
    assert all(item["approximation"] is False for item in records)
    assert all(item["importance_score"] is not None and item["importance_score"] > 0.0 for item in records)
    assert all(item["score_semantics"] == "mean_absolute_attention_output_difference" for item in records)


def test_head_sensitivity_falls_back_to_explicit_proxy_without_sequence_sample():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline")

    records = measure_head_sensitivity(adapter, model)

    assert records
    assert all(item["method"] == "qkv_projection_contribution" for item in records)
    assert all(item["approximation"] is True for item in records)
    assert all(item["fallback_reason"] == "sequence_sample_unavailable" for item in records)


def test_head_sensitivity_can_ablate_one_attention_inside_full_model_forward():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline")
    payload = {"latent": adapter.profile_sample(model, seed=17)}

    records = measure_head_sensitivity(
        adapter,
        model,
        sample=payload,
        model_forward_fn=lambda current_model, current_payload: current_model(current_payload["latent"]),
    )

    assert records
    assert all(item["method"] == "attention_output_ablation" for item in records)
    assert all(item["available"] is True for item in records)
    assert all(item["approximation"] is False for item in records)
    assert all(item["input_source"] == "captured_from_model_forward" for item in records)
    assert all(item["importance_score"] is not None and item["importance_score"] > 0.0 for item in records)
