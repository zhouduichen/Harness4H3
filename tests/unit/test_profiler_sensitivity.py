from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.profiler.sensitivity import measure_channel_sensitivity


def test_full_channel_profile_covers_every_fake_ffn_channel_in_chunks():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=10)
    model = adapter.load_model("fake://baseline")
    rows = measure_channel_sensitivity(adapter, model, method="weight", chunk_size=3, full=True)
    assert [len(row["channels"]) for row in rows] == [10, 10]
    assert {channel["channel"] for channel in rows[0]["channels"]} == set(range(10))
    assert all(channel["evidence"]["status"] == "estimated" for channel in rows[0]["channels"])
    assert rows[0]["evidence"]["status"] == "estimated"


def test_legacy_channel_profile_keeps_probe_limit():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=10)
    rows = measure_channel_sensitivity(adapter, adapter.load_model("fake://baseline"), max_probes=3)
    assert all(len(row["channels"]) == 3 for row in rows)


def test_unsupported_activation_method_is_unavailable_not_zero():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=10)
    model = adapter.load_model("fake://baseline")
    rows = measure_channel_sensitivity(adapter, model, method="activation", full=True)
    assert all(row["available"] is False for row in rows)
    assert all(row["evidence"]["status"] == "unavailable" for row in rows)
    assert all(channel["importance_score"] is None for channel in rows[0]["channels"])


def test_output_perturbation_channel_profile_measures_model_delta():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=10)
    model = adapter.load_model("fake://baseline")
    sample = adapter.profile_sample(model, seed=13)

    rows = measure_channel_sensitivity(
        adapter,
        model,
        max_probes=3,
        method="output_perturbation",
        sample_set=(sample,),
        forward_fn=adapter.forward_profile,
    )

    assert all(row["available"] is True for row in rows)
    assert all(channel["method"] == "output_perturbation" for row in rows for channel in row["channels"])
    assert all(channel["approximation"] is False for row in rows for channel in row["channels"])
    assert all(channel["importance_score"] is not None and channel["importance_score"] > 0.0 for row in rows for channel in row["channels"])
    assert all(channel["score_semantics"] == "mean_absolute_model_output_difference_after_zeroing_ffn_channel" for row in rows for channel in row["channels"])


def test_output_perturbation_without_samples_is_explicitly_unavailable():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=10)
    rows = measure_channel_sensitivity(adapter, adapter.load_model("fake://baseline"), method="output_perturbation")

    assert all(row["available"] is False for row in rows)
    assert all(channel["evidence"]["reason"] == "sample_set_unavailable" for row in rows for channel in row["channels"])
