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
