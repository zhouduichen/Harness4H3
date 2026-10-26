import json

import pytest

from harness4h3.rsi.trace import CostLedger, TraceEvent, TraceStore


def _event(**overrides):
    args = dict(run_id="run-1", task_id="task-1", bundle_id="a" * 64, stage="evaluate", event="finished", status="failed", started_ns=10, ended_ns=30, payload={"input_tokens": 10, "output_tokens": None, "optimizer_updates": 0, "gpu_lease_seconds": 1.5}, failure_class="resource_limited")
    args.update(overrides)
    return TraceEvent.create(**args)


def test_trace_append_roundtrip_and_cost_unknowns_stay_unavailable(tmp_path):
    store = TraceStore(tmp_path)
    event = _event()
    store.append(event)
    assert store.read("run-1") == [event]
    summary = CostLedger.summary([event])
    assert summary["stage_elapsed_seconds_sum"] == pytest.approx(20e-9)
    assert summary["metrics"]["input_tokens"] == {"value": 10, "status": "measured"}
    assert summary["metrics"]["output_tokens"] == {"value": None, "status": "unavailable"}
    assert summary["estimated_usd"]["status"] == "unavailable"


def test_trace_rejects_invalid_failure_class_and_truncated_record(tmp_path):
    with pytest.raises(ValueError, match="failure_class_invalid"):
        _event(failure_class="unknown")
    path = tmp_path / "run-1.jsonl"
    path.write_text(json.dumps({"partial": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record_invalid"):
        TraceStore(tmp_path).read("run-1")
