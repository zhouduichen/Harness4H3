import pytest

from harness4h3.memory.observation import ControllerEventStore


def test_controller_event_store_tail_reads_only_newest_bounded_events(tmp_path):
    store = ControllerEventStore(tmp_path / "controller-events.jsonl")
    for index in range(6):
        store.append("pipeline_stage", {"stage": "stage-%d" % index})

    tail = store.tail(2)

    assert [item["stage"] for item in tail] == ["stage-4", "stage-5"]


def test_controller_event_store_tail_rejects_invalid_limit(tmp_path):
    store = ControllerEventStore(tmp_path / "controller-events.jsonl")

    with pytest.raises(ValueError, match="tail limit"):
        store.tail(0)
