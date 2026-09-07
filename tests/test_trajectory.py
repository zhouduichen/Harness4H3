from __future__ import annotations

from harness4h3.memory.trajectory import Trajectory, TrajectoryStore


def test_trajectory_is_append_only_and_redacts_secrets(tmp_path):
    store = TrajectoryStore(tmp_path / "runs.jsonl")
    item = Trajectory("t", "H0", "dev", {"api_token": "secret"}, [], None, 0.0, "failed", {"wall_time": 1.0})
    store.append(item)
    store.append(item)
    records = list(store.read())
    assert len(records) == 2
    assert records[0].inputs["api_token"] == "[REDACTED]"

