from pathlib import Path

import pytest

pytest.importorskip("torch")

from h3_training.tiny.factory import load_tiny_checkpoint
from research.experiments.tiny_real_closed_loop import run_tiny_real_closed_loop


def test_real_tiny_training_creates_evaluated_lineage(tmp_path):
    result = run_tiny_real_closed_loop(tmp_path / "campaign", max_experiments=2, seed=11)
    lineage = result.model_store.lineage()
    assert [item.id for item in lineage] == ["M0000", "M0001", "M0002"]
    assert lineage[1].parent_id == "M0000"
    assert lineage[2].parent_id == "M0001"
    assert all(Path(item.checkpoint_path).is_file() for item in lineage)
    assert all(item.state.provenance.get("offline_simulation") is False for item in lineage[1:])
    assert result.report["total_experiments"] == 2
    assert not result.report["failed_experiments"]
    assert not result.report["rejected_experiments"]
    assert [load_tiny_checkpoint(Path(item.checkpoint_path))[1]["sampling_nfe"] for item in lineage] == [4, 4, 2]
