from __future__ import annotations

from pathlib import Path

from harness4h3.adapters.wan import FakeWanDiTAdapter
from tools.evaluate_wan_checkpoint import _artifact_from_path


def test_independent_evaluation_artifact_uses_checkpoint_content_hash(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline")
    checkpoint = adapter.save_model(model, tmp_path / "checkpoint")

    artifact = _artifact_from_path(adapter, Path(checkpoint.path), device="cpu", dtype="float32")

    assert artifact.path == str(Path(checkpoint.path).resolve())
    assert artifact.sha256 == checkpoint.sha256
    assert artifact.structure.checkpoint_sha256 == checkpoint.sha256
    assert artifact.metadata["independent_evaluation"] is True
