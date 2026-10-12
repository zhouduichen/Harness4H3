import json

from harness4h3.adapters.wan.contracts import CheckpointArtifact
from tools.run_wan_independent_evaluation import _artifact_from_candidate, _parse_run


def test_independent_evaluation_parses_run_and_preserves_checkpoint_identity(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    digest = "a" * 64
    record = {
        "state": {
            "architecture_name": "WanDiffusersDiT",
            "num_blocks": 29,
            "hidden_size": 1536,
            "ffn_width": 8960,
            "parameter_count": 123,
            "checkpoint_path": str(checkpoint),
            "provenance": {"checkpoint_sha256": digest, "offline_simulation": False},
        }
    }
    path = tmp_path / "M0001.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    artifact = _artifact_from_candidate(path)

    assert isinstance(artifact, CheckpointArtifact)
    assert artifact.sha256 == digest
    assert artifact.structure.block_count == 29


def test_independent_evaluation_requires_name_and_root_separator(tmp_path):
    spec = _parse_run("rsi=%s" % tmp_path, 3)
    assert spec.name == "rsi"
    assert spec.root == tmp_path.resolve()
