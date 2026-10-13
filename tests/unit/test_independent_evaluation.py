import json

from harness4h3.adapters.wan.contracts import CheckpointArtifact
import pytest

from tools.run_wan_independent_evaluation import (
    _active_artifacts,
    _artifact_from_candidate,
    _parse_candidate_models,
    _parse_run,
)


def _write_candidate(root, model_id, checkpoint, *, digest, status="evaluated"):
    path = root / "models" / "candidates" / (model_id + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "state": {
                    "architecture_name": "WanDiffusersDiT",
                    "num_blocks": 29,
                    "hidden_size": 1536,
                    "ffn_width": 8960,
                    "parameter_count": 123,
                    "checkpoint_path": str(checkpoint),
                    "provenance": {"checkpoint_sha256": digest, "offline_simulation": False},
                },
                "status": status,
            }
        ),
        encoding="utf-8",
    )
    return path


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


def test_independent_evaluation_can_select_rejected_child_instead_of_active(tmp_path):
    baseline_path = tmp_path / "parent"
    child_path = tmp_path / "child"
    digest0 = "0" * 64
    digest3 = "3" * 64
    _write_candidate(tmp_path, "M0000", baseline_path, digest=digest0, status="active")
    _write_candidate(tmp_path, "M0003", child_path, digest=digest3)
    active_path = tmp_path / "models" / "active.json"
    active_path.write_text(json.dumps({"id": "M0000"}), encoding="utf-8")

    baseline, candidate, active_id, selection_mode = _active_artifacts(tmp_path, "M0003")

    assert baseline.sha256 == digest0
    assert candidate.sha256 == digest3
    assert candidate.path == str(child_path)
    assert active_id == "M0000"
    assert selection_mode == "explicit"


def test_independent_evaluation_fails_closed_for_missing_explicit_child(tmp_path):
    _write_candidate(tmp_path, "M0000", tmp_path / "parent", digest="0" * 64, status="active")
    (tmp_path / "models" / "active.json").write_text(json.dumps({"id": "M0000"}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="explicit candidate model record not found"):
        _active_artifacts(tmp_path, "M9999")


def test_independent_evaluation_parses_candidate_overrides_and_rejects_duplicates():
    assert _parse_candidate_models(["fixed=M0003", "coding=M0002"]) == {
        "fixed": "M0003",
        "coding": "M0002",
    }
    with pytest.raises(ValueError, match="NAME=MODEL_ID"):
        _parse_candidate_models(["fixed"])
    with pytest.raises(ValueError, match="duplicate"):
        _parse_candidate_models(["fixed=M0003", "fixed=M0002"])
