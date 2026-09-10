from __future__ import annotations

import json

import pytest

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.system_candidate import SystemCandidate
from harness4h3.archive.system_store import SystemCandidateStore, SystemStoreError
from harness4h3.h3.state import ModelState


def model() -> ModelCandidate:
    state = ModelState.from_dict(
        {
            **ModelState.fake_baseline().to_dict(),
            "model_id": "M0001",
            "parent_model_id": "M0000",
            "checkpoint_path": "D:\\models\\nvfp4.safetensors",
            "architecture_name": "MiniMax-H3",
            "quantization": {"bits": 4, "scheme": "nvfp4"},
        }
    )
    return ModelCandidate("M0001", "M0000", 1, state.checkpoint_path, state, "exp_0001", "candidate")


def test_system_candidate_references_model_without_copying_checkpoint():
    candidate = SystemCandidate.from_model_candidate(
        "C0000", model(), runtime_state={"backend": "comfyui"}, status="baseline"
    )
    assert candidate.model_ref == "M0001"
    assert candidate.parent_id is None
    assert candidate.evaluation_state(model().state).model_id == "M0001"
    assert candidate.evaluation_state(model().state).checkpoint_path == "D:\\models\\nvfp4.safetensors"
    assert candidate.to_dict()["runtime_state"] == {"backend": "comfyui"}


def test_system_store_enforces_c_lineage_and_atomic_persistence(tmp_path):
    store = SystemCandidateStore(tmp_path)
    root = SystemCandidate.from_model_candidate("C0000", model(), status="baseline")
    store.initialize(root)
    child = SystemCandidate(
        "C0001",
        "C0000",
        1,
        "M0001",
        runtime_state={"runtime_recipe": [{"kind": "vae_tiling", "args": {"tile_size": 256, "overlap": 32}}]},
        created_by_experiment_id="exp_0002",
        status="rejected",
    )
    store.create(child)
    assert store.next_id() == "C0002"
    assert store.children("C0000")[0].id == "C0001"
    assert json.loads((tmp_path / "candidates" / "C0001.json").read_text())[
        "runtime_state"
    ]["runtime_recipe"][0]["kind"] == "vae_tiling"

    with pytest.raises(SystemStoreError, match="generation"):
        store.create(SystemCandidate("C0002", "C0000", 4, "M0001"))


def test_system_candidate_rejects_model_id_as_system_id():
    with pytest.raises(ValueError, match="system candidate id"):
        SystemCandidate("M0002", None, 0, "M0001")
