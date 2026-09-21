from __future__ import annotations

import json

import numpy as np
import pytest

from harness4h3.student.evaluation_manifest import EvaluationManifest, build_manifest
from harness4h3.student.quality import QualityBackendUnavailable, aggregate_quality


def test_aggregate_quality_is_bounded_and_weighted():
    result = aggregate_quality(semantic=0.8, temporal=0.9, motion=0.7)

    assert result == pytest.approx(0.81)


def test_manifest_digest_is_stable_and_records_caption_and_reused_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    item = {"prompt": np.zeros((4, 8), dtype=np.float32), "video": np.zeros((24, 5, 16, 16), dtype=np.float32), "caption": "a red bird", "latent_frames": 5}
    import torch

    torch.save(item, cache / "00000000.pt")
    output = tmp_path / "manifest.json"
    manifest = build_manifest(cache, output, case_count=4, seeds=(1, 2))
    loaded = EvaluationManifest.from_path(output)

    assert loaded.digest == manifest.digest
    assert len(loaded.cases) == 4
    assert loaded.unique_cache_items == 1
    assert loaded.cases[0].caption == "a red bird"
    assert loaded.cases[0].seeds == (1, 2)
    assert json.loads(output.read_text())["digest"] == manifest.digest


def test_manifest_rejects_missing_caption(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    import torch

    torch.save({"prompt": torch.zeros(4, 8), "video": torch.zeros(24, 5, 16, 16)}, cache / "00000000.pt")

    with pytest.raises(ValueError, match="caption"):
        build_manifest(cache, tmp_path / "manifest.json", case_count=1, seeds=(1,))


def test_quality_backend_fails_closed_when_model_is_unavailable(tmp_path):
    from harness4h3.student.quality import ClipTemporalQualityBackend

    backend = ClipTemporalQualityBackend(tmp_path / "missing-model", device="cpu")
    with pytest.raises(QualityBackendUnavailable):
        backend.evaluate(tmp_path / "missing.mp4", "a prompt")
