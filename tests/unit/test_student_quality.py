from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harness4h3.student.evaluation_manifest import (
    CacheContract,
    EvaluationManifest,
    build_manifest,
    validate_h3_cache_item,
)
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


def test_manifest_uses_one_anchor_per_long_sequence(tmp_path):
    import torch

    cache = tmp_path / "cache"
    cache.mkdir()
    for index, anchor in enumerate((True, False, False, True)):
        torch.save(
            {
                "prompt": torch.zeros(4, 8),
                "video": torch.zeros(24, 5, 16, 16),
                "caption": "a stable subject",
                "sequence_id": "sequence-%d" % (0 if index < 3 else 1),
                "sequence_anchor": anchor,
            },
            cache / ("%08d.pt" % index),
        )
    manifest = build_manifest(cache, tmp_path / "manifest.json", case_count=2, seeds=(1,))
    assert [Path(item.cache_path).name for item in manifest.cases] == ["00000000.pt", "00000003.pt"]
    assert manifest.unique_cache_items == 4


def test_manifest_covers_distinct_long_prompts_before_reusing_seed_anchors(tmp_path):
    import torch

    cache = tmp_path / "cache"
    cache.mkdir()
    for index, (prompt_id, seed) in enumerate(((0, 1), (0, 2), (1, 1), (1, 2), (2, 1))):
        torch.save(
            {
                "prompt": torch.zeros(4, 8),
                "video": torch.zeros(24, 36, 16, 16),
                "caption": "prompt %d" % prompt_id,
                "sequence_id": "prompt_%04d_seed_%d" % (prompt_id, seed),
                "window_count": 7,
                "sequence_anchor": True,
            },
            cache / ("%08d.pt" % index),
        )
    manifest = build_manifest(cache, tmp_path / "manifest.json", case_count=3, seeds=(1, 2))

    assert [item.caption for item in manifest.cases] == ["prompt 0", "prompt 1", "prompt 2"]


def test_quality_backend_fails_closed_when_model_is_unavailable(tmp_path):
    from harness4h3.student.quality import ClipTemporalQualityBackend

    backend = ClipTemporalQualityBackend(tmp_path / "missing-model", device="cpu")
    with pytest.raises(QualityBackendUnavailable):
        backend.evaluate(tmp_path / "missing.mp4", "a prompt")


def test_native_5s_contract_rejects_legacy_30s_item(tmp_path):
    import torch

    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "prompt": torch.zeros(2, 5120),
            "video": torch.zeros(36 * 64, 96),
            "audio": torch.zeros(404, 32),
            "caption": "legacy",
            "latent_frames": 36,
            "num_frames": 732,
            "height": 256,
            "width": 256,
            "teacher_generated": True,
            "synthetic_latent": False,
            "offline_simulation": False,
        },
        path,
    )
    with pytest.raises(ValueError, match="native_5s"):
        validate_h3_cache_item(path, CacheContract.native_5s())


def test_native_5s_contract_accepts_authenticated_item(tmp_path):
    import torch

    path = tmp_path / "native.pt"
    torch.save(
        {
            "prompt": torch.zeros(2, 5120),
            "video": torch.zeros(24 * 64, 96),
            "audio": torch.zeros(404, 32),
            "caption": "native",
            "latent_frames": 24,
            "num_frames": 81,
            "height": 256,
            "width": 256,
            "teacher_generated": True,
            "synthetic_latent": False,
            "offline_simulation": False,
        },
        path,
    )
    metadata = validate_h3_cache_item(path, CacheContract.native_5s())
    assert metadata["latent_frames"] == 24
