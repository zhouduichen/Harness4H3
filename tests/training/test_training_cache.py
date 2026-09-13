import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from h3_training.data.cache import TrainingCache
from h3_training.engine.state import TrainingFailure


def key(cache, **changes):
    material = {
        "kind": "teacher_prediction",
        "sample_digest": "sample-a",
        "model_digest": "model-a",
        "preprocessing": {"crop": "center"},
        "modality": "video",
        "dtype": "float32",
        "shape": [1, 2],
        "schedule": [1.0, 0.0],
        "timestep": 0.5,
        "seed": 1,
        "conditioning_digest": "condition-a",
    }
    material.update(changes)
    return cache.key(**material)


def test_teacher_cache_key_changes_for_every_semantic_dimension(tmp_path):
    cache = TrainingCache(tmp_path)
    base = key(cache)
    changes = {
        "sample_digest": "sample-b",
        "model_digest": "model-b",
        "preprocessing": {"crop": "random"},
        "modality": "audio",
        "dtype": "float16",
        "shape": [2, 2],
        "schedule": [1.0, 0.5, 0.0],
        "timestep": 0.2,
        "seed": 2,
        "conditioning_digest": "condition-b",
    }
    assert all(base != key(cache, **{name: value}) for name, value in changes.items())


def test_cache_roundtrip_and_corrupt_payload_fails_closed(tmp_path):
    cache = TrainingCache(tmp_path)
    cache_key = key(cache)
    manifest = cache.store(cache_key, {"prediction": torch.arange(4).reshape(2, 2).float()}, {"source": "test"})
    tensors, loaded = cache.load(cache_key)
    assert torch.equal(tensors["prediction"], torch.arange(4).reshape(2, 2).float())
    assert loaded == manifest
    _, payload, _ = cache.paths(cache_key)
    payload.write_bytes(b"broken")
    with pytest.raises(TrainingFailure, match="cache_corrupt"):
        cache.load(cache_key)


def test_missing_key_is_a_cache_miss(tmp_path):
    cache = TrainingCache(tmp_path)
    with pytest.raises(KeyError):
        cache.load(key(cache))
