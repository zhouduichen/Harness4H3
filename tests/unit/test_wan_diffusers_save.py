from __future__ import annotations

import os
from pathlib import Path

from harness4h3.adapters.wan.diffusers import _copy_shared_components


def test_shared_wan_components_use_hardlinks_and_skip_transformer(tmp_path):
    source = tmp_path / "parent"
    target = tmp_path / "child"
    (source / "text_encoder").mkdir(parents=True)
    (source / "transformer").mkdir()
    (source / "model_index.json").write_text("{}", encoding="utf-8")
    shared = source / "text_encoder" / "model.safetensors"
    shared.write_bytes(b"unchanged")
    (source / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"parent")

    assert _copy_shared_components(source, target) is True
    child_shared = target / "text_encoder" / "model.safetensors"
    assert child_shared.read_bytes() == b"unchanged"
    assert os.stat(shared).st_ino == os.stat(child_shared).st_ino
    assert not (target / "transformer").exists()


def test_shared_wan_components_dereference_huggingface_cache_symlinks(tmp_path):
    source = tmp_path / "parent"
    target = tmp_path / "child"
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    model_index_blob = blobs / "model-index"
    model_index_blob.write_text("{\"ok\": true}\n", encoding="utf-8")
    (source / "transformer").mkdir(parents=True)
    source.mkdir(exist_ok=True)
    (source / "model_index.json").symlink_to(Path("..") / "blobs" / "model-index")
    (source / "text_encoder").mkdir()
    (source / "text_encoder" / "config.json").write_text("{}\n", encoding="utf-8")

    assert _copy_shared_components(source, target) is True
    copied_index = target / "model_index.json"
    assert copied_index.is_file()
    assert copied_index.is_symlink()
    assert copied_index.resolve() == model_index_blob.resolve()
    assert copied_index.read_text(encoding="utf-8") == "{\"ok\": true}\n"
