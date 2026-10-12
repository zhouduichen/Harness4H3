from __future__ import annotations

import os

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
