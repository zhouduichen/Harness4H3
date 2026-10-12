from __future__ import annotations

from pathlib import Path

from harness4h3.adapters.wan.diffusers import _copy_full_checkpoint


def test_full_checkpoint_fallback_preserves_pipeline_components(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model_index.json").write_text("{}\n", encoding="utf-8")
    (source / "transformer").mkdir()
    (source / "transformer" / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "vae").mkdir()
    (source / "vae" / "config.json").write_text("{}\n", encoding="utf-8")

    target = tmp_path / "target"
    assert _copy_full_checkpoint(source, target) is True
    assert (target / "model_index.json").is_file()
    assert (target / "transformer" / "config.json").is_file()
    assert (target / "vae" / "config.json").is_file()
