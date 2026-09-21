from tools.h3_checkpoint_stage import stage_checkpoint


def test_checkpoint_stage_is_atomic_cached_and_bounded(tmp_path):
    source = tmp_path / "parent.safetensors"
    source.write_bytes(b"checkpoint-payload")
    stage_dir = tmp_path / "stage"

    first = stage_checkpoint(source, stage_dir)
    target = stage_dir / "stale.safetensors"
    target.write_bytes(b"old")
    second = stage_checkpoint(source, stage_dir)

    assert first["status"] == "ready"
    assert first["action"] == "copied"
    assert second["action"] == "cached"
    staged = stage_dir / source.name.replace(
        ".safetensors",
        "-%d-%d.safetensors" % (source.stat().st_size, source.stat().st_mtime_ns),
    )
    assert staged.read_bytes() == source.read_bytes()
    assert not target.exists()
    assert sorted(path.name for path in stage_dir.glob("*.safetensors")) == [staged.name]
