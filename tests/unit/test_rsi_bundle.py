import json

import pytest

from harness4h3.rsi.bundle import HarnessBundle


def test_bundle_identity_is_content_addressed_and_restorable(tmp_path):
    source = tmp_path / "seed"
    source.mkdir()
    (source / "context.py").write_text("def build_context(x): return x\n", encoding="utf-8")
    (source / "ignored.log").write_text("noise", encoding="utf-8")
    store = tmp_path / "store"
    first = HarnessBundle.capture(source, store, parent_id=None, hypothesis={"cause": "test"}, allowed_paths=["context.py"])
    second = HarnessBundle.capture(source, store, parent_id=None, hypothesis={"cause": "different metadata"}, allowed_paths=["context.py"])
    assert first.bundle_id == second.bundle_id
    assert "ignored.log" not in first.files
    restored = first.restore(store, tmp_path / "restored")
    assert (restored / "context.py").read_text(encoding="utf-8") == first.files["context.py"]


def test_bundle_rejects_missing_and_tampered_files(tmp_path):
    source = tmp_path / "seed"
    source.mkdir()
    (source / "context.py").write_text("x = 1\n", encoding="utf-8")
    store = tmp_path / "store"
    bundle = HarnessBundle.capture(source, store, parent_id=None, hypothesis={}, allowed_paths=["context.py"])
    with pytest.raises(ValueError, match="digest_mismatch"):
        (store / bundle.bundle_id / "context.py").write_text("x = 2\n", encoding="utf-8")
        HarnessBundle.load(store, bundle.bundle_id)


def test_bundle_rejects_paths_outside_the_allowlist_shape(tmp_path):
    source = tmp_path / "seed"
    source.mkdir()
    (source / "context.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_relative"):
        HarnessBundle.capture(source, tmp_path / "store", parent_id=None, hypothesis={}, allowed_paths=["../context.py"])
