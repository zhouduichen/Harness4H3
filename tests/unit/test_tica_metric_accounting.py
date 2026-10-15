from __future__ import annotations

import os

from harness4h3.adapters.wan.diffusers import DiffusersWanAdapter
from harness4h3.evolution.algorithm_evaluation import _logical_bytes


class _Linear:
    in_features = 4
    out_features = 8


class _Block:
    up_proj = _Linear()
    activation = object()
    down_proj = _Linear()


def test_logical_checkpoint_size_deduplicates_hardlinks_and_symlinks(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789")
    root = tmp_path / "pipeline"
    root.mkdir()
    os.link(source, root / "transformer.bin")
    os.link(source, root / "shared.bin")
    (root / "config.json").write_bytes(b"{}")
    assert _logical_bytes(root) == len(b"0123456789") + len(b"{}")


def test_real_wan_ffn_resolver_requires_paired_activation_and_projections():
    parts = DiffusersWanAdapter.ffn_components(_Block())
    assert parts is not None
    assert parts[0].out_features == 8
    assert parts[2].in_features == 4
    assert DiffusersWanAdapter.ffn_components(type("Missing", (), {"up_proj": _Linear(), "down_proj": _Linear()})()) is None
