from __future__ import annotations

import json

from research.experiments.m6_runtime_memory import _effective_operator_args
from research.experiments.m6_runtime_recipe import _load_runtime_gene


def test_controller_runtime_args_are_scoped_to_selected_operator():
    args = _effective_operator_args(
        "vae_tiling",
        "vae_tiling",
        {"tile_size": 128, "overlap": 32, "chunk_size": 4096},
    )
    assert args == {"tile_size": 128, "overlap": 32}


def test_runtime_recipe_defaults_are_safe_for_new_operators():
    assert _effective_operator_args("cache_release", "cache_release", {}) == {"stage": "before_decode"}
    assert _effective_operator_args("vae_decode_offload", "vae_decode_offload", {}) == {"mode": "cpu"}


def test_rejected_runtime_gene_is_loaded_as_read_only_context(tmp_path):
    gene_path = tmp_path / "research" / "evidence" / "design-genes" / "design-gene-m6-vae-tiling.json"
    gene_path.parent.mkdir(parents=True)
    gene_path.write_text(json.dumps({"gene_id": "H3-M6-VAE-Tiling-001", "status": "rejected"}))
    assert _load_runtime_gene(tmp_path)["status"] == "rejected"
