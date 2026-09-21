import json

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tools import h3_real_prune_worker as worker


def _checkpoint(path, layers=4):
    state = {}
    for index in range(layers):
        value = float(index + 1)
        for suffix in (
            "adaln_proj.linear.bias",
            "norm1.weight",
            "norm2.weight",
            "attn.q_norm.weight",
            "attn.k_norm.weight",
        ):
            state["blocks.%d.%s" % (index, suffix)] = torch.full((2,), value, dtype=torch.float32)
    metadata = {
        "config": json.dumps(
            {"transformer": {"image_model": "minimax_h3", "hidden_size": 2, "num_layers": layers}},
            separators=(",", ":"),
        )
    }
    save_file(state, str(path), metadata=metadata)


def test_structured_prune_rewrites_layers_and_proves_child_reload(tmp_path, monkeypatch):
    parent = tmp_path / "parent.safetensors"
    _checkpoint(parent)
    monkeypatch.setattr(worker, "_load_api", lambda root: {"fake": True})
    monkeypatch.setattr(worker, "_reload", lambda api, path: len(load_file(str(path), device="cpu")))

    request = {
        "operator": "prune_blocks",
        "child_model_id": "M0001",
        "operator_args": {"ratio": 0.25},
        "parent": {"id": "M0000", "checkpoint_path": str(parent), "state": {"sampling_steps": 16}},
        "artifacts_dir": str(tmp_path / "artifacts"),
    }
    config = {"comfyui_root": str(tmp_path)}
    result_path = tmp_path / "result.json"

    assert worker._run(request, config, result_path) == 0
    result = json.loads(result_path.read_text())
    assert result["status"] == "success"
    state = result["output_state"]
    assert state["num_blocks"] == 3
    assert state["sampling_steps"] == 16
    assert result["metrics"]["structural_change"] is True
    assert result["metrics"]["child_reloaded"] is True
    assert result["metrics"]["removed_parameter_count"] > 0

    child = tmp_path / "artifacts" / "M0001.safetensors"
    with safe_open(str(child), framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["config"])
        assert metadata["transformer"]["num_layers"] == 3
        assert "blocks.0.norm1.weight" in handle.keys()
        assert "blocks.3.norm1.weight" not in handle.keys()
