from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from tools import h3_real_quantize_worker as worker


def _checkpoint(path, *, quantized: bool) -> None:
    config = {"transformer": {"hidden_size": 4, "num_layers": 2, "num_attention_heads": 2, "ffn_hidden_size": 8}}
    tensors = {
        "blocks.0.weight": torch.ones(2, 2),
        "blocks.1.weight": torch.ones(2, 2),
        "final_layer.video_out.weight": torch.ones(2, 2),
    }
    if quantized:
        tensors["blocks.0.weight"] = torch.ones(2, 2, dtype=torch.int8)
        tensors["blocks.0.weight_scale"] = torch.ones(2, 1)
        tensors["blocks.0.comfy_quant"] = torch.ones(1, dtype=torch.uint8)
    save_file(tensors, str(path), metadata={"config": json.dumps(config, separators=(",", ":"))})


def _request(tmp_path, parent, source, bits=8):
    return {
        "operator": "quantize",
        "operator_args": {"bits": bits},
        "parent": {"model_id": "M0000", "checkpoint_path": str(parent)},
        "child_model_id": "M0001",
        "artifacts_dir": str(tmp_path / "artifacts"),
    }, {
        "comfyui_root": str(tmp_path),
        "quantized_variants": {str(bits): str(source)},
    }


def test_prebuilt_quantize_copies_matching_real_variant_and_proves_parent_immutable(tmp_path):
    parent = tmp_path / "parent.safetensors"
    source = tmp_path / "int8.safetensors"
    _checkpoint(parent, quantized=False)
    _checkpoint(source, quantized=True)
    request, config = _request(tmp_path, parent, source)
    result_path = tmp_path / "result.json"

    assert worker._run(request, config, result_path) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    state = result["output_state"]
    assert state["quantization"]["bits"] == 8
    assert state["provenance"]["offline_simulation"] is False
    assert result["metrics"]["child_copy_verified"] is True
    assert result["metrics"]["benchmark_reload_pending"] is True
    assert (tmp_path / "artifacts" / "M0001.safetensors").is_file()
    assert json.loads((tmp_path / "artifacts" / "M0001.safetensors.evidence.json").read_text())["source_is_quantized"] is True


def test_prebuilt_quantize_rejects_nonmatching_architecture(tmp_path):
    parent = tmp_path / "parent.safetensors"
    source = tmp_path / "int8.safetensors"
    _checkpoint(parent, quantized=False)
    _checkpoint(source, quantized=True)
    config = {"quantized_variants": {"8": str(source)}}
    request, _ = _request(tmp_path, parent, source)
    source_config = json.loads(json.dumps({"transformer": {"hidden_size": 8, "num_layers": 2}}))
    # Re-write the source header with a deliberately incompatible config.
    tensors = {"blocks.0.weight": torch.ones(2, 2, dtype=torch.int8), "blocks.0.weight_scale": torch.ones(2, 1), "blocks.0.comfy_quant": torch.ones(1, dtype=torch.uint8)}
    save_file(tensors, str(source), metadata={"config": json.dumps(source_config, separators=(",", ":"))})
    result_path = tmp_path / "result.json"

    assert worker._run(request, config, result_path) == 1
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "architecture" in result["message"]
