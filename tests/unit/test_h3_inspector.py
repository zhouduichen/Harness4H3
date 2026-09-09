from __future__ import annotations

import json
import struct

import pytest

from harness4h3.h3.checkpoint import CheckpointInspectionError
from harness4h3.h3.inspector import H3Inspector


def write_safetensors(path, tensors):
    header = {}
    offset = 0
    for name, shape in tensors.items():
        size = 2
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def test_inspects_real_h3_safetensors_header_without_loading_weights(tmp_path):
    checkpoint = tmp_path / "minimax_h3_tiny_nvfp4.safetensors"
    write_safetensors(
        checkpoint,
        {
            "audio_patch_proj.weight": [2, 2],
            "video_patch_proj.weight": [2, 2],
            "condition_proj.weight": [2, 2],
            "blocks.0.attn.qkv_proj.weight": [6, 2],
            "blocks.0.ffn.up_proj.weight": [8, 2],
            "blocks.1.attn.qkv_proj.weight": [6, 2],
            "token_refiner.blocks.0.attn.qkv_proj.weight": [6, 2],
            "final_layer.audio_out.weight": [2, 2],
            "final_layer.video_out.weight": [2, 2],
        },
    )

    state = H3Inspector().inspect(checkpoint, sampling_steps=20, include_file_sha256=True)

    assert state.architecture_name == "MiniMax-H3"
    assert state.parameter_count == 72
    assert state.num_blocks == 2
    assert state.hidden_size == 2
    assert state.ffn_width == 8
    assert state.dtype == "float16"
    assert state.quantization == {"bits": 4, "scheme": "nvfp4", "source": "filename"}
    assert state.sampling_steps == 20
    assert state.provenance["tensor_count"] == 9
    assert state.provenance["weights_loaded"] is False
    assert len(state.provenance["file_sha256"]) == 64
    assert state.warnings == []


def test_unknown_checkpoint_is_not_mislabeled_h3(tmp_path):
    checkpoint = tmp_path / "other.safetensors"
    write_safetensors(checkpoint, {"linear.weight": [3, 2]})
    state = H3Inspector().inspect(checkpoint)
    assert state.architecture_name == "unknown"
    assert state.warnings


def test_rejects_truncated_or_unsupported_checkpoints(tmp_path):
    truncated = tmp_path / "bad.safetensors"
    truncated.write_bytes(b"bad")
    with pytest.raises(CheckpointInspectionError, match="truncated"):
        H3Inspector().inspect(truncated)
    unsupported = tmp_path / "model.bin"
    unsupported.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointInspectionError, match="unsupported"):
        H3Inspector().inspect(unsupported)
