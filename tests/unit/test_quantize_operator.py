from __future__ import annotations

import json
import struct

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.controller.schemas import CostEstimate
from harness4h3.h3.inspector import H3Inspector
from harness4h3.h3.state import ModelState
from harness4h3.operators.base import ExecutionContext
from harness4h3.operators.quantize import PrebuiltQuantizeOperator
from harness4h3.target.profile import TargetProfile



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


H3_TENSORS = {
    "audio_patch_proj.weight": [2, 2],
    "video_patch_proj.weight": [2, 2],
    "condition_proj.weight": [2, 2],
    "blocks.0.attn.qkv_proj.weight": [6, 2],
    "blocks.1.attn.qkv_proj.weight": [6, 2],
    "token_refiner.blocks.0.attn.qkv_proj.weight": [6, 2],
    "final_layer.audio_out.weight": [2, 2],
    "final_layer.video_out.weight": [2, 2],
}


def test_prebuilt_quantize_produces_real_h3_child_without_overwriting_parent(tmp_path):
    parent_path = tmp_path / "baseline.safetensors"
    variant_path = tmp_path / "minimax_h3_nvfp4.safetensors"
    write_safetensors(parent_path, H3_TENSORS)
    write_safetensors(variant_path, H3_TENSORS)
    baseline = ModelState.fake_baseline()
    baseline = ModelState.from_dict({**baseline.to_dict(), "checkpoint_path": str(parent_path)})
    parent = ModelCandidate("M0000", None, 0, str(parent_path), baseline, None, "baseline")
    target = TargetProfile("target", "gpu", "local")
    operator = PrebuiltQuantizeOperator({4: variant_path}, H3Inspector(), CostEstimate(wall_time_s=0.2))

    result = operator.execute(parent, {"bits": 4}, ExecutionContext(tmp_path / "exp", "M0001"))

    assert result.ok
    assert result.output_state is not None
    assert result.output_state.model_id == "M0001"
    assert result.output_state.parent_model_id == "M0000"
    assert result.output_state.quantization["bits"] == 4
    assert result.output_state.provenance["prebuilt_variant"] is True
    assert result.output_state.runtime_state["metrics_stale"] is True
    assert result.output_state.checkpoint_path == str(variant_path.resolve())
    assert parent_path.read_bytes() == variant_path.read_bytes()
    assert operator.dry_run(baseline, {"bits": 4}, target) == CostEstimate(wall_time_s=0.2)


def test_prebuilt_quantize_rejects_missing_variant(tmp_path):
    state = ModelState.fake_baseline()
    parent = ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")
    operator = PrebuiltQuantizeOperator({4: tmp_path / "missing.safetensors"}, H3Inspector())
    result = operator.execute(parent, {"bits": 4}, ExecutionContext(tmp_path / "exp", "M0001"))
    assert not result.ok
    assert result.failure_type == "quantize_invalid"
