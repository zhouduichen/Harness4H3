from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from harness4h3.student.model import build_student
from harness4h3.student.quantization import quantize_checkpoint
from harness4h3.student.proposal import StudentProposal, StudentTarget
from harness4h3.student.runtime_quantization import QuantizedLinear, load_runtime_quantized_model


def test_quantized_linear_keeps_int8_storage_and_matches_reference():
    reference = torch.nn.Linear(4, 3, bias=True, dtype=torch.bfloat16)
    weight = torch.tensor(
        [[-3, -2, -1, 0], [1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8
    )
    scale = torch.tensor([0.125], dtype=torch.float32)
    layer = QuantizedLinear.from_linear(reference, weight, scale)
    value = torch.randn(2, 4, dtype=torch.bfloat16)

    expected = torch.nn.functional.linear(
        value,
        weight.to(dtype=value.dtype) * scale.to(dtype=value.dtype),
        reference.bias,
    )

    assert layer.weight_int8.dtype is torch.int8
    assert not any(name == "weight" for name, _ in layer.named_parameters())
    assert torch.allclose(layer(value), expected, atol=1e-3, rtol=1e-3)


def test_runtime_loader_rejects_a_non_int8_artifact(tmp_path: Path):
    checkpoint = tmp_path / "student.safetensors"
    save_file({"weight": torch.ones(1)}, str(checkpoint))

    with pytest.raises(ValueError, match="quantization=int8"):
        load_runtime_quantized_model(None, checkpoint, None, torch.device("cpu"))


def test_runtime_loader_keeps_student_linear_weights_quantized(tmp_path: Path):
    proposal = StudentProposal.from_dict(
        {
            "schema_version": 1,
            "proposal_id": "runtime-test",
            "parent_proposal_id": None,
            "teacher": {"checkpoint": "teacher.safetensors", "adapter": "minimax_h3"},
            "architecture": {
                "family": "video_latent_dit",
                "latent_channels": 24,
                "hidden_size": 32,
                "depth": 1,
                "num_heads": 8,
                "mlp_ratio": 2.0,
                "spatial_patch": 2,
                "temporal_patch": 1,
                "temporal_layers": [0],
                "conditioning": "ada_norm_zero",
                "norm": "rmsnorm",
                "activation": "silu",
            },
            "training": {
                "method": "dmd2",
                "source_steps": 2,
                "target_steps": 1,
                "learning_rate": 1e-4,
                "critic_learning_rate": 1e-4,
            },
            "deployment": {"precision": "bf16", "quantization": "int8"},
        }
    )
    target = StudentTarget(
        latent_height=4,
        latent_width=4,
        condition_dim=8,
        min_hidden_size=1,
        max_hidden_size=64,
        min_depth=1,
        max_depth=4,
    )
    reference = build_student(proposal, device=torch.device("cpu"), target=target)
    source = tmp_path / "student.safetensors"
    quantized = tmp_path / "student-int8.safetensors"
    save_file(
        {name: value.detach().contiguous() for name, value in reference.state_dict().items()},
        str(source),
    )
    quantize_checkpoint(source, quantized, bits=8, metadata={"target_device_id": "test"})

    loaded = load_runtime_quantized_model(proposal, quantized, target, torch.device("cpu"))
    quantized_linears = [module for module in loaded.modules() if isinstance(module, QuantizedLinear)]
    assert quantized_linears
    assert all(module.weight_int8.dtype is torch.int8 for module in quantized_linears)
    value = torch.randn(1, 24, 5, 4, 4, dtype=torch.bfloat16)
    conditioning = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    timestep = torch.zeros(1, dtype=torch.float32)
    with torch.inference_mode():
        output = loaded(value, conditioning, timestep)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
