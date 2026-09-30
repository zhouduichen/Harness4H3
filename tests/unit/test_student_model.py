from __future__ import annotations

import torch
from torch import nn

from harness4h3.student.model import ConditionProjector, SemanticResidualBridge, SpatialDiTBlock, build_smoke_student
from harness4h3.student.proposal import ArchitectureSpec, StudentProposal
from harness4h3.student.inference import student_video_sigmas
from tests.unit.test_student_proposal import valid_payload


def test_condition_projector_ignores_h3_bos_token():
    projector = ConditionProjector(2, 1)
    with torch.no_grad():
        projector.projection.weight.zero_()
        projector.projection.bias.zero_()
        projector.projection.weight[0, 0] = 1.0
    value = torch.tensor([[[100.0, 100.0], [1.0, 2.0], [3.0, 4.0]]])
    output = projector(value)
    assert torch.equal(output[0, 0], torch.tensor(2.0))


def test_semantic_residual_bridge_is_zero_initialized():
    bridge = SemanticResidualBridge(4, 8, bottleneck=4)
    value = torch.randn(2, 5, 4)
    output = bridge(value)
    assert torch.count_nonzero(bridge.output.weight) == 0
    assert torch.count_nonzero(bridge.output.bias) == 0
    assert torch.count_nonzero(output) == 0


def test_cross_attention_is_disabled_for_inherited_checkpoint():
    architecture = ArchitectureSpec(
        family="video_latent_dit",
        latent_channels=2,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        spatial_patch=1,
        temporal_patch=1,
        temporal_layers=(0,),
        conditioning="ada_norm_zero",
        norm="rmsnorm",
        activation="silu",
    )
    block = SpatialDiTBlock(architecture, condition_dim=4)
    cross_query = nn.Linear(8, 2)
    cross_output = nn.Linear(2, 8)
    nn.init.zeros_(cross_output.weight)
    nn.init.zeros_(cross_output.bias)
    value = torch.randn(1, 3, 8)
    conditioning = torch.randn(1, 48)
    first_tokens = torch.randn(1, 5, 4)
    second_tokens = torch.randn(1, 5, 4)
    first = block(value, conditioning, first_tokens, cross_query, cross_output, 1)
    second = block(value, conditioning, second_tokens, cross_query, cross_output, 1)
    assert torch.equal(first, second)


def test_cross_output_deltas_are_zero_initialized_per_dit_block():
    proposal = StudentProposal.from_dict(valid_payload())
    student = build_smoke_student(proposal, scale=0.125)

    assert len(student.cross_output_deltas) == student.architecture.depth
    assert all(torch.count_nonzero(delta.weight) == 0 for delta in student.cross_output_deltas)
    assert all(torch.count_nonzero(delta.bias) == 0 for delta in student.cross_output_deltas)


def test_cross_attention_keeps_256_channels_for_semantic_capacity():
    proposal = StudentProposal.from_dict(valid_payload())
    student = build_smoke_student(proposal, scale=0.125)

    assert student.cross_attention_dim == 256
    assert student.semantic_bridge.bottleneck.out_features == 256


def test_student_inference_uses_h3_shifted_video_schedule():
    sigmas = student_video_sigmas(8)
    assert len(sigmas) == 9
    assert sigmas[0] == 1.0
    assert sigmas[-1] == 0.0
    assert sigmas[1] < 1.0
    assert sigmas[1] > 0.875
    assert sigmas[4] > 0.5
