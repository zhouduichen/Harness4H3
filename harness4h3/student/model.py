"""Registered video latent DiT modules used by the Student compiler."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .proposal import ArchitectureSpec, StudentProposal, StudentTarget


def _activation(name: str, value: Tensor) -> Tensor:
    if name == "silu":
        return F.silu(value)
    if name == "gelu":
        return F.gelu(value)
    raise ValueError("unsupported activation: %s" % name)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, *, device: Optional[torch.device] = None, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = float(eps)

    def forward(self, value: Tensor) -> Tensor:
        variance = value.pow(2).mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(variance + self.eps) * self.weight


class AdaNormZero(nn.Module):
    """Apply one of three static AdaLN-Zero modulation slots."""

    def __init__(self, hidden_size: int, slot: int):
        super().__init__()
        if slot not in (0, 1):
            raise ValueError("AdaNormZero slot must be 0 or 1")
        self.hidden_size = int(hidden_size)
        self.slot = int(slot)

    def forward(self, value: Tensor, conditioning: Tensor) -> tuple[Tensor, Tensor]:
        pieces = conditioning.view(-1, 6, self.hidden_size)
        offset = self.slot * 3
        shift = pieces[:, offset].unsqueeze(1)
        scale = pieces[:, offset + 1].unsqueeze(1)
        gate = pieces[:, offset + 2].unsqueeze(1)
        return value * (1.0 + scale) + shift, gate


class VideoPatchEmbed(nn.Module):
    def __init__(self, architecture: ArchitectureSpec, *, device: Optional[torch.device] = None):
        super().__init__()
        patch = architecture.spatial_patch
        self.projection = nn.Conv3d(
            architecture.latent_channels,
            architecture.hidden_size,
            kernel_size=(architecture.temporal_patch, patch, patch),
            stride=(architecture.temporal_patch, patch, patch),
            device=device,
        )

    def forward(self, value: Tensor) -> Tensor:
        projected = self.projection(value)
        batch, hidden, frames, height, width = projected.shape
        return projected.flatten(2).transpose(1, 2).reshape(batch, frames * height * width, hidden)


class ConditionProjector(nn.Module):
    def __init__(self, condition_dim: int, hidden_size: int, *, device: Optional[torch.device] = None):
        super().__init__()
        self.projection = nn.Linear(condition_dim, 6 * hidden_size, device=device)

    def forward(self, value: Tensor) -> Tensor:
        return self.projection(value.mean(dim=1))


class TimestepProjector(nn.Module):
    def __init__(self, hidden_size: int, *, device: Optional[torch.device] = None):
        super().__init__()
        self.projection = nn.Linear(1, 6 * hidden_size, device=device)

    def forward(self, value: Tensor) -> Tensor:
        return self.projection(value.reshape(-1, 1))


class _DiTBlock(nn.Module):
    def __init__(self, architecture: ArchitectureSpec, *, device: Optional[torch.device] = None):
        super().__init__()
        hidden = architecture.hidden_size
        expansion = int(round(hidden * architecture.mlp_ratio))
        if expansion <= 0:
            raise ValueError("MLP expansion must be positive")
        self.hidden_size = hidden
        self.num_heads = architecture.num_heads
        self.activation_name = architecture.activation
        if hidden % architecture.num_heads:
            raise ValueError("hidden size must be divisible by num_heads")
        self.norm1 = RMSNorm(hidden, device=device)
        self.norm2 = RMSNorm(hidden, device=device)
        self.modulation1 = AdaNormZero(hidden, 0)
        self.modulation2 = AdaNormZero(hidden, 1)
        self.qkv = nn.Linear(hidden, 3 * hidden, device=device)
        self.attention_output = nn.Linear(hidden, hidden, device=device)
        self.mlp_in = nn.Linear(hidden, expansion, device=device)
        self.mlp_out = nn.Linear(expansion, hidden, device=device)

    def _attention(self, value: Tensor) -> Tensor:
        batch, tokens, hidden = value.shape
        head_dim = hidden // self.num_heads
        qkv = self.qkv(value).reshape(batch, tokens, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        attended = attended.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.attention_output(attended)

    def forward(self, value: Tensor, conditioning: Tensor) -> Tensor:
        normalized, gate = self.modulation1(self.norm1(value), conditioning)
        value = value + gate * self._attention(normalized)
        normalized, gate = self.modulation2(self.norm2(value), conditioning)
        feed_forward = self.mlp_out(_activation(self.activation_name, self.mlp_in(normalized)))
        return value + gate * feed_forward


class SpatialDiTBlock(_DiTBlock):
    pass


class TemporalDiTBlock(_DiTBlock):
    pass


class VideoUnpatchify(nn.Module):
    def __init__(self, architecture: ArchitectureSpec, target: StudentTarget, *, device: Optional[torch.device] = None):
        super().__init__()
        self.architecture = architecture
        self.target = target
        patch = architecture.spatial_patch
        self.patch_volume = architecture.latent_channels * architecture.temporal_patch * patch * patch
        self.projection = nn.Conv3d(architecture.hidden_size, self.patch_volume, kernel_size=1, device=device)

    def forward(self, tokens: Tensor) -> Tensor:
        architecture = self.architecture
        target = self.target
        batch = tokens.shape[0]
        reduced_frames = target.latent_frames // architecture.temporal_patch
        reduced_height = target.latent_height // architecture.spatial_patch
        reduced_width = target.latent_width // architecture.spatial_patch
        hidden = architecture.hidden_size
        features = tokens.transpose(1, 2).reshape(batch, hidden, reduced_frames, reduced_height, reduced_width)
        patches = self.projection(features)
        patches = patches.reshape(
            batch,
            architecture.latent_channels,
            architecture.temporal_patch,
            architecture.spatial_patch,
            architecture.spatial_patch,
            reduced_frames,
            reduced_height,
            reduced_width,
        )
        patches = patches.permute(0, 1, 2, 5, 3, 6, 4, 7)
        return patches.reshape(
            batch,
            architecture.latent_channels,
            target.latent_frames,
            target.latent_height,
            target.latent_width,
        )


class VideoLatentDiT(nn.Module):
    """A fixed executable graph whose dimensions come from the proposal."""

    def __init__(
        self,
        proposal: StudentProposal,
        target: StudentTarget = StudentTarget(),
        *,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        architecture = proposal.architecture
        self.proposal_digest = proposal.digest
        self.target = target
        self.architecture = architecture
        self.patch_embed = VideoPatchEmbed(architecture, device=device)
        self.condition = ConditionProjector(target.condition_dim, architecture.hidden_size, device=device)
        self.timestep = TimestepProjector(architecture.hidden_size, device=device)
        temporal = set(architecture.temporal_layers)
        blocks = []
        for index in range(architecture.depth):
            block_type = TemporalDiTBlock if index in temporal else SpatialDiTBlock
            blocks.append(block_type(architecture, device=device))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(architecture.hidden_size, device=device)
        self.unpatchify = VideoUnpatchify(architecture, target, device=device)

    def forward(self, video_latents: Tensor, conditioning: Tensor, timestep: Tensor) -> Tensor:
        value = self.patch_embed(video_latents)
        modulation = self.condition(conditioning) + self.timestep(timestep)
        for block in self.blocks:
            value = block(value, modulation)
        value = self.final_norm(value)
        return self.unpatchify(value)


def build_student(
    proposal: StudentProposal,
    device: Union[str, torch.device] = "meta",
    target: StudentTarget = StudentTarget(),
) -> VideoLatentDiT:
    """Build the registered graph; ``meta`` is the safe production compiler mode."""

    return VideoLatentDiT(proposal, target, device=torch.device(device))


def build_smoke_student(
    proposal: StudentProposal,
    scale: float = 0.125,
    target: StudentTarget = StudentTarget(),
) -> VideoLatentDiT:
    """Build a tiny same-topology graph for contract-only CPU tests."""

    if not 0 < float(scale) <= 1:
        raise ValueError("smoke scale must be in (0, 1]")
    architecture = proposal.architecture
    hidden = max(32, int(round(architecture.hidden_size * scale)))
    heads = min(architecture.num_heads, hidden)
    while hidden % heads:
        heads -= 1
    depth = max(1, int(round(architecture.depth * scale)))
    temporal_layers = tuple(index for index in architecture.temporal_layers if index < depth)
    scaled_architecture = ArchitectureSpec(
        family=architecture.family,
        latent_channels=architecture.latent_channels,
        hidden_size=hidden,
        depth=depth,
        num_heads=max(1, heads),
        mlp_ratio=architecture.mlp_ratio,
        spatial_patch=architecture.spatial_patch,
        temporal_patch=architecture.temporal_patch,
        temporal_layers=temporal_layers,
        conditioning=architecture.conditioning,
        norm=architecture.norm,
        activation=architecture.activation,
    )
    scaled = StudentProposal(
        schema_version=proposal.schema_version,
        proposal_id=proposal.proposal_id,
        parent_proposal_id=proposal.parent_proposal_id,
        teacher=proposal.teacher,
        architecture=scaled_architecture,
        training=proposal.training,
        deployment=proposal.deployment,
    )
    return VideoLatentDiT(scaled, target, device=torch.device("cpu"))


__all__ = [
    "AdaNormZero",
    "ConditionProjector",
    "RMSNorm",
    "SpatialDiTBlock",
    "TemporalDiTBlock",
    "VideoLatentDiT",
    "VideoPatchEmbed",
    "VideoUnpatchify",
    "build_smoke_student",
    "build_student",
]
