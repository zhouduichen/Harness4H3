"""Tiny joint video/audio rectified-flow network."""

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from h3_training.data.schema import Conditioning, ModalLatents, ModalPrediction, ModalTimesteps


@dataclass(frozen=True)
class TinyH3Config:
    latent_dim: int = 8
    condition_dim: int = 8
    hidden_size: int = 32
    num_layers: int = 2
    num_heads: int = 4
    ffn_width: int = 64
    video_tokens: int = 4
    audio_tokens: int = 3

    def to_dict(self):
        return asdict(self)


class TinyH3Model(nn.Module):
    """Shared conditioned Transformer with modality-specific heads."""

    def __init__(self, config: TinyH3Config = TinyH3Config()) -> None:
        super().__init__()
        self.config = config
        self.video_in = nn.Linear(config.latent_dim, config.hidden_size)
        self.audio_in = nn.Linear(config.latent_dim, config.hidden_size)
        self.condition_in = nn.Linear(config.condition_dim, config.hidden_size)
        self.time_in = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.modality_embedding = nn.Parameter(torch.zeros(2, config.hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.video_head = nn.Linear(config.hidden_size, config.latent_dim)
        self.audio_head = nn.Linear(config.hidden_size, config.latent_dim)

    def _tokens(
        self,
        value: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
        projection: nn.Linear,
        modality: int,
    ) -> Tensor:
        if timestep.ndim == 0:
            timestep = timestep.expand(value.shape[0])
        time = self.time_in(timestep.reshape(value.shape[0], 1).to(value.dtype)).unsqueeze(1)
        condition = self.condition_in(conditioning).unsqueeze(1)
        return projection(value) + time + condition + self.modality_embedding[modality]

    def forward(
        self,
        latents: ModalLatents,
        conditioning: Conditioning,
        timesteps: ModalTimesteps,
    ) -> ModalPrediction:
        pieces = []
        video_length = 0
        if latents.video is not None:
            if timesteps.video is None:
                raise ValueError("video timestep is required")
            video_length = latents.video.shape[1]
            pieces.append(self._tokens(latents.video, timesteps.video, conditioning.text, self.video_in, 0))
        if latents.audio is not None:
            if timesteps.audio is None:
                raise ValueError("audio timestep is required")
            pieces.append(self._tokens(latents.audio, timesteps.audio, conditioning.text, self.audio_in, 1))
        hidden = self.final_norm(self.backbone(torch.cat(pieces, dim=1)))
        video = self.video_head(hidden[:, :video_length]) if latents.video is not None else None
        audio = self.audio_head(hidden[:, video_length:]) if latents.audio is not None else None
        return ModalPrediction(video=video, audio=audio)
