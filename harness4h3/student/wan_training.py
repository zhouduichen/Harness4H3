"""Wan2.1-native training helpers for the real H3 distillation path.

The existing ``VideoLatentDiT`` Student is intentionally not used here.  Wan
keeps its own VAE/text encoder/Flow-Match geometry, while H3 is converted to a
Wan training target through an explicit RGB boundary.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from .wan import WanStudentError, _resolve_device, _torch_dtype


@dataclass(frozen=True)
class WanTrainingSpec:
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    pretrained_path: Optional[str] = None
    lineage_root: str = "W0000"
    width: int = 256
    height: int = 256
    num_frames: int = 81
    native_fps: float = 16.0
    dtype: str = "bf16"
    max_sequence_length: int = 512
    learning_rate: float = 1.0e-6
    standard_loss_weight: float = 1.0
    distill_loss_weight: float = 0.5
    train_steps: int = 1
    h3_sampling_steps: int = 4
    h3_latent_frames: int = 36
    h3_latent_height: int = 16
    h3_latent_width: int = 16
    h3_latent_channels: int = 24
    vae_name: str = "minimax_h3_video_vae_fp16.safetensors"

    def __post_init__(self) -> None:
        if self.model_id != "Wan-AI/Wan2.1-T2V-1.3B-Diffusers":
            raise ValueError("Wan training requires the official Wan2.1-T2V-1.3B model id")
        if self.lineage_root != "W0000":
            raise ValueError("Wan training must start at lineage root W0000")
        if self.width != 256 or self.height != 256:
            raise ValueError("Wan 5s training contract requires 256x256")
        if self.num_frames != 81:
            raise ValueError("Wan 5s training contract requires 81 frames")
        if float(self.native_fps) != 16.0:
            raise ValueError("Wan 5s training contract requires native 16 FPS")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        for name in (
            "max_sequence_length",
            "train_steps",
            "h3_sampling_steps",
            "h3_latent_frames",
            "h3_latent_height",
            "h3_latent_width",
            "h3_latent_channels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError("%s must be positive" % name)
        for name in ("learning_rate", "standard_loss_weight", "distill_loss_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        if not str(self.vae_name).strip() or "/" in self.vae_name or "\\" in self.vae_name:
            raise ValueError("vae_name must be a model filename")

    @property
    def duration_s(self) -> float:
        return float(self.num_frames) / float(self.native_fps)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({"pretrained": True, "duration_s": self.duration_s})
        return value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WanTrainingSpec":
        if not isinstance(raw, Mapping):
            raise ValueError("Wan training spec must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError("Wan training spec has unknown field(s): %s" % ", ".join(map(str, unknown)))
        values = asdict(cls())
        values.update(dict(raw))
        return cls(**values)


@dataclass
class WanTrainingComponents:
    pipeline: Any
    transformer: Any
    vae: Any
    text_encoder: Any
    tokenizer: Any
    scheduler: Any
    device: torch.device
    dtype: torch.dtype


def load_wan_training_components(spec: WanTrainingSpec, *, device: Optional[str] = None) -> WanTrainingComponents:
    """Load real Wan components and freeze non-Transformer modules."""

    try:
        diffusers = importlib.import_module("diffusers")
        pipeline_class = getattr(diffusers, "WanPipeline")
    except (ImportError, AttributeError) as exc:
        raise WanStudentError("diffusers_missing", "Wan training requires diffusers.WanPipeline") from exc
    try:
        torch_dtype = _torch_dtype(torch, spec)
        selected = _resolve_device(torch, device)
        source = str(spec.pretrained_path or spec.model_id)
        pipeline = pipeline_class.from_pretrained(source, torch_dtype=torch_dtype, local_files_only=True)
        pipeline = pipeline.to(selected)
    except Exception as exc:
        raise WanStudentError("pretrained_load_failed", str(exc)) from exc
    transformer = pipeline.transformer
    if transformer is None:
        raise WanStudentError("transformer_missing", "Wan checkpoint has no Transformer component")
    transformer.train()
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    for module in (pipeline.vae, pipeline.text_encoder):
        if module is not None:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return WanTrainingComponents(
        pipeline=pipeline,
        transformer=transformer,
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        scheduler=pipeline.scheduler,
        device=selected,
        dtype=torch_dtype,
    )


def _wan_latent_normalize(raw_latent: torch.Tensor, vae: Any) -> torch.Tensor:
    mean = torch.as_tensor(vae.config.latents_mean, device=raw_latent.device, dtype=raw_latent.dtype).view(
        1, -1, 1, 1, 1
    )
    std = torch.as_tensor(vae.config.latents_std, device=raw_latent.device, dtype=raw_latent.dtype).view(
        1, -1, 1, 1, 1
    )
    if raw_latent.shape[1] != mean.shape[1]:
        raise WanStudentError(
            "wan_latent_channels_mismatch",
            "Wan VAE returned %d channels but config has %d means" % (raw_latent.shape[1], mean.shape[1]),
        )
    return (raw_latent - mean) / std


def encode_wan_video(vae: Any, frames: torch.Tensor, spec: WanTrainingSpec, device: torch.device) -> torch.Tensor:
    """Encode RGB frames ``[T,H,W,C]`` or ``[B,C,T,H,W]`` into normalized Wan latents."""

    value = torch.as_tensor(frames, device=device)
    if value.ndim == 4 and value.shape[-1] in (3, 4):
        value = value[..., :3].permute(3, 0, 1, 2).unsqueeze(0)
    elif value.ndim == 5 and value.shape[1] in (3, 4):
        value = value[:, :3]
    else:
        raise WanStudentError("wan_video_shape_invalid", "expected RGB frames, got %s" % (tuple(value.shape),))
    if value.shape[2] != spec.num_frames or tuple(value.shape[-2:]) != (spec.height, spec.width):
        raise WanStudentError(
            "wan_video_shape_invalid",
            "expected [*,3,%d,%d,%d], got %s"
            % (spec.num_frames, spec.height, spec.width, tuple(value.shape)),
        )
    value = value.to(dtype=next(vae.parameters()).dtype)
    if float(value.min()) >= 0.0 and float(value.max()) <= 1.0:
        value = value * 2.0 - 1.0
    elif float(value.min()) >= 0.0 and float(value.max()) <= 255.0:
        value = value / 127.5 - 1.0
    if not torch.isfinite(value).all():
        raise WanStudentError("wan_video_nonfinite", "input RGB frames contain non-finite values")
    with torch.inference_mode():
        posterior = vae.encode(value, return_dict=True).latent_dist
        raw_latent = posterior.mode()
    return _wan_latent_normalize(raw_latent, vae).detach()


def resample_video_frames(frames: torch.Tensor, *, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Convert H3-decoded RGB frames to the Wan native temporal contract."""

    value = torch.as_tensor(frames).float()
    if value.ndim != 4 or value.shape[-1] not in (3, 4):
        raise WanStudentError("teacher_video_shape_invalid", "expected H3 RGB frames, got %s" % (tuple(value.shape),))
    value = value[..., :3].permute(3, 0, 1, 2).unsqueeze(0)
    value = F.interpolate(value, size=(int(num_frames), int(height), int(width)), mode="trilinear", align_corners=False)
    return value[0].permute(1, 2, 3, 0).contiguous()


def flow_match_batch(
    clean_latent: torch.Tensor,
    *,
    scheduler: Any,
    generator: torch.Generator,
    device: torch.device,
    sigma: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a Wan Flow-Match sample with ``v = noise - clean`` target."""

    clean = clean_latent.to(device=device)
    if noise is None:
        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
    else:
        noise = noise.to(device=device, dtype=clean.dtype)
    if sigma is None:
        sigma = torch.rand((clean.shape[0],), generator=generator, device=device, dtype=torch.float32)
    else:
        sigma = sigma.to(device=device, dtype=torch.float32).reshape(clean.shape[0])
    if torch.any(sigma <= 0.0) or torch.any(sigma >= 1.0):
        raise ValueError("Flow-Match sigma must be strictly between zero and one")
    if hasattr(scheduler, "config"):
        train_timesteps = float(getattr(scheduler.config, "num_train_timesteps", 1000))
    else:
        train_timesteps = 1000.0
    timestep = sigma * train_timesteps
    broadcast = sigma.to(dtype=clean.dtype).view(-1, 1, 1, 1, 1)
    noisy = (1.0 - broadcast) * clean + broadcast * noise
    target = noise - clean
    return noisy, timestep, target, sigma


def wan_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    sigma: torch.Tensor,
    *,
    standard_weight: float = 1.0,
    distill_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, standard velocity, and clean-anchor distillation losses."""

    standard = F.mse_loss(prediction.float(), target.float())
    sigma_value = sigma.to(device=noisy.device, dtype=noisy.dtype).view(-1, 1, 1, 1, 1)
    predicted_clean = noisy - sigma_value * prediction
    distill = F.mse_loss(predicted_clean.float(), clean.float())
    total = float(standard_weight) * standard + float(distill_weight) * distill
    return total, standard.detach(), distill.detach()


def sha256_path(path: Path) -> str:
    """Hash one file or a deterministic directory tree."""

    selected = Path(path)
    digest = hashlib.sha256()
    if selected.is_file():
        with selected.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not selected.is_dir():
        raise FileNotFoundError(selected)
    for child in sorted(item for item in selected.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(selected)).encode("utf-8"))
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def save_wan_checkpoint(transformer: Any, path: Path, metadata: Mapping[str, Any]) -> str:
    """Save only the trained Wan Transformer and return its SHA-256 digest."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise WanStudentError("safetensors_missing", "safetensors is required for Wan checkpoints") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().to(device="cpu").contiguous() for name, value in transformer.state_dict().items()}
    save_file(state, str(target), metadata={str(key): str(value) for key, value in metadata.items()})
    return sha256_path(target)


__all__ = [
    "WanTrainingComponents",
    "WanTrainingSpec",
    "WanStudentError",
    "encode_wan_video",
    "flow_match_batch",
    "load_wan_training_components",
    "resample_video_frames",
    "save_wan_checkpoint",
    "sha256_path",
    "wan_velocity_loss",
]
