"""Wan-native overlapping-latent continuation for approximately 30s videos.

Wan2.1's temporal VAE maps an 81-frame window to 21 latent frames.  This
module keeps the Wan pipeline and latent geometry intact, carries the clean
latent tail into the next window during every denoising step, and only then
crossfades the decoded overlap.  It is intentionally separate from the H3
Student continuation code because the two VAEs have different temporal
geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .wan import WanStudentError, WanStudentSpec, _as_frame_array, _resolve_device, _torch_dtype
from .wan_training import sha256_path


class WanLongVideoError(WanStudentError):
    """Raised when Wan continuation geometry or evidence is invalid."""


@dataclass(frozen=True)
class WanLongVideoSpec:
    """A reproducible Wan continuation contract.

    The defaults are chosen for the requested 30-second target: eight 81-
    frame windows with seven latent-frame overlap produce 473 native frames,
    which are resampled to exactly 720 frames at 24 FPS.
    """

    window_frames: int = 81
    temporal_scale_factor: int = 4
    overlap_latent_frames: int = 7
    window_count: int = 8
    native_fps: float = 16.0
    output_fps: float = 24.0
    target_frames: int = 720
    width: int = 256
    height: int = 256
    inference_steps: int = 30
    guidance_scale: float = 5.0

    def __post_init__(self) -> None:
        for name in ("window_frames", "temporal_scale_factor", "overlap_latent_frames", "window_count", "target_frames", "width", "height", "inference_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError("%s must be positive" % name)
        if (self.window_frames - 1) % self.temporal_scale_factor:
            raise ValueError("window_frames must match the Wan temporal scale")
        latent_frames = self.latent_window_frames
        if self.window_count <= 1:
            raise ValueError("Wan long video requires multiple windows")
        if not 0 < self.overlap_latent_frames < latent_frames:
            raise ValueError("overlap_latent_frames must be between zero and the latent window length")
        for name in ("native_fps", "output_fps", "guidance_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        object.__setattr__(self, "window_frames", int(self.window_frames))
        object.__setattr__(self, "temporal_scale_factor", int(self.temporal_scale_factor))
        object.__setattr__(self, "overlap_latent_frames", int(self.overlap_latent_frames))
        object.__setattr__(self, "window_count", int(self.window_count))
        object.__setattr__(self, "target_frames", int(self.target_frames))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "inference_steps", int(self.inference_steps))

    @property
    def latent_window_frames(self) -> int:
        return (int(self.window_frames) - 1) // int(self.temporal_scale_factor) + 1

    @property
    def decoded_overlap_frames(self) -> int:
        return 1 + (int(self.overlap_latent_frames) - 1) * int(self.temporal_scale_factor)

    @property
    def window_stride_latent_frames(self) -> int:
        return self.latent_window_frames - self.overlap_latent_frames

    @property
    def composed_native_frames(self) -> int:
        return self.window_frames + (self.window_count - 1) * (self.window_frames - self.decoded_overlap_frames)

    @property
    def composed_native_duration_s(self) -> float:
        return float(self.composed_native_frames) / float(self.native_fps)

    @property
    def output_duration_s(self) -> float:
        return float(self.target_frames) / float(self.output_fps)

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "latent_window_frames": self.latent_window_frames,
                "decoded_overlap_frames": self.decoded_overlap_frames,
                "window_stride_latent_frames": self.window_stride_latent_frames,
                "composed_native_frames": self.composed_native_frames,
                "composed_native_duration_s": self.composed_native_duration_s,
                "output_duration_s": self.output_duration_s,
            }
        )
        return value


def continuation_seed(seed: int, window_index: int) -> int:
    """Derive stable, distinct per-window seeds without Python hash randomization."""

    if int(window_index) < 0:
        raise ValueError("window_index must be non-negative")
    return int((int(seed) + 0x9E3779B97F4A7C15 * (int(window_index) + 1)) & ((1 << 63) - 1))


def anchor_wan_latent_prefix(
    latent: torch.Tensor,
    previous_tail: Optional[torch.Tensor],
    overlap_latent_frames: int,
    weight: float,
) -> torch.Tensor:
    """Move a new Wan latent prefix toward the previous clean latent tail."""

    overlap = int(overlap_latent_frames)
    if previous_tail is None or overlap == 0:
        return latent
    if latent.ndim != 5 or previous_tail.ndim != 5:
        raise WanLongVideoError("latent_shape_invalid", "Wan continuation latents must be five-dimensional")
    if tuple(latent.shape[:2]) != tuple(previous_tail.shape[:2]) or tuple(latent.shape[-2:]) != tuple(previous_tail.shape[-2:]):
        raise WanLongVideoError("latent_shape_invalid", "Wan continuation latent shape mismatch")
    if latent.shape[-3] < overlap or previous_tail.shape[-3] < overlap:
        raise WanLongVideoError("latent_overlap_invalid", "Wan latent overlap exceeds the temporal shape")
    result = latent.clone()
    prefix = previous_tail[..., -overlap:, :, :].to(device=latent.device, dtype=latent.dtype)
    result[..., :overlap, :, :] = torch.lerp(result[..., :overlap, :, :], prefix, max(0.0, min(1.0, float(weight))))
    return result


def compose_wan_decoded_windows(windows: Sequence[np.ndarray], overlap_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Crossfade decoded overlap and append only each window's new suffix."""

    if not windows:
        raise WanLongVideoError("windows_empty", "at least one Wan decoded window is required")
    overlap = int(overlap_frames)
    if overlap <= 0:
        raise WanLongVideoError("decoded_overlap_invalid", "decoded overlap must be positive")
    output = _as_frame_array(windows[0]).astype(np.float32)
    seam_diffs: list[float] = []
    for current_raw in windows[1:]:
        current = _as_frame_array(current_raw).astype(np.float32)
        if tuple(current.shape[1:]) != tuple(output.shape[1:]):
            raise WanLongVideoError("decoded_shape_invalid", "Wan decoded windows have different shapes")
        if overlap >= output.shape[0] or overlap >= current.shape[0]:
            raise WanLongVideoError("decoded_overlap_invalid", "decoded overlap must be smaller than every window")
        weights = np.linspace(0.0, 1.0, overlap, dtype=np.float32).reshape(overlap, 1, 1, 1)
        blended = output[-overlap:] * (1.0 - weights) + current[:overlap] * weights
        boundary = output[-overlap - 1] if output.shape[0] > overlap else output[-overlap]
        seam_diffs.append(float(np.mean(np.abs(blended[0] - boundary))))
        output = np.concatenate((output[:-overlap], blended, current[overlap:]), axis=0)
    report = {
        "window_count": len(windows),
        "decoded_overlap_frames": overlap,
        "seam_count": len(seam_diffs),
        "seam_mean_abs_diff": float(np.mean(seam_diffs)) if seam_diffs else 0.0,
        "seam_max_abs_diff": float(max(seam_diffs)) if seam_diffs else 0.0,
        "frames": int(output.shape[0]),
    }
    return np.clip(np.rint(output), 0.0, 255.0).astype(np.uint8), report


def resample_wan_frames(frames: np.ndarray, *, target_frames: int, height: int, width: int) -> np.ndarray:
    """Resample a composed native-FPS RGB sequence to the output contract."""

    value = torch.from_numpy(_as_frame_array(frames)).float().permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    value = F.interpolate(value, size=(int(target_frames), int(height), int(width)), mode="trilinear", align_corners=False)
    return np.clip(np.rint(value[0].permute(1, 2, 3, 0).numpy() * 255.0), 0.0, 255.0).astype(np.uint8)


def load_wan_transformer_checkpoint(pipeline: Any, checkpoint: Path) -> str:
    """Load a saved Wan Transformer child checkpoint strictly."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise WanLongVideoError("safetensors_missing", "safetensors is required for Wan checkpoint restore") from exc
    selected = Path(checkpoint).resolve()
    if not selected.is_file():
        raise WanLongVideoError("checkpoint_missing", "Wan Student checkpoint does not exist: %s" % selected)
    state = load_file(str(selected), device="cpu")
    try:
        pipeline.transformer.load_state_dict(state, strict=True)
    except Exception as exc:
        raise WanLongVideoError("checkpoint_restore_failed", str(exc)) from exc
    return sha256_path(selected)


def _decode_wan_latent(pipeline: Any, latent: torch.Tensor) -> np.ndarray:
    vae = pipeline.vae
    vae_dtype = next(vae.parameters()).dtype
    value = latent.to(device=latent.device, dtype=vae_dtype)
    mean = torch.as_tensor(vae.config.latents_mean, device=value.device, dtype=vae_dtype).view(1, -1, 1, 1, 1)
    inverse_std = (1.0 / torch.as_tensor(vae.config.latents_std, device=value.device, dtype=vae_dtype)).view(1, -1, 1, 1, 1)
    value = value / inverse_std + mean
    with torch.no_grad():
        decoded = vae.decode(value, return_dict=False)[0]
    processed = pipeline.video_processor.postprocess_video(decoded, output_type="np")
    return _as_frame_array(processed)


def generate_wan_long_video(
    pipeline: Any,
    spec: WanStudentSpec,
    contract: WanLongVideoSpec,
    prompt: str,
    *,
    seed: int,
    device: Any = None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Generate one continuous Wan sequence using overlapping latent windows."""

    if not str(prompt).strip():
        raise WanLongVideoError("prompt_empty", "prompt must not be empty")
    selected = device or _resolve_device(torch, None)
    selected = torch.device(selected)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise WanLongVideoError("device_unavailable", "CUDA is not available")
    temporal_scale = int(getattr(pipeline, "vae_scale_factor_temporal", contract.temporal_scale_factor))
    if temporal_scale != contract.temporal_scale_factor:
        raise WanLongVideoError("wan_geometry_mismatch", "pipeline temporal scale does not match the continuation contract")
    if int(spec.num_frames) != int(contract.window_frames) or int(spec.width) != int(contract.width) or int(spec.height) != int(contract.height):
        raise WanLongVideoError("wan_geometry_mismatch", "Wan config does not match the continuation contract")
    if selected.type == "cuda":
        torch.cuda.set_device(selected)
    transformer = pipeline.transformer
    if transformer is None:
        raise WanLongVideoError("transformer_missing", "Wan pipeline has no Transformer")
    latent_channels = int(transformer.config.in_channels)
    previous: Optional[torch.Tensor] = None
    decoded_windows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for index in range(contract.window_count):
        window_seed = continuation_seed(seed, index)
        generator = torch.Generator(device=str(selected)).manual_seed(window_seed)
        latents = pipeline.prepare_latents(
            1,
            latent_channels,
            contract.height,
            contract.width,
            contract.window_frames,
            torch.float32,
            selected,
            generator,
            None,
        )
        previous_tail = previous[..., -contract.overlap_latent_frames:, :, :].detach().clone() if previous is not None else None

        def on_step_end(_pipeline: Any, step_index: int, _timestep: Any, callback_kwargs: Mapping[str, Any]) -> dict[str, Any]:
            current = callback_kwargs["latents"]
            if previous_tail is None:
                return {"latents": current}
            weight = float(step_index + 1) / float(contract.inference_steps)
            return {"latents": anchor_wan_latent_prefix(current, previous_tail, contract.overlap_latent_frames, weight)}

        result = pipeline(
            prompt=str(prompt).strip(),
            height=contract.height,
            width=contract.width,
            num_frames=contract.window_frames,
            num_inference_steps=contract.inference_steps,
            guidance_scale=contract.guidance_scale,
            generator=generator,
            latents=latents,
            output_type="latent",
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        window_latent = getattr(result, "frames", None)
        if not isinstance(window_latent, torch.Tensor):
            raise WanLongVideoError("latent_output_missing", "Wan pipeline did not return latent output")
        expected_shape = (1, latent_channels, contract.latent_window_frames, contract.height // 8, contract.width // 8)
        if tuple(window_latent.shape) != expected_shape:
            raise WanLongVideoError("latent_shape_invalid", "Wan latent shape %s does not match %s" % (tuple(window_latent.shape), expected_shape))
        prefix_match = 0.0
        if previous_tail is not None:
            prefix_match = float(torch.max(torch.abs(window_latent[..., :contract.overlap_latent_frames, :, :] - previous_tail)).detach().float().cpu().item())
        decoded_windows.append(_decode_wan_latent(pipeline, window_latent))
        records.append(
            {
                "window_index": index,
                "seed": window_seed,
                "latent_shape": list(window_latent.shape),
                "latent_sha256": hashlib.sha256(window_latent.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),
                "previous_tail_sha256": hashlib.sha256(previous_tail.float().cpu().contiguous().numpy().tobytes()).hexdigest() if previous_tail is not None else None,
                "prefix_match_max_abs": prefix_match,
                "continuation": index == 0 or previous_tail is not None,
            }
        )
        previous = window_latent.detach()
        del result, latents, window_latent
        if selected.type == "cuda":
            torch.cuda.empty_cache()
    composed, composition = compose_wan_decoded_windows(decoded_windows, contract.decoded_overlap_frames)
    output = resample_wan_frames(composed, target_frames=contract.target_frames, height=contract.height, width=contract.width)
    trace = {
        "continuation_verified": all(
            bool(item["continuation"]) and float(item["prefix_match_max_abs"]) <= 1e-5 for item in records
        ),
        "window_spec": contract.to_dict(),
        "windows": records,
        "composition": composition,
        "composed_native_frames": int(composed.shape[0]),
        "output_frames": int(output.shape[0]),
        "offline_simulation": False,
        "contract_digest": contract.digest,
    }
    return output, trace


def validate_wan_long_video(path: Path, contract: WanLongVideoSpec) -> dict[str, Any]:
    """Decode and validate the long-video stability contract."""

    from .video_contract import inspect_video

    report = inspect_video(Path(path), max_decode_frames=max(4096, contract.target_frames + 8))
    if report.get("decode_limit_reached"):
        raise WanLongVideoError("video_decode_limit", "long video exceeded the bounded decode limit")
    if not 708 <= int(report.get("frames_read", 0)) <= 732:
        raise WanLongVideoError("frame_count_mismatch", "decoded frame count is outside 708-732")
    if int(report.get("width", 0)) != contract.width or int(report.get("height", 0)) != contract.height:
        raise WanLongVideoError("resolution_mismatch", "decoded resolution is not 256x256")
    if abs(float(report.get("fps", 0.0)) - contract.output_fps) > 0.5:
        raise WanLongVideoError("fps_mismatch", "decoded FPS is outside the 24 FPS contract")
    if not 29.5 <= float(report.get("duration_s", 0.0)) <= 30.5:
        raise WanLongVideoError("duration_mismatch", "decoded duration is outside 29.5-30.5 seconds")
    if float(report.get("black_frame_ratio", 1.0)) > 0.0:
        raise WanLongVideoError("black_frames", "long video contains black frames")
    if not bool(report.get("finite_pixels", False)):
        raise WanLongVideoError("nonfinite_pixels", "long video contains non-finite pixels")
    if int(report.get("frames_read", 0)) > 1 and float(report.get("adjacent_repeat_ratio", 1.0)) >= 1.0:
        raise WanLongVideoError("repeated_frames", "all decoded frames are repeated")
    if float(report.get("decoded_to_declared_ratio", 0.0)) < 0.99:
        raise WanLongVideoError("truncated_video", "decoded frames do not match the declared frame count")
    return {**report, "valid": True, "contract_digest": contract.digest, "contract": contract.to_dict()}


__all__ = [
    "WanLongVideoError",
    "WanLongVideoSpec",
    "anchor_wan_latent_prefix",
    "compose_wan_decoded_windows",
    "continuation_seed",
    "generate_wan_long_video",
    "load_wan_transformer_checkpoint",
    "resample_wan_frames",
    "validate_wan_long_video",
]
