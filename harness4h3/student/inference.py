"""Load and sample a compiled Student, then decode its video latent through H3 VAE."""

from __future__ import annotations

import sys
import time
import asyncio
import inspect
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from h3_training.adapters.real_h3 import RealMiniMaxH3Adapter

from .model import build_student
from .proposal import StudentProposal, StudentTarget
from .quantization import load_student_state


class StudentGenerationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


def _resolve_cache_path(cache_dir: Path, cache_path: Optional[Path]) -> Path:
    if cache_path is not None:
        selected = Path(cache_path).resolve()
        if not selected.is_file():
            raise StudentGenerationError("cache_missing", "H3 cache item does not exist: %s" % selected)
        return selected
    paths = sorted(Path(cache_dir).glob("*.pt"))
    if not paths:
        raise StudentGenerationError("cache_missing", "no H3 cache item under %s" % cache_dir)
    return paths[0].resolve()


def load_h3_cache_item(
    cache_dir: Path,
    target: StudentTarget,
    device: torch.device,
    dtype: torch.dtype,
    *,
    cache_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    selected = _resolve_cache_path(cache_dir, cache_path)
    try:
        raw = torch.load(selected, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise StudentGenerationError("cache_corrupt", str(exc)) from exc
    if not isinstance(raw, Mapping) or "prompt" not in raw or "video" not in raw:
        raise StudentGenerationError("invalid_h3_cache", "cache item lacks prompt or video: %s" % selected)
    prompt = torch.as_tensor(raw["prompt"])
    if prompt.ndim != 2 or prompt.shape[1] != target.condition_dim:
        raise StudentGenerationError("invalid_h3_conditioning", "expected prompt [tokens,%d], got %s" % (target.condition_dim, tuple(prompt.shape)))
    try:
        latent = RealMiniMaxH3Adapter._unpatch_video(
            torch.as_tensor(raw["video"]).float(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            latent_frames=int(raw.get("latent_frames", target.latent_frames)),
        )
    except Exception as exc:
        raise StudentGenerationError("invalid_h3_latent", str(exc)) from exc
    expected = (1, target.latent_channels, target.latent_frames, target.latent_height, target.latent_width)
    if tuple(latent.shape) != expected:
        raise StudentGenerationError("student_target_mismatch", "H3 cache latent %s does not match target %s" % (tuple(latent.shape), expected))
    return {
        "path": selected,
        "prompt": prompt.unsqueeze(0).to(device=device, dtype=dtype),
        "latent": latent,
        "caption": str(raw.get("caption", "")).strip(),
    }


def _load_prompt(
    cache_dir: Path,
    target: StudentTarget,
    device: torch.device,
    dtype: torch.dtype,
    *,
    cache_path: Optional[Path] = None,
) -> torch.Tensor:
    return torch.as_tensor(
        load_h3_cache_item(cache_dir, target, device, dtype, cache_path=cache_path)["prompt"]
    )


def load_student_model(
    proposal: StudentProposal,
    checkpoint: Path,
    target: StudentTarget,
    device: torch.device,
) -> torch.nn.Module:
    dtype = torch.bfloat16 if proposal.deployment.precision == "bf16" else torch.float16
    try:
        state, _metadata = load_student_state(Path(checkpoint))
        model = build_student(proposal, device=device, target=target).to(dtype=dtype)
        missing, unexpected = model.load_state_dict(state, strict=False)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise StudentGenerationError("student_checkpoint_load_failed", str(exc)) from exc
    if missing or unexpected:
        raise StudentGenerationError(
            "student_checkpoint_incompatible",
            "missing=%s unexpected=%s" % (list(missing)[:4], list(unexpected)[:4]),
        )
    model.eval()
    return model


def sample_student_latent(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    proposal: StudentProposal,
    target: StudentTarget,
    device: torch.device,
    *,
    seed: int = 20260920,
) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latent = torch.randn(
        (1, target.latent_channels, target.latent_frames, target.latent_height, target.latent_width),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    steps = int(proposal.training.target_steps)
    for index in range(steps):
        sigma = 1.0 - index / float(steps)
        next_sigma = 1.0 - (index + 1) / float(steps)
        timestep = torch.tensor([sigma], device=device, dtype=torch.float32)
        with torch.inference_mode():
            velocity = model(latent, prompt, timestep)
        if not torch.isfinite(velocity).all():
            raise StudentGenerationError("student_nonfinite_output", "Student produced a non-finite velocity")
        # The trained target is clean-minus-noise. Integrating from sigma=1
        # to sigma=0 therefore subtracts the negative sigma delta.
        latent = latent - (next_sigma - sigma) * velocity
    return latent


def decode_video_latent(comfyui_root: Path, vae_name: str, latent: torch.Tensor) -> torch.Tensor:
    root = str(Path(comfyui_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import nodes

        initialized = nodes.init_extra_nodes(init_custom_nodes=False)
        if inspect.isawaitable(initialized):
            asyncio.run(initialized)
        mapping = nodes.NODE_CLASS_MAPPINGS
        loader = mapping["VAELoader"]()
        vae = loader.load_vae(str(vae_name))[0]
        # Full-frame H3 VAE decoding can approach the capacity of a 48 GB
        # card even for the small fixed manifest. Tiled spatial+temporal
        # decoding keeps evaluation reproducible without changing the VAE.
        decoded = mapping["VAEDecodeTiled"]().decode(
            vae,
            {"samples": latent},
            tile_size=128,
            overlap=32,
            temporal_size=8,
            temporal_overlap=2,
        )[0]
    except (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise StudentGenerationError("student_vae_decode_failed", str(exc)) from exc
    if not isinstance(decoded, torch.Tensor) or decoded.ndim not in (4, 5):
        raise StudentGenerationError("student_vae_decode_failed", "VAE output must be a 4D/5D tensor")
    value = decoded.detach()
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise StudentGenerationError("student_vae_decode_failed", "VAE output batch must be one")
        value = value[0]
    if value.shape[-1] in (3, 4):
        frames = value
    elif value.shape[1] in (3, 4):
        frames = value.permute(0, 2, 3, 1)
    else:
        raise StudentGenerationError("student_vae_decode_failed", "unable to identify RGB channel axis in %s" % (tuple(value.shape),))
    return frames.float().cpu()


def write_video(frames: torch.Tensor, path: Path, *, fps: float = 24.0) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise StudentGenerationError("video_writer_unavailable", str(exc)) from exc
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4) or frames.shape[0] <= 0:
        raise StudentGenerationError("student_video_empty", "decoded frames have invalid shape %s" % (tuple(frames.shape),))
    value = frames[..., :3]
    if float(value.min()) < 0:
        value = (value + 1.0) / 2.0
    if float(value.max()) <= 1.5:
        value = value * 255.0
    value = value.clamp(0, 255).to(torch.uint8).numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(value.shape[1]), int(value.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise StudentGenerationError("video_writer_failed", "unable to open %s" % path)
    try:
        for frame in value:
            writer.write(np.ascontiguousarray(frame[..., ::-1]))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise StudentGenerationError("video_writer_failed", "video output is empty")


def generate_video(
    proposal: StudentProposal,
    checkpoint: Path,
    target: StudentTarget,
    comfyui_root: Path,
    cache_dir: Path,
    vae_name: str,
    output_path: Path,
    *,
    device: Optional[str] = None,
    cache_path: Optional[Path] = None,
    seed: int = 20260920,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise StudentGenerationError("device_unavailable", "CUDA is not available")
    if selected.type == "cuda":
        # Initialize the selected CUDA context before querying allocator
        # statistics.  On a fresh worker, reset_peak_memory_stats(device)
        # otherwise passes the raw index to an uninitialized CUDA runtime.
        torch.cuda.set_device(selected)
        torch.cuda.reset_peak_memory_stats(selected)
    dtype = torch.bfloat16 if proposal.deployment.precision == "bf16" else torch.float16
    prompt = _load_prompt(cache_dir, target, selected, dtype, cache_path=cache_path)
    model = load_student_model(proposal, checkpoint, target, selected)
    latent = sample_student_latent(model, prompt, proposal, target, selected, seed=seed)
    del model
    if selected.type == "cuda":
        torch.cuda.empty_cache()
    frames = decode_video_latent(comfyui_root, vae_name, latent)
    write_video(frames, output_path)
    peak = torch.cuda.max_memory_allocated(selected) / float(1024**3) if selected.type == "cuda" else 0.0
    return {
        "video_path": str(Path(output_path).resolve()),
        "latency_s": time.perf_counter() - started,
        "peak_memory_gb": float(peak),
        "frame_count": int(frames.shape[0]),
        "resolution": [int(frames.shape[2]), int(frames.shape[1])],
        "checkpoint_bytes": int(Path(checkpoint).stat().st_size),
        "device": str(selected),
    }


__all__ = [
    "StudentGenerationError",
    "decode_video_latent",
    "generate_video",
    "load_h3_cache_item",
    "load_student_model",
    "sample_student_latent",
    "write_video",
]
