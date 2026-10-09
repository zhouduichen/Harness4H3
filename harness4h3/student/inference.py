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
from h3_training.data.schema import Conditioning, ModalInterval, ModalLatents, ModalTimesteps

from .model import build_student
from .proposal import StudentProposal, StudentTarget
from .quantization import load_student_state
from .schedule import h3_video_sigmas
from .evaluation_manifest import CacheContract, validate_h3_cache_item


class StudentGenerationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


_VAE_CACHE: dict[tuple[str, str, str], Any] = {}
_COMFY_INITIALIZED: set[str] = set()


def student_video_sigmas(steps: int, *, shift: float = 12.0) -> tuple[float, ...]:
    """Return the H3 video sigma schedule used by Student training.

    Real H3 does not integrate over an unshifted linear sigma grid.  The
    training adapter applies the same video time shift as the Teacher sampler;
    keeping that schedule at inference is required for a distilled Student to
    see the intervals it was trained to predict.
    """

    return h3_video_sigmas(steps, shift=shift)


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
    cache_contract: Optional[CacheContract] = None,
) -> Mapping[str, Any]:
    selected = _resolve_cache_path(cache_dir, cache_path)
    if cache_contract is not None:
        try:
            validate_h3_cache_item(selected, cache_contract)
        except ValueError as exc:
            raise StudentGenerationError("invalid_h3_cache_contract", str(exc)) from exc
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
        "audio_latent": (
            RealMiniMaxH3Adapter._unpack_audio(
                torch.as_tensor(raw["audio"]).float(), device=torch.device("cpu"), dtype=torch.float32
            )
            if "audio" in raw else None
        ),
        "text_token_tags": (
            torch.as_tensor(raw["text_token_tags"]).detach().cpu()
            if raw.get("text_token_tags") is not None
            else None
        ),
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
    continuation_latent: Optional[torch.Tensor] = None,
    continuation_overlap_latent_frames: int = 0,
) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latent = torch.randn(
        (1, target.latent_channels, target.latent_frames, target.latent_height, target.latent_width),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    if continuation_latent is not None:
        if continuation_latent.ndim != 5 or tuple(continuation_latent.shape[:2]) != tuple(latent.shape[:2]) or tuple(continuation_latent.shape[-2:]) != tuple(latent.shape[-2:]):
            raise StudentGenerationError("invalid_continuation_latent", "continuation latent shape does not match Student window")
        overlap = int(continuation_overlap_latent_frames)
        if overlap <= 0 or overlap >= target.latent_frames or continuation_latent.shape[-3] < overlap:
            raise StudentGenerationError("invalid_continuation_overlap", "continuation overlap is outside the Student window")
    else:
        overlap = 0
    steps = int(proposal.training.target_steps)
    sigmas = student_video_sigmas(steps)
    for index in range(steps):
        if continuation_latent is not None:
            from .long_video import anchor_latent_prefix

            latent = anchor_latent_prefix(
                latent,
                continuation_latent,
                overlap,
                float(index) / float(max(1, steps)),
            )
        sigma = sigmas[index]
        next_sigma = sigmas[index + 1]
        timestep = torch.tensor([sigma], device=device, dtype=torch.float32)
        with torch.inference_mode():
            velocity = model(latent, prompt, timestep)
        if not torch.isfinite(velocity).all():
            raise StudentGenerationError("student_nonfinite_output", "Student produced a non-finite velocity")
        # The trained target is clean-minus-noise. Integrating from sigma=1
        # to sigma=0 therefore subtracts the negative sigma delta.
        latent = latent - (next_sigma - sigma) * velocity
        if continuation_latent is not None:
            from .long_video import anchor_latent_prefix

            latent = anchor_latent_prefix(latent, continuation_latent, overlap, float(index + 1) / float(steps))
    return latent


def sample_h3_latent(
    checkpoint: Path,
    prompt: torch.Tensor,
    video_shape: tuple[int, ...],
    audio_shape: tuple[int, ...],
    comfyui_root: Path,
    device: torch.device,
    *,
    sampling_steps: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    conditioning_metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[torch.Tensor, float]:
    """Run the authentic H3 teacher forward/scheduler loop for one case."""

    if int(sampling_steps) <= 0:
        raise StudentGenerationError("invalid_sampling_steps", "sampling_steps must be positive")
    adapter = RealMiniMaxH3Adapter(comfyui_root, device=str(device), dtype=dtype)
    role = adapter.load_role(Path(checkpoint), trainable=False)
    return sample_h3_latent_with_role(
        adapter,
        role,
        prompt,
        video_shape,
        audio_shape,
        device,
        sampling_steps=sampling_steps,
        seed=seed,
        dtype=dtype,
        conditioning_metadata=conditioning_metadata,
    )


def sample_h3_latent_with_role(
    adapter: RealMiniMaxH3Adapter,
    role: Any,
    prompt: torch.Tensor,
    video_shape: tuple[int, ...],
    audio_shape: tuple[int, ...],
    device: torch.device,
    *,
    sampling_steps: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    conditioning_metadata: Optional[Mapping[str, Any]] = None,
    continuation_latent: Optional[torch.Tensor] = None,
    continuation_overlap_latent_frames: int = 0,
) -> tuple[torch.Tensor, float]:
    """Sample with an already-loaded H3 role so a manifest reuses one model."""

    if int(sampling_steps) <= 0:
        raise StudentGenerationError("invalid_sampling_steps", "sampling_steps must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    video = torch.randn(video_shape, generator=generator, device="cpu", dtype=torch.float32).to(device=device, dtype=dtype)
    audio = torch.randn(audio_shape, generator=generator, device="cpu", dtype=torch.float32).to(device=device, dtype=dtype)
    schedule = adapter.schedule(int(sampling_steps))
    conditioning = Conditioning(prompt.to(device=device, dtype=dtype), metadata=dict(conditioning_metadata or {}))
    if continuation_latent is not None:
        overlap = int(continuation_overlap_latent_frames)
        if overlap <= 0 or overlap >= int(video_shape[-3]) or continuation_latent.shape[-3] < overlap:
            raise StudentGenerationError("invalid_continuation_overlap", "continuation overlap is outside the H3 window")
    else:
        overlap = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(int(sampling_steps)):
            if continuation_latent is not None:
                from .long_video import anchor_latent_prefix

                video = anchor_latent_prefix(video, continuation_latent, overlap, float(index) / float(sampling_steps))
            timestep = ModalTimesteps(
                video=torch.full((video.shape[0],), float(schedule.video_sigmas[index]), device=device, dtype=torch.float32),
                audio=torch.full((audio.shape[0],), float(schedule.audio_sigmas[index]), device=device, dtype=torch.float32),
            )
            prediction = adapter.predict(role, ModalLatents(video=video, audio=audio), timestep, conditioning)
            interval = ModalInterval(
                video=(schedule.video_sigmas[index], schedule.video_sigmas[index + 1]),
                audio=(schedule.audio_sigmas[index], schedule.audio_sigmas[index + 1]),
            )
            updated = adapter.scheduler_step(role, ModalLatents(video=video, audio=audio), prediction, interval)
            if updated.video is None or updated.audio is None:
                raise StudentGenerationError("h3_teacher_output_missing", "H3 teacher returned incomplete latent output")
            video, audio = updated.video, updated.audio
            if continuation_latent is not None:
                from .long_video import anchor_latent_prefix

                video = anchor_latent_prefix(video, continuation_latent, overlap, float(index + 1) / float(sampling_steps))
    return video, time.perf_counter() - started


def sample_h3_latent_with_predictor(
    predictor: Any,
    prompt: torch.Tensor,
    video_shape: tuple[int, ...],
    audio_shape: tuple[int, ...],
    device: torch.device,
    *,
    sampling_steps: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    conditioning_metadata: Optional[Mapping[str, Any]] = None,
    return_audio: bool = False,
    continuation_latent: Optional[torch.Tensor] = None,
    continuation_overlap_latent_frames: int = 0,
) -> tuple[torch.Tensor, float]:
    """Sample with the collective online Teacher request boundary.

    This is deliberately the same shifted H3 schedule and scheduler update as
    ``sample_h3_latent_with_role``; only the prediction provider changes from
    a local full model to the three-rank FSDP service.
    """

    if int(sampling_steps) <= 0:
        raise StudentGenerationError("invalid_sampling_steps", "sampling_steps must be positive")

    def shifted(value: float, amount: float) -> float:
        return amount * value / (1.0 + (amount - 1.0) * value)

    base = [1.0 - index / float(sampling_steps) for index in range(int(sampling_steps) + 1)]
    video_sigmas = tuple(shifted(value, 12.0) for value in base)
    audio_sigmas = tuple(shifted(value, 3.0) for value in base)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    video = torch.randn(video_shape, generator=generator, device="cpu", dtype=torch.float32).to(device=device, dtype=dtype)
    audio = torch.randn(audio_shape, generator=generator, device="cpu", dtype=torch.float32).to(device=device, dtype=dtype)
    conditioning = Conditioning(prompt.to(device=device, dtype=dtype), metadata=dict(conditioning_metadata or {}))
    if continuation_latent is not None:
        overlap = int(continuation_overlap_latent_frames)
        if overlap <= 0 or overlap >= int(video_shape[-3]) or continuation_latent.shape[-3] < overlap:
            raise StudentGenerationError("invalid_continuation_overlap", "continuation overlap is outside the H3 window")
    else:
        overlap = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(int(sampling_steps)):
            if continuation_latent is not None:
                from .long_video import anchor_latent_prefix

                video = anchor_latent_prefix(video, continuation_latent, overlap, float(index) / float(sampling_steps))
            timestep = ModalTimesteps(
                video=torch.full((video.shape[0],), float(video_sigmas[index]), device=device, dtype=torch.float32),
                audio=torch.full((audio.shape[0],), float(audio_sigmas[index]), device=device, dtype=torch.float32),
            )
            prediction = predictor.predict(
                ModalLatents(video=video, audio=audio), timestep, conditioning
            )
            if prediction.video is None or prediction.audio is None:
                raise StudentGenerationError("h3_teacher_output_missing", "online H3 Teacher returned incomplete output")
            video = video + (video_sigmas[index] - video_sigmas[index + 1]) * prediction.video.to(device=device, dtype=dtype)
            audio = audio + (audio_sigmas[index] - audio_sigmas[index + 1]) * prediction.audio.to(device=device, dtype=dtype)
            if continuation_latent is not None:
                from .long_video import anchor_latent_prefix

                video = anchor_latent_prefix(video, continuation_latent, overlap, float(index + 1) / float(sampling_steps))
    elapsed = time.perf_counter() - started
    if return_audio:
        return video, audio, elapsed
    return video, elapsed


def decode_video_latent(comfyui_root: Path, vae_name: str, latent: torch.Tensor) -> torch.Tensor:
    root = str(Path(comfyui_root).resolve())
    previous_cuda_device = None
    if latent.device.type == "cuda":
        previous_cuda_device = torch.cuda.current_device()
        torch.cuda.set_device(latent.device)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import nodes

        if root not in _COMFY_INITIALIZED:
            initialized = nodes.init_extra_nodes(init_custom_nodes=False)
            if inspect.isawaitable(initialized):
                asyncio.run(initialized)
            _COMFY_INITIALIZED.add(root)
        mapping = nodes.NODE_CLASS_MAPPINGS
        loader = mapping["VAELoader"]()
        cache_key = (root, str(vae_name), str(latent.device))
        vae = _VAE_CACHE.get(cache_key)
        if vae is None:
            vae = loader.load_vae(str(vae_name))[0]
            _VAE_CACHE[cache_key] = vae
        # Full-frame H3 VAE decoding can approach the capacity of a 48 GB
        # card even for the small fixed manifest. Tiled spatial+temporal
        # decoding keeps evaluation reproducible without changing the VAE.
        with torch.inference_mode():
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
    finally:
        if previous_cuda_device is not None:
            torch.cuda.set_device(previous_cuda_device)
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
    if not torch.isfinite(torch.as_tensor(fps)) or float(fps) <= 0:
        raise StudentGenerationError("invalid_video_fps", "video FPS must be finite and positive")
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
    output_fps: float = 24.0,
) -> Mapping[str, Any]:
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
    started = time.perf_counter()
    latent = sample_student_latent(model, prompt, proposal, target, selected, seed=seed)
    sampling_latency = time.perf_counter() - started
    del model
    if selected.type == "cuda":
        torch.cuda.empty_cache()
    decode_started = time.perf_counter()
    frames = decode_video_latent(comfyui_root, vae_name, latent)
    write_video(frames, output_path, fps=output_fps)
    decode_latency = time.perf_counter() - decode_started
    peak = torch.cuda.max_memory_allocated(selected) / float(1024**3) if selected.type == "cuda" else 0.0
    return {
        "video_path": str(Path(output_path).resolve()),
        "latency_s": float(sampling_latency + decode_latency),
        "sampling_latency_s": float(sampling_latency),
        "decode_latency_s": float(decode_latency),
        "generation_latency_s": float(sampling_latency + decode_latency),
        "peak_memory_gb": float(peak),
        "frame_count": int(frames.shape[0]),
        "resolution": [int(frames.shape[2]), int(frames.shape[1])],
        "fps": float(output_fps),
        "checkpoint_bytes": int(Path(checkpoint).stat().st_size),
        "device": str(selected),
    }


def generate_long_video(
    proposal: StudentProposal,
    checkpoint: Path,
    target: StudentTarget,
    comfyui_root: Path,
    cache_dir: Path,
    vae_name: str,
    output_path: Path,
    video_spec: Any,
    *,
    device: Optional[str] = None,
    cache_path: Optional[Path] = None,
    seed: int = 20260920,
) -> Mapping[str, Any]:
    """Generate one continuous video from authenticated overlap windows."""

    from .long_video import LongVideoWindowSpec, compose_decoded_windows, sample_student_long_latents

    if not hasattr(video_spec, "window_count") or int(video_spec.window_count) <= 1:
        raise StudentGenerationError("invalid_long_video_contract", "generate_long_video requires a multi-window VideoOutputSpec")
    try:
        video_spec.validate_window_geometry()
    except (TypeError, ValueError) as exc:
        raise StudentGenerationError("invalid_long_video_contract", str(exc)) from exc
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise StudentGenerationError("device_unavailable", "CUDA is not available")
    if selected.type == "cuda":
        torch.cuda.set_device(selected)
        torch.cuda.reset_peak_memory_stats(selected)
    dtype = torch.bfloat16 if proposal.deployment.precision == "bf16" else torch.float16
    prompt = _load_prompt(cache_dir, target, selected, dtype, cache_path=cache_path)
    model = load_student_model(proposal, checkpoint, target, selected)
    window_spec = LongVideoWindowSpec.from_video_spec(video_spec)
    started = time.perf_counter()
    latents, trace = sample_student_long_latents(
        model,
        prompt,
        proposal,
        target,
        selected,
        window_spec,
        seed=seed,
    )
    sampling_latency = time.perf_counter() - started
    del model
    if selected.type == "cuda":
        torch.cuda.empty_cache()
    decoded_windows: list[torch.Tensor] = []
    decode_started = time.perf_counter()
    try:
        for latent in latents:
            decoded_windows.append(decode_video_latent(comfyui_root, vae_name, latent))
    finally:
        del latents
    frames, composition = compose_decoded_windows(decoded_windows, window_spec.decoded_overlap_frames)
    write_video(frames, output_path, fps=float(video_spec.fps))
    decode_latency = time.perf_counter() - decode_started
    peak = torch.cuda.max_memory_allocated(selected) / float(1024**3) if selected.type == "cuda" else 0.0
    trace = {
        **trace,
        "contract_digest": video_spec.digest,
        "composition": composition,
        "expected_composed_frames": video_spec.expected_composed_frames,
        "offline_simulation": False,
    }
    return {
        "video_path": str(Path(output_path).resolve()),
        "latency_s": float(sampling_latency + decode_latency),
        "sampling_latency_s": float(sampling_latency),
        "decode_latency_s": float(decode_latency),
        "generation_latency_s": float(sampling_latency + decode_latency),
        "peak_memory_gb": float(peak),
        "frame_count": int(frames.shape[0]),
        "resolution": [int(frames.shape[2]), int(frames.shape[1])],
        "fps": float(video_spec.fps),
        "checkpoint_bytes": int(Path(checkpoint).stat().st_size),
        "device": str(selected),
        "long_video": True,
        "continuation_verified": bool(trace["continuation_verified"]),
        "offline_simulation": False,
        "video_contract": video_spec.to_dict(),
        "video_contract_digest": video_spec.digest,
        "window_trace": trace,
    }


__all__ = [
    "StudentGenerationError",
    "decode_video_latent",
    "generate_video",
    "generate_long_video",
    "load_h3_cache_item",
    "load_student_model",
    "sample_student_latent",
    "sample_h3_latent",
    "sample_h3_latent_with_role",
    "sample_h3_latent_with_predictor",
    "write_video",
]
