"""Fail-closed adapter for the pretrained Wan2.1 T2V Student.

This module is deliberately independent from the H3 latent Student.  Wan has
its own VAE, text encoder, scheduler, and latent geometry; forcing its
checkpoint through ``VideoLatentDiT`` would silently discard most of the
pretrained prior.  Optional Diffusers imports are lazy so configuration and
contract tests do not download or instantiate a 1.3B model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import yaml


class WanStudentError(RuntimeError):
    """A fail-closed Wan Student integration error."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        self.message = str(message)
        super().__init__("%s: %s" % (self.code, self.message))


@dataclass(frozen=True)
class WanStudentSpec:
    """Reproducible inference contract for the pretrained Wan Student."""

    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    lineage_root: str = "W0000"
    width: int = 256
    height: int = 256
    num_frames: int = 81
    native_fps: float = 16.0
    output_fps: float = 24.0
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        if not str(self.model_id).strip():
            raise ValueError("model_id must not be empty")
        if not str(self.lineage_root).startswith("W"):
            raise ValueError("lineage_root must start with W")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 8:
                raise ValueError("%s must be a positive multiple of 8" % name)
        if isinstance(self.num_frames, bool) or not isinstance(self.num_frames, int) or self.num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        if isinstance(self.num_inference_steps, bool) or not isinstance(self.num_inference_steps, int) or self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        for name in ("native_fps", "output_fps", "guidance_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if self.dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError("dtype must be one of bf16, fp16, fp32")

    @property
    def native_duration_s(self) -> float:
        return float(self.num_frames) / float(self.native_fps)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["pretrained"] = True
        value["native_duration_s"] = self.native_duration_s
        return value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WanStudentSpec":
        if not isinstance(raw, Mapping):
            raise ValueError("wan student config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError("wan student config has unknown field(s): %s" % ", ".join(map(str, unknown)))
        values = asdict(cls())
        values.update(dict(raw))
        try:
            return cls(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid wan student config: %s" % exc) from exc


def load_wan_spec(path: Path) -> WanStudentSpec:
    """Load a standalone Wan config without touching model files."""

    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WanStudentError("config_read_failed", str(exc)) from exc
    if isinstance(raw, Mapping) and "wan_student" in raw:
        raw = raw["wan_student"]
    try:
        return WanStudentSpec.from_mapping(raw or {})
    except (TypeError, ValueError) as exc:
        raise WanStudentError("config_invalid", str(exc)) from exc


def _torch_dtype(torch: Any, spec: WanStudentSpec) -> Any:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[spec.dtype]


def _resolve_device(torch: Any, device: Optional[str]) -> Any:
    selected = str(device or "auto")
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(selected)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise WanStudentError("device_unavailable", "CUDA was requested but is not available")
    return resolved


def load_pretrained_wan(
    spec: WanStudentSpec,
    *,
    device: Optional[str] = None,
    local_files_only: bool = False,
) -> Tuple[Any, Any]:
    """Load the real Wan checkpoint; never substitute a random model."""

    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise WanStudentError("torch_missing", "install the optional Wan training/runtime dependencies") from exc
    try:
        diffusers = importlib.import_module("diffusers")
        pipeline_class = getattr(diffusers, "WanPipeline")
    except (ImportError, AttributeError) as exc:
        raise WanStudentError(
            "diffusers_missing",
            "install the optional 'wan' dependency; no random-init fallback is permitted",
        ) from exc
    selected = _resolve_device(torch, device)
    try:
        pipeline = pipeline_class.from_pretrained(
            spec.model_id,
            torch_dtype=_torch_dtype(torch, spec),
            local_files_only=bool(local_files_only),
        )
        pipeline = pipeline.to(selected)
    except Exception as exc:  # Diffusers raises several model/cache-specific exception types.
        raise WanStudentError("pretrained_load_failed", "%s: %s" % (spec.model_id, exc)) from exc
    if hasattr(getattr(pipeline, "vae", None), "enable_tiling"):
        pipeline.vae.enable_tiling()
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    return pipeline, selected


def _as_frame_array(frames: Any) -> Any:
    """Normalize Diffusers output to uint8 ``[T,H,W,3]`` without guessing."""

    try:
        import numpy as np
    except ImportError as exc:
        raise WanStudentError("numpy_missing", "numpy is required to validate Wan video output") from exc
    if hasattr(frames, "detach"):
        frames = frames.detach().cpu().numpy()
    elif not hasattr(frames, "shape"):
        try:
            frames = np.asarray([np.asarray(frame) for frame in frames])
        except Exception as exc:
            raise WanStudentError("video_output_invalid", "unable to convert pipeline frames") from exc
    frames = np.asarray(frames)
    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise WanStudentError("video_output_invalid", "expected one generated video, got batch %s" % (frames.shape,))
        frames = frames[0]
    if frames.ndim != 4:
        raise WanStudentError("video_output_invalid", "expected [T,H,W,C] frames, got %s" % (frames.shape,))
    if frames.shape[-1] not in (3, 4) and frames.shape[1] in (3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.shape[-1] != 3:
        raise WanStudentError("video_output_invalid", "expected RGB frames, got %s" % (frames.shape,))
    frames = frames.astype(np.float32, copy=False)
    if float(frames.min()) < 0.0:
        frames = (frames + 1.0) * 127.5
    elif float(frames.max()) <= 1.0:
        frames = frames * 255.0
    return np.clip(frames, 0.0, 255.0).astype(np.uint8)


def generate_wan_clip(
    pipeline: Any,
    spec: WanStudentSpec,
    prompt: str,
    *,
    seed: int = 20260929,
    device: Any = None,
) -> Tuple[Any, Mapping[str, Any]]:
    """Generate one clip from the loaded pretrained Wan pipeline."""

    if not str(prompt).strip():
        raise WanStudentError("prompt_empty", "prompt must not be empty")
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise WanStudentError("torch_missing", "torch is required for seeded generation") from exc
    selected = device or _resolve_device(torch, None)
    generator = torch.Generator(device=str(selected)).manual_seed(int(seed))
    try:
        result = pipeline(
            prompt=str(prompt).strip(),
            height=int(spec.height),
            width=int(spec.width),
            num_frames=int(spec.num_frames),
            num_inference_steps=int(spec.num_inference_steps),
            guidance_scale=float(spec.guidance_scale),
            generator=generator,
        )
    except Exception as exc:
        raise WanStudentError("generation_failed", str(exc)) from exc
    raw_frames = getattr(result, "frames", None)
    if raw_frames is None:
        raise WanStudentError("video_output_missing", "Wan pipeline result has no frames")
    frames = _as_frame_array(raw_frames)
    if int(frames.shape[0]) != int(spec.num_frames):
        raise WanStudentError(
            "frame_count_mismatch",
            "expected %d native frames, got %d" % (spec.num_frames, frames.shape[0]),
        )
    if tuple(frames.shape[1:3]) != (spec.height, spec.width):
        raise WanStudentError(
            "resolution_mismatch",
            "expected %dx%d, got %dx%d" % (spec.width, spec.height, frames.shape[2], frames.shape[1]),
        )
    metadata = {
        "lineage_root": spec.lineage_root,
        "pretrained": True,
        "model_id": spec.model_id,
        "seed": int(seed),
        "native_fps": float(spec.native_fps),
        "target_output_fps": float(spec.output_fps),
        "native_frame_count": int(frames.shape[0]),
        "native_duration_s": float(spec.native_duration_s),
        "resolution": [int(spec.width), int(spec.height)],
        "num_inference_steps": int(spec.num_inference_steps),
        "guidance_scale": float(spec.guidance_scale),
    }
    return frames, metadata


def write_wan_video(frames: Any, path: Path, fps: float) -> None:
    """Write validated RGB frames using the repository's OpenCV dependency."""

    try:
        import cv2
    except ImportError as exc:
        raise WanStudentError("opencv_missing", "opencv-python-headless is required to write video") from exc
    value = _as_frame_array(frames)
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise WanStudentError("fps_invalid", "fps must be positive and finite")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (value.shape[2], value.shape[1]))
    if not writer.isOpened():
        raise WanStudentError("video_writer_failed", "unable to open %s" % output)
    try:
        for frame in value:
            writer.write(frame[..., ::-1])
    finally:
        writer.release()
    if not output.is_file() or output.stat().st_size <= 0:
        raise WanStudentError("video_writer_failed", "video output is empty")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a fail-closed clip from pretrained Wan2.1-T2V-1.3B")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    spec = load_wan_spec(args.config)
    if args.dry_run:
        print(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        raise WanStudentError("output_missing", "--output is required unless --dry-run is used")
    pipeline, device = load_pretrained_wan(
        spec,
        device=args.device,
        local_files_only=bool(args.local_files_only),
    )
    frames, metadata = generate_wan_clip(pipeline, spec, args.prompt, seed=args.seed, device=device)
    # W0 is a native Wan baseline.  Do not relabel 16 FPS frames as 24 FPS;
    # temporal interpolation/resampling belongs to W2 and must be evaluated
    # separately from the pretrained baseline.
    write_wan_video(frames, args.output, spec.native_fps)
    metadata = {
        **metadata,
        "video_path": str(Path(args.output).resolve()),
        "written_fps": float(spec.native_fps),
        "target_output_fps": float(spec.output_fps),
    }
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "WanStudentError",
    "WanStudentSpec",
    "generate_wan_clip",
    "load_pretrained_wan",
    "load_wan_spec",
    "write_wan_video",
    "main",
]
