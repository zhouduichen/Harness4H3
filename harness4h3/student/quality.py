"""Server-local semantic and temporal quality metrics for Student videos."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


class QualityBackendUnavailable(RuntimeError):
    """Raised when a configured real quality backend cannot run."""


@dataclass(frozen=True)
class QualityEvidence:
    semantic: float
    temporal: float
    motion: float
    aggregate: float
    backend: str
    model_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def aggregate_quality(*, semantic: float, temporal: float, motion: float) -> float:
    values = (float(semantic), float(temporal), float(motion))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("quality components must be finite")
    return _bounded(0.70 * values[0] + 0.20 * values[1] + 0.10 * values[2])


class ClipTemporalQualityBackend:
    """CLIP image-text alignment plus adjacent-frame temporal consistency."""

    def __init__(self, model_path: Path | str, *, device: str = "auto", sample_frames: int = 4):
        self.model_path = Path(model_path)
        self.device_name = device
        self.sample_frames = max(2, int(sample_frames))
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device: Optional[Any] = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._processor is not None and self._device is not None:
            return self._processor, self._model, self._device
        if not self.model_path.is_dir():
            raise QualityBackendUnavailable("CLIP model path does not exist: %s" % self.model_path)
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except (ImportError, OSError) as exc:
            raise QualityBackendUnavailable("CLIP dependencies are unavailable: %s" % exc) from exc
        try:
            processor = CLIPProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            model = CLIPModel.from_pretrained(str(self.model_path), local_files_only=True)
        except Exception as exc:
            raise QualityBackendUnavailable("unable to load local CLIP model: %s" % exc) from exc
        if self.device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_name)
        model.to(device)
        model.eval()
        self._processor, self._model, self._device = processor, model, device
        return processor, model, device

    def _read_frames(self, video_path: Path) -> list[Any]:
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise QualityBackendUnavailable("video quality dependencies are unavailable: %s" % exc) from exc
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise QualityBackendUnavailable("unable to open video for quality evaluation: %s" % video_path)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            capture.release()
            raise QualityBackendUnavailable("video has no frames: %s" % video_path)
        positions = np.linspace(0, count - 1, num=min(self.sample_frames, count), dtype=int).tolist()
        frames = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        capture.release()
        if len(frames) < 2:
            raise QualityBackendUnavailable("unable to sample enough video frames: %s" % video_path)
        return frames

    @staticmethod
    def _motion(frames: list[Any]) -> float:
        import numpy as np

        arrays = [np.asarray(frame).astype("float32") / 255.0 for frame in frames]
        deltas = [float(np.mean(np.abs(current - previous))) for previous, current in zip(arrays, arrays[1:])]
        return _bounded(sum(deltas) / max(1, len(deltas)) * 4.0)

    def evaluate(self, video_path: Path | str, caption: str) -> QualityEvidence:
        caption = str(caption).strip()
        if not caption:
            raise ValueError("quality evaluation requires a caption")
        processor, model, device = self._load()
        frames = self._read_frames(Path(video_path))
        try:
            import torch
            import torch.nn.functional as F

            image_inputs = processor(images=frames, return_tensors="pt")
            text_inputs = processor(text=[caption], return_tensors="pt", padding=True, truncation=True)
            image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            with torch.inference_mode():
                image_features = F.normalize(model.get_image_features(**image_inputs), dim=-1)
                text_features = F.normalize(model.get_text_features(**text_inputs), dim=-1)
                semantic_values = ((image_features @ text_features[0].unsqueeze(-1)).squeeze(-1) + 1.0) / 2.0
                temporal_values = (F.cosine_similarity(image_features[:-1], image_features[1:], dim=-1) + 1.0) / 2.0
            semantic = _bounded(float(semantic_values.mean().item()))
            temporal = _bounded(float(temporal_values.mean().item())) if temporal_values.numel() else 1.0
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise QualityBackendUnavailable("CLIP quality evaluation failed: %s" % exc) from exc
        motion = self._motion(frames)
        return QualityEvidence(
            semantic=semantic,
            temporal=temporal,
            motion=motion,
            aggregate=aggregate_quality(semantic=semantic, temporal=temporal, motion=motion),
            backend="clip_temporal",
            model_path=str(self.model_path),
        )


__all__ = ["ClipTemporalQualityBackend", "QualityBackendUnavailable", "QualityEvidence", "aggregate_quality"]
