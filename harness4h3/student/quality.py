"""Server-local semantic and temporal quality metrics for Student videos."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


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
    semantic_mean: float = 0.0
    semantic_p10: float = 0.0
    visual_metrics: dict[str, float] = field(default_factory=dict)
    visual_quality_gate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def aggregate_quality(*, semantic: float, temporal: float, motion: float) -> float:
    values = (float(semantic), float(temporal), float(motion))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("quality components must be finite")
    return _bounded(0.70 * values[0] + 0.20 * values[1] + 0.10 * values[2])


def visual_structure_metrics(
    frames: list[Any],
    *,
    low_texture_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Measure spatial information and sampled-frame change without a model.

    The metrics deliberately describe structure rather than brightness.  A
    smooth colour field can have a valid mean pixel value and a respectable
    CLIP score, but it has very little local gradient, Laplacian energy, or
    block-level variance.
    """

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise QualityBackendUnavailable("visual quality dependencies are unavailable: %s" % exc) from exc
    if len(frames) < 2:
        raise QualityBackendUnavailable("visual quality requires at least two sampled frames")
    gradients: list[float] = []
    laplacians: list[float] = []
    block_variances: list[float] = []
    arrays = []
    for frame in frames:
        array = np.asarray(frame).astype("float32")
        if array.ndim == 3:
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        elif array.ndim == 2:
            gray = array
        else:
            raise QualityBackendUnavailable("visual quality frame has unexpected shape: %s" % (array.shape,))
        if gray.size == 0:
            raise QualityBackendUnavailable("visual quality frame is empty")
        gray = np.clip(gray, 0.0, 255.0)
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        gradients.append(float((np.mean(np.abs(gx)) + np.mean(np.abs(gy))) * 0.5))
        laplacians.append(float(np.std(cv2.Laplacian(gray, cv2.CV_32F))))
        height, width = gray.shape[:2]
        block_h = max(1, height // 16)
        block_w = max(1, width // 16)
        cropped = gray[: block_h * 16, : block_w * 16]
        blocks = cropped.reshape(16, block_h, 16, block_w)
        block_variances.append(float(np.var(blocks, axis=(1, 3)).mean()))
        arrays.append(array / 255.0)
    frame_differences = [float(np.mean(np.abs(current - previous))) for previous, current in zip(arrays, arrays[1:])]

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _p10(values: list[float]) -> float:
        return float(np.quantile(np.asarray(values, dtype=np.float32), 0.10)) if values else 0.0

    metrics = {
        "spatial_gradient_mean": _mean(gradients),
        "spatial_gradient_p10": _p10(gradients),
        "laplacian_std_mean": _mean(laplacians),
        "laplacian_std_p10": _p10(laplacians),
        "block_variance_mean": _mean(block_variances),
        "block_variance_p10": _p10(block_variances),
        "frame_difference_mean": _mean(frame_differences),
        "frame_difference_p10": _p10(frame_differences),
    }
    if low_texture_thresholds and bool(low_texture_thresholds.get("enabled", False)):
        gradient_floor = float(low_texture_thresholds["min_spatial_gradient"])
        laplacian_floor = float(low_texture_thresholds["min_laplacian_std"])
        block_floor = float(low_texture_thresholds["min_block_variance"])
        low_frames = sum(
            int(gradient < gradient_floor or laplacian < laplacian_floor or block < block_floor)
            for gradient, laplacian, block in zip(gradients, laplacians, block_variances)
        )
        metrics["low_texture_frame_ratio"] = float(low_frames / len(gradients))
    return metrics


def evaluate_visual_quality_gate(
    metrics: Mapping[str, Any],
    gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the configured information-content hard gate."""

    config = dict(gate or {})
    if not bool(config.get("enabled", False)):
        return {"enabled": False, "passed": True, "failures": [], "version": config.get("version", "disabled")}
    required = {
        "min_spatial_gradient": "spatial_gradient_mean",
        "min_laplacian_std": "laplacian_std_mean",
        "min_block_variance": "block_variance_mean",
        "min_frame_difference": "frame_difference_mean",
    }
    failures: list[str] = []
    thresholds: dict[str, float] = {}
    for threshold_name, metric_name in required.items():
        threshold = float(config[threshold_name])
        value = float(metrics.get(metric_name, float("nan")))
        thresholds[threshold_name] = threshold
        if not math.isfinite(value) or value < threshold:
            failures.append("%s_below_floor" % metric_name)
    max_low_ratio = float(config["max_low_texture_frame_ratio"])
    low_ratio = float(metrics.get("low_texture_frame_ratio", float("nan")))
    thresholds["max_low_texture_frame_ratio"] = max_low_ratio
    if not math.isfinite(low_ratio) or low_ratio > max_low_ratio:
        failures.append("low_texture_frame_ratio_above_ceiling")
    return {
        "enabled": True,
        "passed": not failures,
        "failures": failures,
        "version": str(config.get("version", "visual-structure-v1")),
        "thresholds": thresholds,
        "low_texture_frame_ratio": low_ratio,
    }


class ClipTemporalQualityBackend:
    """CLIP image-text alignment plus adjacent-frame temporal consistency."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        device: str = "auto",
        sample_frames: int = 4,
        drift_penalty: float = 0.0,
        visual_quality_gate: Mapping[str, Any] | None = None,
    ):
        self.model_path = Path(model_path)
        self.device_name = device
        self.sample_frames = max(2, int(sample_frames))
        if not 0.0 <= float(drift_penalty) <= 1.0:
            raise ValueError("drift_penalty must be in [0, 1]")
        self.drift_penalty = float(drift_penalty)
        self.visual_quality_gate = dict(visual_quality_gate or {})
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

    @staticmethod
    def _feature_tensor(output: Any, torch_module: Any) -> Any:
        """Normalize the transformers 4.x tensor and 5.x model-output APIs."""

        if torch_module.is_tensor(output):
            return output
        for name in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            value = getattr(output, name, None)
            if value is not None and torch_module.is_tensor(value):
                if name == "last_hidden_state" and value.ndim == 3:
                    return value[:, 0]
                return value
        if isinstance(output, dict):
            for name in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
                value = output.get(name)
                if value is not None and torch_module.is_tensor(value):
                    if name == "last_hidden_state" and value.ndim == 3:
                        return value[:, 0]
                    return value
        raise TypeError("CLIP feature output has no tensor representation")

    def evaluate(self, video_path: Path | str, caption: str) -> QualityEvidence:
        caption = str(caption).strip()
        if not caption:
            raise ValueError("quality evaluation requires a caption")
        processor, model, device = self._load()
        frames = self._read_frames(Path(video_path))
        visual_metrics = visual_structure_metrics(
            frames,
            low_texture_thresholds=self.visual_quality_gate,
        )
        visual_quality_gate = evaluate_visual_quality_gate(visual_metrics, self.visual_quality_gate)
        try:
            import torch
            import torch.nn.functional as F

            image_inputs = processor(images=frames, return_tensors="pt")
            text_inputs = processor(text=[caption], return_tensors="pt", padding=True, truncation=True)
            image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            with torch.inference_mode():
                image_features = F.normalize(self._feature_tensor(model.get_image_features(**image_inputs), torch), dim=-1)
                text_features = F.normalize(self._feature_tensor(model.get_text_features(**text_inputs), torch), dim=-1)
                semantic_values = ((image_features @ text_features[0].unsqueeze(-1)).squeeze(-1) + 1.0) / 2.0
                temporal_values = (F.cosine_similarity(image_features[:-1], image_features[1:], dim=-1) + 1.0) / 2.0
            semantic_mean = _bounded(float(semantic_values.mean().item()))
            semantic_p10 = _bounded(float(torch.quantile(semantic_values.float(), 0.10).item()))
            semantic = _bounded((1.0 - self.drift_penalty) * semantic_mean + self.drift_penalty * semantic_p10)
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
            semantic_mean=semantic_mean,
            semantic_p10=semantic_p10,
            visual_metrics=visual_metrics,
            visual_quality_gate=visual_quality_gate,
        )


__all__ = [
    "ClipTemporalQualityBackend",
    "QualityBackendUnavailable",
    "QualityEvidence",
    "aggregate_quality",
    "evaluate_visual_quality_gate",
    "visual_structure_metrics",
]
