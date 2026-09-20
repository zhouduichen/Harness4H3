"""Independent video validity and hardware/quality evidence normalization."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class StudentEvaluation:
    valid: bool
    promotable: bool
    failure_code: Optional[str]
    message: str
    video_path: str
    validity: Mapping[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    quality_metrics: Mapping[str, Any] = field(default_factory=dict)
    hardware: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudentExperienceRecord:
    experience_id: str
    proposal_digest: str
    compiler_digest: str
    teacher_sha256: str
    parent_checkpoint: Optional[str]
    child_checkpoint: Optional[str]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    outcome: str
    failure_code: Optional[str]
    diagnosis: str
    next_round_hints: tuple[str, ...]
    created_at: str
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["next_round_hints"] = list(self.next_round_hints)
        return value


class StudentEvaluator:
    def __init__(
        self,
        *,
        black_frame_ratio_threshold: float = 0.0,
        max_frames: int = 64,
        max_quality_drop: float = 0.05,
    ):
        if not 0 <= float(black_frame_ratio_threshold) <= 1:
            raise ValueError("black_frame_ratio_threshold must be in [0, 1]")
        if int(max_frames) <= 0:
            raise ValueError("max_frames must be positive")
        if float(max_quality_drop) < 0 or not math.isfinite(float(max_quality_drop)):
            raise ValueError("max_quality_drop must be finite and non-negative")
        self.black_frame_ratio_threshold = float(black_frame_ratio_threshold)
        self.max_frames = int(max_frames)
        self.max_quality_drop = float(max_quality_drop)

    @staticmethod
    def _failure(path: Path, code: str, message: str, validity: Optional[Mapping[str, Any]] = None, **kwargs) -> StudentEvaluation:
        return StudentEvaluation(
            valid=False,
            promotable=False,
            failure_code=code,
            message=message,
            video_path=str(path),
            validity=dict(validity or {}),
            **kwargs,
        )

    def evaluate(
        self,
        video_path: Path,
        *,
        quality: Optional[Mapping[str, Any]] = None,
        hardware: Optional[Mapping[str, Any]] = None,
        baseline_quality: Optional[float] = None,
    ) -> StudentEvaluation:
        path = Path(video_path).resolve()
        quality = dict(quality or {})
        hardware = dict(hardware or {})
        if not path.is_file():
            return self._failure(path, "video_missing", "video artifact does not exist", hardware=hardware, quality_metrics=quality)
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            return self._failure(path, "video_reader_unavailable", str(exc), hardware=hardware, quality_metrics=quality)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return self._failure(path, "video_decode_failed", "video container could not be opened", hardware=hardware, quality_metrics=quality)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = 0
        black = 0
        finite = True
        means = []
        try:
            while frames < self.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frames += 1
                array = np.asarray(frame)
                finite = finite and bool(np.isfinite(array).all())
                mean = float(array.mean()) if array.size else 0.0
                means.append(mean)
                if mean <= 1.0:
                    black += 1
        finally:
            capture.release()
        validity = {
            "frames_read": frames,
            "declared_frames": declared_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_s": frames / fps if fps > 0 else 0.0,
            "black_frame_ratio": black / float(frames) if frames else 1.0,
            "finite_pixels": finite,
            "sample_mean": sum(means) / len(means) if means else 0.0,
        }
        if frames <= 0 or width <= 0 or height <= 0:
            return self._failure(path, "video_empty", "video has no readable frames", validity, hardware=hardware, quality_metrics=quality)
        if not finite:
            return self._failure(path, "video_nonfinite", "video contains non-finite pixels", validity, hardware=hardware, quality_metrics=quality)
        if validity["black_frame_ratio"] > self.black_frame_ratio_threshold:
            return self._failure(path, "video_black_frames", "black-frame ratio exceeds threshold", validity, hardware=hardware, quality_metrics=quality)
        score = quality.get("score")
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                return self._failure(path, "quality_invalid", "quality score is not numeric", validity, hardware=hardware, quality_metrics=quality)
            if not math.isfinite(score):
                return self._failure(path, "quality_invalid", "quality score is not finite", validity, hardware=hardware, quality_metrics=quality)
        if baseline_quality is not None and score is not None and score < float(baseline_quality) - self.max_quality_drop:
            return StudentEvaluation(
                valid=True,
                promotable=False,
                failure_code="quality_regression",
                message="quality score regressed beyond configured tolerance",
                video_path=str(path),
                validity=validity,
                quality_score=score,
                quality_metrics=quality,
                hardware=hardware,
            )
        return StudentEvaluation(
            valid=True,
            promotable=True,
            failure_code=None,
            message="evaluation_ok",
            video_path=str(path),
            validity=validity,
            quality_score=score,
            quality_metrics=quality,
            hardware=hardware,
        )


def append_experience(path: Path, record: StudentExperienceRecord) -> None:
    """Append and fsync one compact experience record before retention."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def make_experience_record(
    *,
    experience_id: str,
    proposal_digest: str,
    compiler_digest: str,
    teacher_sha256: str,
    parent_checkpoint: Optional[str],
    child_checkpoint: Optional[str],
    training: Mapping[str, Any],
    evaluation: StudentEvaluation,
    outcome: str,
    diagnosis: str,
    next_round_hints: tuple[str, ...] = (),
) -> StudentExperienceRecord:
    return StudentExperienceRecord(
        experience_id=str(experience_id),
        proposal_digest=str(proposal_digest),
        compiler_digest=str(compiler_digest),
        teacher_sha256=str(teacher_sha256),
        parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None,
        child_checkpoint=str(child_checkpoint) if child_checkpoint else None,
        training=dict(training),
        evaluation=evaluation.to_dict(),
        outcome=str(outcome),
        failure_code=evaluation.failure_code,
        diagnosis=str(diagnosis),
        next_round_hints=tuple(str(item) for item in next_round_hints),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


__all__ = [
    "StudentEvaluation",
    "StudentEvaluator",
    "StudentExperienceRecord",
    "append_experience",
    "make_experience_record",
]
