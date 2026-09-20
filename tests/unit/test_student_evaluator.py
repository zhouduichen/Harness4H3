from __future__ import annotations

from pathlib import Path

from harness4h3.student.evaluator import StudentEvaluator


def test_invalid_video_is_hard_failure(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not-video")
    result = StudentEvaluator().evaluate(path, quality={"score": 0.9}, hardware={"peak_vram_gb": 4.0})
    assert result.valid is False
    assert result.failure_code == "video_decode_failed"
    assert result.promotable is False


def test_missing_video_is_not_a_quality_success(tmp_path):
    result = StudentEvaluator().evaluate(tmp_path / "missing.mp4", quality={"score": 1.0})
    assert result.valid is False
    assert result.failure_code == "video_missing"
    assert result.quality_score is None


def test_quality_regression_is_valid_video_but_not_promotable(tmp_path):
    # A nonexistent path is intentionally not enough to exercise quality
    # regression; use a tiny valid MP4 emitted by OpenCV when available.
    cv2 = __import__("cv2")
    path = tmp_path / "valid.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (8, 8))
    try:
        import numpy as np

        for _ in range(2):
            writer.write(np.full((8, 8, 3), 128, dtype=np.uint8))
    finally:
        writer.release()
    result = StudentEvaluator().evaluate(path, quality={"score": 0.2}, baseline_quality=0.9)
    assert result.valid is True
    assert result.promotable is False
    assert result.failure_code == "quality_regression"
