from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


def _media_metrics(path: Path, expected: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return {"media_probe_available": False}, None

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"media_probe_available": True, "decodable": 0.0}, "decode_failed"
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    luma_values = []
    frame_diffs = []
    previous = None
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luma_values.append(float(gray.mean()))
        if previous is not None:
            frame_diffs.append(float(cv2.absdiff(gray, previous).mean()))
        previous = gray
    capture.release()
    if decoded == 0:
        return {"media_probe_available": True, "decodable": 0.0}, "decode_failed"

    mean_luma = sum(luma_values) / len(luma_values)
    black_ratio = sum(1 for value in luma_values if value <= 18.0) / len(luma_values)
    mean_diff = sum(frame_diffs) / len(frame_diffs) if frame_diffs else 0.0
    metrics: Dict[str, Any] = {
        "media_probe_available": True,
        "decodable": 1.0,
        "width": width,
        "height": height,
        "frames": decoded,
        "reported_frames": reported_frames,
        "mean_luma": round(mean_luma, 4),
        "black_frame_ratio": round(black_ratio, 4),
        "mean_frame_difference": round(mean_diff, 4),
        "luma_score": max(0.0, min(1.0, mean_luma / 80.0, (255.0 - mean_luma) / 40.0)),
        "stability_score": max(0.0, min(1.0, 1.0 - mean_diff / 64.0)),
        "motion_score": max(0.0, min(1.0, mean_diff / 4.0, (48.0 - mean_diff) / 16.0)),
    }
    if expected.get("width") is not None:
        metrics["width_match"] = 1.0 if width == int(expected["width"]) else 0.0
    if expected.get("height") is not None:
        metrics["height_match"] = 1.0 if height == int(expected["height"]) else 0.0
    if expected.get("frames") is not None:
        metrics["frames_match"] = 1.0 if decoded == int(expected["frames"]) else 0.0
    if mean_luma <= 18.0 or black_ratio > 0.25:
        return metrics, "low_luma"
    if mean_diff > 32.0 or mean_diff < 0.5:
        return metrics, "temporal_instability"
    return metrics, None


def evaluate(request: Mapping[str, Any]) -> Dict[str, Any]:
    backend_success = bool(request.get("backend_success", False))
    artifacts = [Path(str(item)) for item in request.get("artifacts", [])]
    existing = [path for path in artifacts if path.is_file()]
    non_empty = [path for path in existing if path.stat().st_size > 0]
    metrics: Dict[str, Any] = {
        "backend_success": 1.0 if backend_success else 0.0,
        "artifact_exists": 1.0 if existing else 0.0,
        "artifact_non_empty": 1.0 if non_empty else 0.0,
        "artifact_count": len(non_empty),
        "wall_time_s": float(request.get("wall_time_s", 0)),
    }
    failure_type = None
    critical = False
    if not backend_success:
        failure_type, critical = "backend_failure", True
    elif not existing:
        failure_type, critical = "artifact_missing", True
    elif not non_empty:
        failure_type, critical = "artifact_empty", True

    if non_empty:
        media_path = next((path for path in non_empty if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}), non_empty[0])
        media, media_failure = _media_metrics(media_path, request.get("expected") or {})
        metrics.update(media)
        if media_failure:
            failure_type = media_failure
            critical = media_failure in {"decode_failed", "low_luma"}
        elif any(metrics.get(name) == 0.0 for name in ("width_match", "height_match", "frames_match") if name in metrics):
            failure_type = "output_mismatch"
            critical = True

    weighted = [(metrics["backend_success"], 0.15), (metrics["artifact_exists"], 0.10), (metrics["artifact_non_empty"], 0.05)]
    for name, weight in (("decodable", 0.15), ("luma_score", 0.15), ("stability_score", 0.12), ("motion_score", 0.12), ("width_match", 0.04), ("height_match", 0.04), ("frames_match", 0.08)):
        if name in metrics:
            weighted.append((float(metrics[name]), weight))
    available_weight = sum(weight for _, weight in weighted)
    score = sum(value * weight for value, weight in weighted) / available_weight if available_weight else 0.0
    if critical:
        score = min(score, 0.49)
    return {
        "score": round(max(0.0, min(1.0, score)), 6),
        "metrics": metrics,
        "critical_regression": critical,
        "failure_type": failure_type,
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        print(json.dumps(evaluate(request), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print("evaluator: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
