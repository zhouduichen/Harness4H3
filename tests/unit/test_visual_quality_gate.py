from __future__ import annotations

import cv2
import numpy as np

from harness4h3.student.quality import evaluate_visual_quality_gate, visual_structure_metrics


GATE = {
    "enabled": True,
    "version": "visual-structure-v1",
    "min_spatial_gradient": 0.30,
    "min_laplacian_std": 1.8,
    "min_block_variance": 12.0,
    "min_frame_difference": 0.01,
    "max_low_texture_frame_ratio": 0.25,
}


def test_smooth_colour_field_fails_visual_gate():
    frames = [np.full((64, 64, 3), (120 + index) % 255, dtype=np.uint8) for index in range(12)]
    metrics = visual_structure_metrics(frames, low_texture_thresholds=GATE)
    decision = evaluate_visual_quality_gate(metrics, GATE)

    assert decision["passed"] is False
    assert "spatial_gradient_mean_below_floor" in decision["failures"]
    assert decision["low_texture_frame_ratio"] == 1.0


def test_textured_motion_sample_passes_visual_gate():
    frames = []
    for index in range(12):
        rng = np.random.default_rng(1000 + index)
        frame = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (index, 8), (index + 20, 32), (255, 0, 0), thickness=-1)
        frames.append(frame)
    metrics = visual_structure_metrics(frames, low_texture_thresholds=GATE)
    decision = evaluate_visual_quality_gate(metrics, GATE)

    assert decision["passed"] is True
    assert decision["failures"] == []

