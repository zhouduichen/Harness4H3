import numpy as np
import torch

from harness4h3.student.wan_long_video import (
    WanLongVideoSpec,
    anchor_wan_latent_prefix,
    compose_wan_decoded_windows,
    resample_wan_frames,
)


def test_wan_long_contract_targets_roughly_thirty_seconds():
    contract = WanLongVideoSpec()
    assert contract.latent_window_frames == 21
    assert contract.decoded_overlap_frames == 25
    assert contract.window_stride_latent_frames == 14
    assert contract.composed_native_frames == 473
    assert contract.target_frames == 720
    assert 29.9 < contract.output_duration_s < 30.1


def test_wan_latent_prefix_is_exactly_carried_at_final_weight():
    latent = torch.zeros(1, 16, 21, 2, 2)
    previous = torch.ones_like(latent)
    anchored = anchor_wan_latent_prefix(latent, previous, 7, 1.0)
    assert torch.equal(anchored[..., :7, :, :], previous[..., -7:, :, :])
    assert torch.equal(anchored[..., 7:, :, :], latent[..., 7:, :, :])


def test_wan_decoded_overlap_is_crossfaded_and_resampled():
    first = np.zeros((81, 4, 4, 3), dtype=np.uint8)
    second = np.full((81, 4, 4, 3), 255, dtype=np.uint8)
    composed, report = compose_wan_decoded_windows([first, second], 25)
    assert composed.shape == (137, 4, 4, 3)
    assert report["seam_count"] == 1
    resized = resample_wan_frames(composed, target_frames=720, height=8, width=8)
    assert resized.shape == (720, 8, 8, 3)
