from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.tica_calibration import TICACalibrationManifest, TICACalibrationSpec


def test_fake_tica_manifest_is_deterministic_and_cpu_only():
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline")
    manifest = TICACalibrationManifest.from_fake(
        hidden_size=8, sample_count=2, timesteps=(0.9, 0.5, 0.1), seed=17
    )
    first = adapter.tica_calibration_inputs(model, manifest)
    second = adapter.tica_calibration_inputs(model, manifest)
    assert first[0].digest() == second[0].digest()
    assert all(item.metadata["device"] == "cpu" for item in first)
    assert manifest.timesteps() == (0.9, 0.5, 0.1)


def test_real_manifest_requires_explicit_cached_tensors():
    spec = TICACalibrationSpec("s0", "prompt", 7, (1, 4, 8, 8), (0.9, 0.5))
    manifest = TICACalibrationManifest((spec,), provider_name="cached_real_wan")
    with pytest.raises(ValueError, match="adapter does not implement"):
        manifest.samples(model=None, adapter=object())


def test_fake_denoising_output_depends_on_timestep():
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=8)
    model = adapter.load_model("fake://baseline")
    manifest = TICACalibrationManifest.from_fake(hidden_size=8, sample_count=1)
    sample = adapter.tica_calibration_inputs(model, manifest)[0]
    high = adapter.forward_denoising(model, sample, 0.9)
    low = adapter.forward_denoising(model, sample, 0.1)
    assert not torch.equal(high, low)
