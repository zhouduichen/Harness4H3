from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan.diffusers import DiffusersWanAdapter


def test_diffusers_runtime_profile_uses_sequence_batch_with_mixed_timesteps(monkeypatch):
    adapter = DiffusersWanAdapter()
    sample = {
        "hidden_states": torch.randn(1, 4, 1, 2, 2),
        "timestep": torch.tensor([500]),
        "encoder_hidden_states": torch.randn(1, 8, 6),
    }
    observed = []
    monkeypatch.setattr(adapter, "profile_sample", lambda model, seed=7: sample)
    monkeypatch.setattr(
        adapter,
        "forward_profile",
        lambda model, probe: observed.append(probe) or probe["hidden_states"],
    )

    output = adapter.runtime_profile(object(), seed=11)

    assert tuple(output.shape) == (2, 4, 1, 2, 2)
    assert len(observed) == 1
    assert observed[0]["hidden_states"].shape[0] == 2
    assert observed[0]["encoder_hidden_states"].shape[0] == 2
    assert observed[0]["timestep"].tolist() == [100, 900]
