from types import SimpleNamespace

import pytest


def test_power_sampler_integrates_sum_of_gpu_power():
    from harness4h3.remote.power import integrate_power_samples

    assert integrate_power_samples([(0.0, 100.0), (1.0, 120.0), (2.0, 80.0)]) == pytest.approx(220.0)


def test_power_sampler_parses_all_gpu_values():
    from harness4h3.remote.power import RemotePowerSampler

    assert RemotePowerSampler._parse("100.0\n120.5\n") == pytest.approx(220.5)
    assert RemotePowerSampler._parse("N/A\n") is None


def test_power_sampler_keeps_energy_unknown_with_one_sample():
    from harness4h3.remote.power import RemotePowerSampler

    class Client:
        def run(self, *args, **kwargs):
            return SimpleNamespace(stdout="100\n")

    sampler = RemotePowerSampler(Client(), interval_s=0.01)
    sampler._sample()
    assert sampler.summary()["energy_j"] is None
