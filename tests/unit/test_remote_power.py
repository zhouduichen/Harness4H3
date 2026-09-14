from types import SimpleNamespace
import io

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


def test_power_sampler_stream_command_starts_and_collects_samples(monkeypatch):
    import harness4h3.remote.power as power_module

    class Process:
        def __init__(self):
            self.stdout = io.StringIO("1000.0,100\n1001.0,120\n")
            self.stderr = io.StringIO("")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    process = Process()
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return process

    monkeypatch.setattr(power_module.subprocess, "Popen", fake_popen)
    sampler = power_module.RemotePowerSampler(SimpleNamespace(config=SimpleNamespace(host="fake")), interval_s=0.01)
    sampler.start()
    sampler.stop()

    assert captured["argv"][0] == "ssh"
    assert "printf '%s,%s\\n'" in captured["argv"][-1]
    assert sampler.summary()["samples"] == 2
