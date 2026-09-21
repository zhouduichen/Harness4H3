from types import SimpleNamespace
import io
import time

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
    assert "done done" not in captured["argv"][-1]
    assert captured["argv"][-1].rstrip().endswith("done")
    assert sampler.summary()["samples"] == 2


def test_power_sampler_stream_uses_configured_ssh_port(monkeypatch):
    import harness4h3.remote.power as power_module

    class Process:
        def __init__(self):
            self.stdout = io.StringIO("")
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

    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return Process()

    monkeypatch.setattr(power_module.subprocess, "Popen", fake_popen)
    sampler = power_module.RemotePowerSampler(
        SimpleNamespace(config=SimpleNamespace(host="intern@222.29.98.132", ssh_port=30902)),
        interval_s=0.01,
    )
    sampler.start()
    sampler.stop()

    assert captured["argv"][:5] == ["ssh", "-T", "-p", "30902", "intern@222.29.98.132"]


def test_power_sampler_local_transport_does_not_open_nested_ssh(monkeypatch):
    from harness4h3.remote.power import RemotePowerSampler

    class LocalClient:
        is_local = True

        def run(self, *args, **kwargs):
            return SimpleNamespace(stdout="0,100,90\n1,120,91\n")

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("local power sampling must not open nested SSH")

    monkeypatch.setattr("harness4h3.remote.power.subprocess.Popen", unexpected_popen)
    sampler = RemotePowerSampler(LocalClient(), interval_s=0.01)
    sampler.start()
    time.sleep(0.02)
    sampler.stop()

    assert sampler.summary()["samples"] >= 1
    assert sampler.summary()["per_gpu"]["0"]["utilization_gpu_pct_avg"] == pytest.approx(90)


def test_power_sampler_summary_keeps_per_gpu_utilization_rows():
    from harness4h3.remote.power import RemotePowerSampler

    class Process:
        def __init__(self):
            self.stdout = io.StringIO(
                "1000.0,570\n"
                "gpu,1000.0,0,290,99\n"
                "gpu,1000.0,1,280,98\n"
                "1001.0,560\n"
            )
            self.stderr = io.StringIO("")

        def wait(self):
            return 0

    sampler = RemotePowerSampler(SimpleNamespace(config=SimpleNamespace(host="fake")), interval_s=0.01)
    sampler._process = Process()
    sampler._run_stream()

    summary = sampler.summary()
    assert summary["per_gpu"]["0"]["power_w_peak"] == pytest.approx(290)
    assert summary["per_gpu"]["1"]["utilization_gpu_pct_avg"] == pytest.approx(98)
    assert summary["per_gpu"]["0"]["power_mean_w"] == pytest.approx(290)
    assert summary["per_gpu"]["1"]["utilization_mean"] == pytest.approx(98)
    assert summary["per_gpu"]["1"]["sample_count"] == 1
    assert summary["per_gpu"]["1"]["lane_unknown"] is True


def test_power_sampler_labels_gpu_lanes_from_campaign_leases():
    from harness4h3.remote.power import RemotePowerSampler

    class Client:
        config = SimpleNamespace(resolved_campaign_root="/srv/harness/campaign")

        def read_json(self, path):
            return {
                "/srv/harness/campaign/.controller-gpu-lease.json": {
                    "allocated_gpus": [3],
                },
                "/srv/harness/campaign/.h3-worker-gpu-lease.json": {
                    "allocated_gpus": [1, 2],
                },
                "/srv/harness/campaign/.comfyui-gpu-lease.json": {
                    "gpu_index": 0,
                },
            }[path]

    sampler = RemotePowerSampler(Client(), interval_s=0.01)
    sampler._append_gpu_sample(1.0, 0, 290.0, 99.0)
    sampler._append_gpu_sample(1.0, 1, 280.0, 98.0)
    sampler._append_gpu_sample(1.0, 2, 270.0, 97.0)
    sampler._append_gpu_sample(1.0, 3, 110.0, 12.0)

    summary = sampler.summary()
    assert summary["per_gpu"]["0"]["lane"] == "comfyui"
    assert summary["per_gpu"]["1"]["lane"] == "worker"
    assert summary["per_gpu"]["2"]["lane"] == "worker"
    assert summary["per_gpu"]["3"]["lane"] == "controller"


def test_power_sampler_labels_secondary_comfyui_leases():
    from harness4h3.remote.power import RemotePowerSampler

    class Client:
        config = SimpleNamespace(resolved_campaign_root="/srv/harness/campaign")

        def read_json(self, path):
            return {
                "/srv/harness/campaign/.comfyui-gpu-lease-2.json": {
                    "gpu_index": 2,
                },
            }[path]

    sampler = RemotePowerSampler(Client(), interval_s=0.01)
    sampler._append_gpu_sample(1.0, 2, 285.0, 96.0)

    assert sampler.summary()["per_gpu"]["2"]["lane"] == "comfyui"
