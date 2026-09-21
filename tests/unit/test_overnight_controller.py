from pathlib import Path
from types import SimpleNamespace

from tools import run_overnight_controller as launcher


def _args(**overrides):
    values = {
        "controller": "rulebased",
        "local_resources": False,
        "max_iterations": 8,
        "resource_poll_interval_s": 0.0,
        "max_restarts": 3,
        "restart_backoff_s": 0.25,
        "controller_remote_port": None,
        "controller_fallback_ports": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_supervisor_rebuilds_campaign_after_process_level_failure(monkeypatch, tmp_path):
    built = []
    sleeps = []

    class Events:
        def __init__(self):
            self.items = []

        def append(self, event_type, payload):
            self.items.append((event_type, dict(payload)))

    class Campaign:
        def __init__(self, index):
            self.events = Events()
            self.index = index

        def run_loop(self, **kwargs):
            if self.index == 0:
                raise ValueError("transient launcher fault")
            return "completed"

    def build(*args, **kwargs):
        campaign = Campaign(len(built))
        built.append(campaign)
        return campaign

    monkeypatch.setattr(launcher, "build_campaign_from_config", build)
    monkeypatch.setattr(launcher.time, "sleep", sleeps.append)

    result = launcher._run_supervised(_args(), Path("configs/test.yaml"), tmp_path)

    assert result == "completed"
    assert len(built) == 2
    assert built[0].events.items[0][0] == "launcher_exception"
    assert built[0].events.items[0][1]["error_type"] == "ValueError"
    assert sleeps == [1.0]


def test_supervisor_passes_explicit_controller_port_override(monkeypatch, tmp_path):
    captured = []
    controller_calls = []

    class Campaign:
        events = SimpleNamespace(append=lambda *args, **kwargs: None)

        def run_loop(self, **kwargs):
            return "completed"

    def build(*args, **kwargs):
        captured.append(kwargs)
        return Campaign()

    def build_controller(*args, **kwargs):
        controller_calls.append(kwargs)
        return SimpleNamespace(provider_name="vllm", model_name="test")

    monkeypatch.setattr(launcher, "build_campaign_from_config", build)
    monkeypatch.setattr(launcher, "build_controller_from_config", build_controller)

    result = launcher._run_supervised(
        _args(
            controller="vllm",
            controller_remote_port=8001,
            controller_fallback_ports="8000,8000",
        ),
        Path("configs/test.yaml"),
        tmp_path,
    )

    assert result == "completed"
    assert captured[0]["controller"].provider_name == "vllm"
    assert controller_calls[0]["remote_port"] == 8001
    assert captured[0]["controller"].fallback_remote_ports == (8000,)


def test_supervisor_reraises_after_restart_budget(monkeypatch, tmp_path):
    built = []

    class Campaign:
        events = SimpleNamespace(append=lambda *args, **kwargs: None)

        def run_loop(self, **kwargs):
            raise ValueError("persistent launcher fault")

    def build(*args, **kwargs):
        built.append(True)
        return Campaign()

    monkeypatch.setattr(launcher, "build_campaign_from_config", build)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)

    try:
        launcher._run_supervised(_args(max_restarts=2), Path("configs/test.yaml"), tmp_path)
    except ValueError as exc:
        assert str(exc) == "persistent launcher fault"
    else:
        raise AssertionError("supervisor should re-raise after the restart budget")
    assert len(built) == 2
