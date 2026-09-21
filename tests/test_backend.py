from __future__ import annotations

import pytest

from harness4h3.backends.comfyui import BackendError
from harness4h3.model.minimax_h3 import MiniMaxH3Adapter


def test_adapter_submits_polls_and_downloads(fake_comfyui, tmp_path, workflow):
    adapter = MiniMaxH3Adapter(fake_comfyui.url, poll_interval_s=0, task_timeout_s=2)
    result = adapter.run(workflow, tmp_path)
    assert result.prompt_id == "prompt-1"
    assert result.artifacts[0].read_bytes() == b"fake-video"
    assert fake_comfyui.calls[0][0] == "/prompt"
    assert any(path.startswith("/history/prompt-1") for path, _ in fake_comfyui.calls)
    assert any(path.startswith("/view?") for path, _ in fake_comfyui.calls)


def test_adapter_timeout_cancels_exact_prompt(monkeypatch):
    adapter = MiniMaxH3Adapter("http://comfy", poll_interval_s=0, task_timeout_s=0.001)
    calls = []

    def fake_json(path, payload=None):
        calls.append((path, payload))
        return {}

    monkeypatch.setattr(adapter, "_json", fake_json)

    with pytest.raises(BackendError, match="prompt_cancelled=true"):
        adapter.wait("prompt-stuck")

    assert ("/interrupt", {"prompt_id": "prompt-stuck"}) in calls
