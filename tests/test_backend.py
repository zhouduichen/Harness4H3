from __future__ import annotations

from harness4h3.model.minimax_h3 import MiniMaxH3Adapter


def test_adapter_submits_polls_and_downloads(fake_comfyui, tmp_path, workflow):
    adapter = MiniMaxH3Adapter(fake_comfyui.url, poll_interval_s=0, task_timeout_s=2)
    result = adapter.run(workflow, tmp_path)
    assert result.prompt_id == "prompt-1"
    assert result.artifacts[0].read_bytes() == b"fake-video"
    assert fake_comfyui.calls[0][0] == "/prompt"
    assert any(path.startswith("/history/prompt-1") for path, _ in fake_comfyui.calls)
    assert any(path.startswith("/view?") for path, _ in fake_comfyui.calls)

