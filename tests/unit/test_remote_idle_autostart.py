from pathlib import Path


def test_idle_autostart_has_fail_closed_gpu_gate_and_project_only_start():
    source = Path("tools/remote-idle-autostart.sh").read_text(encoding="utf-8")

    assert "query-compute-apps=pid,process_name,used_memory" in source
    assert "query-gpu=index,memory.used,memory.total" in source
    assert "MIN_FREE_MIB" in source
    assert "IDLE_STABILITY_SAMPLES" in source
    assert "gpu_idle_stable" in source
    assert "if gpu_idle_stable" in source
    assert 'bash "$SERVICE" start' in source
    assert "remote-campaign-service.sh" in source
    assert "pkill" not in source
    assert "killall" not in source


def test_idle_autostart_stops_after_new_terminal_result():
    source = Path("tools/remote-idle-autostart.sh").read_text(encoding="utf-8")

    assert "overnight-result.json" in source
    assert "START_MARKER" in source
    assert "terminal_result_is_newer_than_start" in source
    assert "idle watcher stopped" in source


def test_idle_autostart_honors_operator_pause_without_starting_service():
    source = Path("tools/remote-idle-autostart.sh").read_text(encoding="utf-8")

    assert "PAUSE_FILE" in source
    assert "operator pause is active" in source
    assert "without starting project service" in source
