from pathlib import Path


def test_campaign_service_owns_controller_launcher_lifecycle():
    source = Path("tools/remote-campaign-service.sh").read_text(encoding="utf-8")

    assert "CONTROLLER_LAUNCHER_PID_FILE" in source
    assert "pid_is_controller_launcher" in source
    assert "ensure_controller_launcher" in source
    assert "CONTROLLER_GPU_LEASE_FILE=\"$CAMPAIGN_ROOT/.controller-gpu-lease.json\"" in source
    assert "campaign_has_finished" in source
    assert "stop_controller_launcher" in source
    assert 'kill_process_tree "$pid" TERM' in source
    assert 'kill_process_tree "$pid" KILL' in source
    assert "refusing to stop PID $pid: command is not the configured Controller launcher" in source


def test_campaign_service_does_not_use_broad_process_kill_patterns():
    source = Path("tools/remote-campaign-service.sh").read_text(encoding="utf-8")

    assert "pkill" not in source
    assert "killall" not in source
    assert "kill -TERM \"$pid\"" not in source


def test_campaign_service_has_explicit_operator_pause_gate():
    source = Path("tools/remote-campaign-service.sh").read_text(encoding="utf-8")

    assert "PAUSE_FILE" in source
    assert 'ACTION" != "pause"' in source
    assert 'ACTION" != "resume"' in source
    assert 'campaign automation is operator-paused' in source
    assert 'campaign automation resumed; no campaign was started' in source


def test_campaign_service_dispatches_wan_watchdog_without_changing_default_mode():
    source = Path("tools/remote-campaign-service.sh").read_text(encoding="utf-8")

    assert 'CAMPAIGN_KIND="${REMOTE_CAMPAIGN_KIND:-overnight}"' in source
    assert 'if [ "$CAMPAIGN_KIND" = "wan_vllm" ]' in source
    assert "wan-vllm-campaign-watchdog.sh" in source
    assert "WAN_CAMPAIGN_RESTART_LIMIT" in source
    assert 'CAMPAIGN_KIND="${REMOTE_CAMPAIGN_KIND:-overnight}"' in source


def test_campaign_service_wan_stop_checks_watchdog_command_before_signaling():
    source = Path("tools/remote-campaign-service.sh").read_text(encoding="utf-8")

    assert 'refusing to stop PID $pid: command is not the configured Wan watchdog' in source
    assert 'kill -TERM -- "$pid"' in source
    assert '[[ "$command" == *"$WAN_WATCHDOG"* ]]' in source
    assert 'Wan campaign already has a durable final result' in source
    assert 'state="completed"' in source
