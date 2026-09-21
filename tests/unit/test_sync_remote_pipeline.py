from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_remote_pipeline_sync_uses_configured_nonstandard_ssh_port_and_allowlist():
    script = (ROOT / "tools" / "sync-remote-pipeline.sh").read_text(encoding="utf-8")

    assert 'REMOTE_PIPELINE_SSH_PORT:-30902' in script
    assert 'ssh_args=(-p "$SSH_PORT"' in script
    assert 'rsync -a --files-from="$manifest"' in script
    assert 'harness4h3/remote/power.py' in script
    assert 'harness4h3/controller/schemas.py' in script
    assert 'tools/remote-campaign-service.sh' in script
    assert 'tools/remote-validation-window.sh' in script
    assert 'tools/summarize_remote_validation.py' in script
    assert "Activation and service start remain separate" in script


def test_remote_handoff_waits_for_external_gpu_jobs_before_activation():
    script = (ROOT / "tools" / "remote-pipeline-handoff.sh").read_text(encoding="utf-8")

    assert "query-compute-apps=pid,process_name,used_memory" in script
    assert "while ! gpu_idle" in script
    assert "all GPUs passed handoff idle gate" in script
    assert "COMFY_MAX_IDLE_USED_MIB" in script
    assert "queue_running" in script
    assert "IDLE_STABILITY_SAMPLES" in script
    assert "idle stability reset" in script
    assert "PAUSE_FILE" in script
    assert "operator pause appeared" in script
    assert "leaving staged pipeline untouched" in script


def test_remote_activation_copies_and_validates_all_unattended_supervisors():
    script = (ROOT / "tools" / "activate-remote-pipeline.sh").read_text(encoding="utf-8")

    for relative in (
        "tools/remote-campaign-service.sh",
        "tools/remote-campaign-supervisor.sh",
        "tools/remote-validation-window.sh",
        "tools/summarize_remote_validation.py",
        "tools/remote-idle-autostart.sh",
        "tools/remote-pipeline-handoff.sh",
    ):
        assert f'"{relative}"' in script
        assert f'"$STAGE_ROOT/{relative}"' in script
        assert f'"$REPO_ROOT/{relative}"' in script

    assert '"$STAGE_ROOT/harness4h3/controller/provider.py"' in script
    assert '"$REPO_ROOT/harness4h3/controller/provider.py"' in script
    assert '"$STAGE_ROOT/harness4h3/controller/schemas.py"' in script
    assert '"$REPO_ROOT/harness4h3/controller/schemas.py"' in script


def test_on_demand_comfyui_launcher_is_lease_bound():
    launcher = (ROOT / "tools" / "comfyui-wait-launch.sh").read_text(encoding="utf-8")
    campaign = (ROOT / "research" / "experiments" / "remote_h3_closed_loop.py").read_text(encoding="utf-8")

    assert "COMFY_LEASE_FILE" in launcher
    assert "COMFY_LEASE_MAX_AGE_SECONDS" in launcher
    assert "lease_active" in launcher
    assert "run_overnight_controller.py" in launcher
    assert "remote-campaign-supervisor.sh" in launcher
    assert "stop_child" in launcher
    assert 'env["COMFY_LEASE_FILE"]' in campaign


def test_controller_launcher_shares_only_an_idle_primary_comfyui_process():
    launcher = (ROOT / "tools" / "controller-wait-launch.sh").read_text(encoding="utf-8")

    assert "comfy_idle_compute_pid_allowed" in launcher
    assert "comfy_cache_released" in launcher
    assert "comfy_process_pid" in launcher
    assert "for pid in $compute_pids" in launcher
    assert "shared primary ComfyUI became active" in launcher


def test_controller_launcher_prefers_idle_primary_comfyui_gpu_on_tp_tie():
    launcher = (ROOT / "tools" / "controller-wait-launch.sh").read_text(encoding="utf-8")

    assert 'preferred_gpu="$COMFY_GPU_INDEX"' in launcher
    assert 'preferred_gpu="$preferred_gpu"' in launcher
    assert "best_preferred" in launcher


def test_controller_launcher_does_not_cool_down_after_controlled_handoff():
    launcher = (ROOT / "tools" / "controller-wait-launch.sh").read_text(encoding="utf-8")

    assert "child_exit_was_controlled=1" in launcher
    assert 'child_exit_was_controlled=0' in launcher
    assert 'skipping tensor_parallel cooldown' in launcher
    assert 'CONTROLLER_API_STARTUP_TIMEOUT_SECONDS:-900' in launcher
