import json
from pathlib import Path

from tools.summarize_remote_validation import summarize_manifest, write_report


def _manifest(tmp_path, *, telemetry=True, event_offset=0):
    evidence = tmp_path / "evidence"
    campaign = tmp_path / "campaign"
    evidence.mkdir()
    campaign.mkdir()
    telemetry_path = evidence / "gpu-telemetry.csv"
    if telemetry:
        telemetry_path.write_text(
            "timestamp,index,power_w,utilization_gpu_pct,memory_used_mib,memory_total_mib\n"
            "1.0,0,250,80,100,46000\n"
            "1.0,1,260,81,110,46000\n"
            "1.0,2,270,82,120,46000\n"
            "1.0,3,280,83,130,46000\n",
            encoding="utf-8",
        )
    events = campaign / "controller-events.jsonl"
    events.write_text(
        "old event\n" if event_offset else "",
        encoding="utf-8",
    )
    offset = events.stat().st_size
    events.open("a", encoding="utf-8").write(
        json.dumps({"event_type": "controller_plan_prefetch_reused", "event_id": "evt-1", "gpu_indices": [1, 2]}) + "\n"
        + json.dumps({"event_type": "round_policy_activated", "event_id": "evt-2"}) + "\n"
        + json.dumps({
            "event_type": "lane_allocation",
            "event_id": "evt-3",
            "lanes": [
                {"lane": "worker", "allocated_gpus": [1, 2]},
                {"lane": "controller", "allocated_gpus": [3]},
                {"lane": "comfyui", "gpu_index": 0},
            ],
        }) + "\n"
    )
    state = campaign / "campaign_state.json"
    state.write_text(
        json.dumps({"current_model_id": "M0002", "training_calls": 2, "pipeline": {"stage": "waiting", "iteration": 2}}),
        encoding="utf-8",
    )
    result = campaign / "overnight-result.json"
    result.write_text(json.dumps({"status": "max_iterations", "current_model_id": "M0002"}), encoding="utf-8")
    status = evidence / "status-timeline.tsv"
    status.write_text("1\trunning\n2\tpaused\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "evidence_root": str(evidence),
        "campaign_output": str(campaign),
        "event_log": str(events),
        "event_start_byte": offset if event_offset else 0,
        "state_file": str(state),
        "result_file": str(result),
        "telemetry_file": str(telemetry_path),
        "status_timeline": str(status),
        "expected_gpu_count": 4,
        "started_at_epoch": 1,
        "ended_at_epoch": 2,
        "terminal_reason": "campaign_terminal_result",
    }
    manifest_path = evidence / "window-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path


def test_summary_aggregates_four_gpu_measurements_and_key_events(tmp_path):
    manifest, _ = _manifest(tmp_path)

    report = summarize_manifest(manifest)

    assert report["evidence_status"] == "measured"
    assert report["telemetry"]["all_expected_gpus_sampled"] is True
    assert report["telemetry"]["per_gpu"]["0"]["power_avg_w"] == 250
    assert report["events"]["observed_key_event_types"] == ["controller_plan_prefetch_reused", "round_policy_activated"]
    assert report["events"]["lane_by_gpu"] == {
        "1": ["worker"],
        "2": ["worker"],
        "3": ["controller"],
        "0": ["comfyui"],
    }
    assert report["telemetry"]["per_gpu"]["1"]["observed_lanes"] == ["worker"]
    assert report["claims"]["four_gpu_full_power"] is False


def test_summary_filters_event_log_to_window_offset(tmp_path):
    manifest, _ = _manifest(tmp_path, event_offset=1)

    report = summarize_manifest(manifest)

    assert report["events"]["event_rows"] == 3
    assert "old" not in report["events"]["event_type_counts"]


def test_summary_marks_missing_telemetry_unverified(tmp_path):
    manifest, manifest_path = _manifest(tmp_path, telemetry=False)

    report = write_report(manifest_path)

    assert report["evidence_status"] == "unverified"
    assert (manifest_path.parent / "validation-report.json").exists()
    assert (manifest_path.parent / "validation-report.md").exists()
