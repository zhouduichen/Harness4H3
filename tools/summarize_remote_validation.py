#!/usr/bin/env python3
"""Summarize one bounded remote validation window without copying model bytes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


KEY_EVENT_TYPES = {
    "controller_plan_prefetch_ready",
    "controller_plan_prefetch_reused",
    "controller_plan_prefetch_unavailable",
    "parallel_gpu_fill_ready",
    "parallel_gpu_fill_waiting",
    "parallel_gpu_fill_skipped",
    "round_policy_activated",
    "round_policy_unavailable",
    "comfyui_lease_reserved",
    "comfyui_cache_release_requested",
    "comfyui_cache_released",
    "comfyui_cache_release_failed",
    "worker_started",
    "worker_finished",
    "speculative_worker_started",
    "speculative_worker_finished",
    "evaluation_started",
    "evaluation_finished",
}
MAX_EVENT_ROWS = 2048
MAX_LANE_ROWS = 256


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _read_events(path: Path, offset: int) -> List[Mapping[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Mapping[str, Any]] = []
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, int(offset)))
            for raw in handle:
                if len(rows) >= MAX_EVENT_ROWS:
                    break
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(value, Mapping):
                    rows.append(dict(value))
    except OSError:
        return []
    return rows


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _telemetry(path: Path, expected_gpu_count: int) -> Dict[str, Any]:
    per_gpu: MutableMapping[str, Dict[str, Any]] = OrderedDict()
    total_rows = 0
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        index = int(str(row.get("index", "")).strip())
                    except (TypeError, ValueError):
                        continue
                    power = _number(row.get("power_w"))
                    utilization = _number(row.get("utilization_gpu_pct"))
                    memory_used = _number(row.get("memory_used_mib"))
                    memory_total = _number(row.get("memory_total_mib"))
                    if index < 0 or power is None or utilization is None or memory_used is None or memory_total is None:
                        continue
                    if power < 0 or utilization < 0 or memory_used < 0 or memory_total <= 0:
                        continue
                    total_rows += 1
                    item = per_gpu.setdefault(
                        str(index),
                        {
                            "samples": 0,
                            "power_sum_w": 0.0,
                            "power_peak_w": 0.0,
                            "utilization_sum_pct": 0.0,
                            "utilization_peak_pct": 0.0,
                            "memory_used_sum_mib": 0.0,
                            "memory_used_peak_mib": 0.0,
                            "memory_total_mib": memory_total,
                        },
                    )
                    item["samples"] += 1
                    item["power_sum_w"] += power
                    item["power_peak_w"] = max(item["power_peak_w"], power)
                    item["utilization_sum_pct"] += utilization
                    item["utilization_peak_pct"] = max(item["utilization_peak_pct"], utilization)
                    item["memory_used_sum_mib"] += memory_used
                    item["memory_used_peak_mib"] = max(item["memory_used_peak_mib"], memory_used)
        except OSError:
            pass
    for item in per_gpu.values():
        samples = max(1, int(item["samples"]))
        item["power_avg_w"] = item.pop("power_sum_w") / samples
        item["utilization_avg_pct"] = item.pop("utilization_sum_pct") / samples
        item["memory_used_avg_mib"] = item.pop("memory_used_sum_mib") / samples
        item["power_target_w"] = 300.0
        item["utilization_target_pct"] = 100.0
        # The report deliberately exposes measurements and targets separately.
        # It never turns a short sample into a sustained full-power claim.
        item["target_claim"] = "unverified"
    observed = sorted(int(index) for index in per_gpu)
    complete = total_rows > 0 and len(observed) >= int(expected_gpu_count) and all(
        per_gpu[str(index)]["samples"] > 0 for index in range(int(expected_gpu_count))
    )
    return {
        "sample_rows": total_rows,
        "observed_gpu_indices": observed,
        "expected_gpu_count": int(expected_gpu_count),
        "all_expected_gpus_sampled": bool(complete),
        "per_gpu": dict(per_gpu),
        "evidence_status": "measured" if complete else "unverified",
    }


def _gpu_indices(event: Mapping[str, Any]) -> List[int]:
    values: List[int] = []
    for key, raw in event.items():
        key_text = str(key).lower()
        if key_text not in {
            "allocated_gpus",
            "gpu_indices",
            "training_gpu_indices",
            "candidate_training_gpu_indices",
            "selected_gpus",
            "gpu_index",
        }:
            continue
        candidates = raw if isinstance(raw, (list, tuple)) else [raw]
        for candidate in candidates:
            try:
                index = int(candidate)
            except (TypeError, ValueError):
                continue
            if index >= 0 and index not in values:
                values.append(index)
    return sorted(values)


def _lane_rows(event: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract nested campaign lane snapshots without copying raw payloads."""

    rows: List[Dict[str, Any]] = []
    lanes = event.get("lanes")
    if isinstance(lanes, (list, tuple)):
        for lane in lanes:
            if not isinstance(lane, Mapping):
                continue
            indices = _gpu_indices(lane)
            name = str(lane.get("lane") or "unknown")[:64]
            if indices:
                rows.append({"lane": name, "gpu_indices": indices})
    lane_evidence = event.get("lane_evidence")
    if isinstance(lane_evidence, Mapping):
        for name, key in (
            ("evaluator", "evaluator_gpus"),
            ("worker", "worker_gpus"),
            ("controller", "controller_gpus"),
        ):
            values = lane_evidence.get(key)
            if isinstance(values, (list, tuple)):
                indices = _gpu_indices({key: values})
                if indices:
                    rows.append({"lane": name, "gpu_indices": indices})
    return rows


def _event_summary(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    selected: List[Dict[str, Any]] = []
    lanes: List[Dict[str, Any]] = []
    lane_by_gpu: MutableMapping[str, List[str]] = OrderedDict()
    for event in events:
        event_type = str(event.get("event_type") or event.get("type") or "unknown")
        counts[event_type] += 1
        indices = _gpu_indices(event)
        if indices and len(lanes) < MAX_LANE_ROWS:
            lanes.append({
                "event_type": event_type,
                "event_id": event.get("event_id"),
                "gpu_indices": indices,
            })
        for lane_row in _lane_rows(event):
            if len(lanes) < MAX_LANE_ROWS:
                lanes.append({
                    "event_type": event_type,
                    "event_id": event.get("event_id"),
                    "lane": lane_row["lane"],
                    "gpu_indices": list(lane_row["gpu_indices"]),
                })
            for index in lane_row["gpu_indices"]:
                labels = lane_by_gpu.setdefault(str(index), [])
                if lane_row["lane"] not in labels:
                    labels.append(lane_row["lane"])
        if event_type in KEY_EVENT_TYPES and len(selected) < MAX_LANE_ROWS:
            selected.append({
                "event_type": event_type,
                "event_id": event.get("event_id"),
                "created_at": event.get("created_at"),
                "gpu_indices": indices,
            })
    return {
        "event_rows": len(events),
        "event_type_counts": dict(counts),
        "key_events": selected,
        "lane_events": lanes,
        "lane_by_gpu": {key: values[:8] for key, values in lane_by_gpu.items()},
        "observed_key_event_types": [name for name in counts if name in KEY_EVENT_TYPES],
    }


def _status_summary(path: Path) -> Dict[str, Any]:
    states: List[str] = []
    if path.is_file():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t", 1)
                if len(fields) == 2 and fields[1].strip():
                    states.append(fields[1].strip())
        except OSError:
            pass
    return {
        "samples": len(states),
        "states": states[-64:],
        "final_state": states[-1] if states else None,
        "paused_or_not_running": bool(states) and states[-1] not in {"running", "stop_requested", "running_without_service_pid"},
    }


def summarize_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    evidence_root = Path(str(manifest.get("evidence_root") or "."))
    telemetry = _telemetry(Path(str(manifest.get("telemetry_file") or evidence_root / "gpu-telemetry.csv")), int(manifest.get("expected_gpu_count", 4)))
    events = _read_events(Path(str(manifest.get("event_log") or "")), int(manifest.get("event_start_byte", 0)))
    event_summary = _event_summary(events)
    for index, item in (telemetry.get("per_gpu") or {}).items():
        item["observed_lanes"] = list(event_summary.get("lane_by_gpu", {}).get(str(index), ()))
    state = _read_json(Path(str(manifest.get("state_file") or "")))
    result = _read_json(Path(str(manifest.get("result_file") or "")))
    status = _status_summary(Path(str(manifest.get("status_timeline") or evidence_root / "status-timeline.tsv")))
    terminal_result = {
        "exists": bool(result),
        "status": result.get("status"),
        "current_model_id": result.get("current_model_id"),
    }
    pipeline = state.get("pipeline") if isinstance(state.get("pipeline"), Mapping) else {}
    report = {
        "schema_version": 1,
        "evidence_status": "measured" if telemetry["evidence_status"] == "measured" and status["paused_or_not_running"] else "unverified",
        "window": {
            "started_at_epoch": manifest.get("started_at_epoch"),
            "ended_at_epoch": manifest.get("ended_at_epoch"),
            "terminal_reason": manifest.get("terminal_reason"),
            "max_iterations": manifest.get("max_iterations"),
            "max_runtime_s": manifest.get("max_runtime_s"),
        },
        "telemetry": telemetry,
        "events": event_summary,
        "status": status,
        "campaign_state": {
            "current_model_id": state.get("current_model_id"),
            "training_calls": state.get("training_calls"),
            "pipeline_stage": pipeline.get("stage"),
            "pipeline_iteration": pipeline.get("iteration"),
            "pipeline_waiting_on": state.get("pipeline_waiting_on"),
            "active_round_policy": bool(state.get("active_round_policy")),
        },
        "terminal_result": terminal_result,
        "claims": {
            "four_gpu_full_power": False,
            "four_gpu_full_utilization": False,
            "reason": "targets require sustained live measurements and are never inferred from a short window",
        },
        "evidence_files": {
            "manifest": str(Path(str(manifest.get("evidence_root") or evidence_root)) / "window-manifest.json"),
            "telemetry": str(manifest.get("telemetry_file") or ""),
            "event_log": str(manifest.get("event_log") or ""),
            "state": str(manifest.get("state_file") or ""),
        },
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    telemetry = report.get("telemetry", {})
    lines = [
        "# Remote validation window",
        "",
        "- evidence_status: `%s`" % report.get("evidence_status"),
        "- terminal_reason: `%s`" % report.get("window", {}).get("terminal_reason"),
        "- sample_rows: `%s`" % telemetry.get("sample_rows", 0),
        "- claims: 4-GPU/300 W/100%% = **not asserted**",
        "",
        "## Per-GPU measurements",
        "",
        "| GPU | samples | avg W | peak W | avg util | peak util | observed lanes | target claim |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for index, item in sorted((telemetry.get("per_gpu") or {}).items(), key=lambda pair: int(pair[0])):
        lines.append(
            "| %s | %s | %.1f | %.1f | %.1f%% | %.1f%% | %s | %s |"
            % (
                index,
                item.get("samples", 0),
                float(item.get("power_avg_w", 0.0)),
                float(item.get("power_peak_w", 0.0)),
                float(item.get("utilization_avg_pct", 0.0)),
                float(item.get("utilization_peak_pct", 0.0)),
                ",".join(item.get("observed_lanes", ())) or "unverified",
                item.get("target_claim", "unverified"),
            )
        )
    events = report.get("events", {})
    lines.extend([
        "",
        "## Key lifecycle events",
        "",
        ", ".join(events.get("observed_key_event_types", [])) or "none observed",
        "",
        "## Resource safety",
        "",
        "Final service state: `%s`; paused/not running: `%s`." % (
            report.get("status", {}).get("final_state"),
            report.get("status", {}).get("paused_or_not_running"),
        ),
    ])
    return "\n".join(lines) + "\n"


def write_report(manifest_path: Path) -> Dict[str, Any]:
    manifest = _read_json(manifest_path)
    report = summarize_manifest(manifest)
    root = Path(str(manifest.get("evidence_root") or manifest_path.parent))
    json_path = root / "validation-report.json"
    markdown_path = root / "validation-report.md"
    root.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    json_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_tmp.replace(json_path)
    md_tmp = markdown_path.with_name(markdown_path.name + ".tmp")
    md_tmp.write_text(render_markdown(report), encoding="utf-8")
    md_tmp.replace(markdown_path)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    report = write_report(args.manifest.resolve())
    print(json.dumps({"evidence_status": report["evidence_status"], "report": str(args.manifest.parent / "validation-report.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
