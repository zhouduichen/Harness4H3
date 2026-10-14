#!/usr/bin/env python3
"""Summarize comparable evolutionary-compression runs.

This tool is intentionally read-only.  It does not infer missing metrics and
does not treat a rejected candidate as the active model.  Use it after the
four protocol runs described in ``docs/evolutionary-compression.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _round_dirs(root: Path) -> list[Path]:
    rounds = []
    for path in (root / "rounds").glob("R*"):
        try:
            int(path.name[1:])
        except ValueError:
            continue
        if path.is_dir():
            rounds.append(path)
    return sorted(rounds, key=lambda path: int(path.name[1:]))


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _metric(evaluation: Mapping[str, Any], name: str) -> float | None:
    hardware = evaluation.get("hardware_metrics")
    if not isinstance(hardware, Mapping):
        hardware = evaluation
    aliases = {
        "inference_wall_time_s_mean": ("inference_wall_time_s_mean", "latency_s"),
        "latency_s": ("latency_s", "inference_wall_time_s_mean"),
        "peak_memory_gb_max": ("peak_memory_gb_max", "peak_memory_gb"),
        "peak_memory_gb": ("peak_memory_gb", "peak_memory_gb_max"),
    }
    for candidate in aliases.get(name, (name,)):
        value = _number(hardware.get(candidate))
        if value is not None:
            return value
    return None


def _positive_metric(evaluation: Mapping[str, Any], name: str) -> float | None:
    value = _metric(evaluation, name)
    return value if value is not None and value > 0.0 else None


def _metrics_available(metrics: Mapping[str, Any] | None) -> bool:
    if not isinstance(metrics, Mapping):
        return False
    if _number(metrics.get("quality_score")) is not None:
        return True
    hardware = metrics.get("hardware_metrics")
    if not isinstance(hardware, Mapping):
        hardware = metrics
    return any(
        (_number(hardware.get(name)) or 0.0) > 0.0
        for name in (
            "inference_wall_time_s_mean",
            "latency_s",
            "peak_memory_gb_max",
            "peak_memory_gb",
            "model_size_gb",
        )
    )


def _active_record(root: Path) -> tuple[str | None, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Return the authoritative Active pointer, candidate, and measurements.

    ``ModelStore`` writes ``{"id": ...}``.  The two legacy names remain read
    compatible for old artifacts, but no candidate is guessed when the pointer
    or its file is missing.
    """

    pointer = _read_json(root / "models" / "active.json")
    if not pointer:
        return None, None, None
    active_id = pointer.get("id") or pointer.get("active_model_id") or pointer.get("active")
    if not isinstance(active_id, str) or not active_id:
        return None, None, None
    candidate = _read_json(root / "models" / "candidates" / (active_id + ".json"))
    if not candidate:
        return active_id, None, None
    state = candidate.get("state")
    if not isinstance(state, Mapping):
        return active_id, candidate, None
    metrics = state.get("measured_metrics")
    metrics = dict(metrics) if isinstance(metrics, Mapping) else None
    return active_id, candidate, metrics


def _active_metrics(root: Path) -> Mapping[str, Any] | None:
    """Backward-compatible accessor for the measured Active model only."""

    _active_id, _candidate, metrics = _active_record(root)
    return metrics if _metrics_available(metrics) else None


def _baseline_metrics(evaluation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Extract only the baseline measured beside this candidate evaluation."""

    baseline_hardware = evaluation.get("baseline_hardware_metrics")
    if not isinstance(baseline_hardware, Mapping):
        baseline_hardware = None
    baseline_quality = _number(evaluation.get("baseline_quality_score"))
    if baseline_quality is None and baseline_hardware is None:
        return None
    result: dict[str, Any] = {}
    if baseline_quality is not None:
        result["quality_score"] = baseline_quality
    if baseline_hardware is not None:
        result["hardware_metrics"] = dict(baseline_hardware)
    baseline_quality_metrics = evaluation.get("baseline_quality_metrics")
    if isinstance(baseline_quality_metrics, Mapping):
        result["quality_metrics"] = dict(baseline_quality_metrics)
    return result if _metrics_available(result) else None


def _candidate_rank(record: Mapping[str, Any]) -> tuple[float, float, float, str]:
    evaluation = record.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    quality = _number(evaluation.get("quality_score"))
    search_score = _number(evaluation.get("search_score"))
    hardware = evaluation.get("hardware_metrics")
    latency = _number(hardware.get("inference_wall_time_s_mean")) if isinstance(hardware, Mapping) else None
    return (
        search_score if search_score is not None else -math.inf,
        quality if quality is not None else -math.inf,
        -(latency if latency is not None else math.inf),
        str(record.get("child_model_id") or ""),
    )


def summarize_run(name: str, root: Path) -> dict[str, Any]:
    rounds: list[Mapping[str, Any]] = []
    evaluations: list[Mapping[str, Any]] = []
    candidate_records: list[Mapping[str, Any]] = []
    recipe_digests: set[str] = set()
    source_revisions: set[str] = set()
    failure_count = 0
    promoted_rounds: list[str] = []

    for round_dir in _round_dirs(root):
        decision = _read_json(round_dir / "decision.json") or {}
        evaluation = _read_json(round_dir / "evaluation.json")
        nested = decision.get("evaluation")
        if isinstance(nested, Mapping):
            evaluation = dict(nested)
        status = str(decision.get("status", "unknown"))
        if status == "failed":
            failure_count += 1
        if status == "promoted":
            promoted_rounds.append(round_dir.name)
        if isinstance(evaluation, Mapping):
            evaluations.append(evaluation)
            candidate_records.append(
                {
                    "round_id": round_dir.name,
                    "status": status,
                    "child_model_id": decision.get("child_model_id"),
                    "evaluation": evaluation,
                }
            )
            digest = evaluation.get("evaluator_recipe_sha256")
            if isinstance(digest, str) and digest:
                recipe_digests.add(digest)
            revision = evaluation.get("source_revision")
            source_revisions.add(str(revision).strip() if revision is not None else "unknown")
        rounds.append(
            {
                "round_id": round_dir.name,
                "status": status,
                "algorithm_id": decision.get("algorithm_id"),
                "parent_model_id": decision.get("parent_model_id"),
                "child_model_id": decision.get("child_model_id"),
                "promotion": decision.get("promotion"),
            }
        )

    active_id, _active_candidate, active = _active_record(root)
    active_is_measured = _metrics_available(active)
    final = active if active_is_measured else None
    final_source = "active_model" if active_is_measured else "missing"
    final_source_round_id = None
    # A root/legacy Active record may contain provenance but no metrics.  The
    # only safe recovery is the baseline measured in a round that explicitly
    # names this model as its parent.  A child evaluation without that
    # association is never used as the final model.
    if not active_is_measured and active_id:
        for record in candidate_records:
            if record.get("status") not in {"promoted", "rejected"}:
                continue
            round_dir = root / "rounds" / str(record["round_id"])
            decision = _read_json(round_dir / "decision.json") or {}
            if decision.get("parent_model_id") != active_id:
                continue
            baseline = _baseline_metrics(record["evaluation"])
            if baseline is not None:
                final = baseline
                final_source = "associated_parent_baseline"
                final_source_round_id = record["round_id"]

    last_candidate = None
    for record in reversed(candidate_records):
        child_id = record.get("child_model_id")
        if isinstance(child_id, str) and child_id:
            last_candidate = record
            break
    historical_records = [
        record
        for record in candidate_records
        if isinstance(record.get("child_model_id"), str) and str(record["child_model_id"])
    ]
    historical_best = max(historical_records, key=_candidate_rank) if historical_records else None
    rejected_model_ids = [
        str(record["child_model_id"])
        for record in candidate_records
        if record.get("status") == "rejected"
        and isinstance(record.get("child_model_id"), str)
        and str(record["child_model_id"])
    ]
    candidate_case_counts = sorted(
        {
            int(_number(record["evaluation"].get("hardware_metrics", {}).get("case_count")))
            for record in candidate_records
            if isinstance(record.get("evaluation"), Mapping)
            and isinstance(record["evaluation"].get("hardware_metrics"), Mapping)
            and _number(record["evaluation"]["hardware_metrics"].get("case_count")) is not None
        }
    )
    gpu_cost_values = [
        value
        for evaluation in evaluations
        for value in [_metric(evaluation, "gpu_runtime_cost_s")]
        if value is not None and value > 0.0
    ]
    final_hardware = final.get("hardware_metrics") if isinstance(final, Mapping) else None
    if not isinstance(final_hardware, Mapping):
        final_hardware = final if isinstance(final, Mapping) else {}
    return {
        "name": name,
        "root": str(root),
        "round_count": len(rounds),
        "experiment_count": len(evaluations),
        "failure_count": failure_count,
        "promoted_rounds": promoted_rounds,
        "active_model_id": active_id,
        "active_model_metrics_available": active_is_measured,
        "last_candidate_model_id": last_candidate.get("child_model_id") if last_candidate else None,
        "last_candidate_status": last_candidate.get("status") if last_candidate else None,
        "historical_best_candidate_model_id": historical_best.get("child_model_id") if historical_best else None,
        "historical_best_candidate_status": historical_best.get("status") if historical_best else None,
        "rejected_model_ids": rejected_model_ids,
        "candidate_case_counts": candidate_case_counts,
        "final_source": final_source,
        "final_source_round_id": final_source_round_id,
        "evaluator_recipe_sha256": sorted(recipe_digests),
        "source_revision": sorted(source_revisions),
        "comparable": (
            len(recipe_digests) <= 1
            and len(source_revisions) == 1
            and "unknown" not in source_revisions
            and bool(evaluations)
            and final is not None
        ),
        "final": {
            "quality_score": _number(final.get("quality_score")) if isinstance(final, Mapping) else None,
            "quality_delta": _number(final.get("quality_delta")) if isinstance(final, Mapping) else None,
            "latency_s": _positive_metric(final_hardware, "inference_wall_time_s_mean") if isinstance(final_hardware, Mapping) else None,
            "peak_memory_gb": _positive_metric(final_hardware, "peak_memory_gb_max") if isinstance(final_hardware, Mapping) else None,
            "model_size_gb": _positive_metric(final_hardware, "model_size_gb") if isinstance(final_hardware, Mapping) else None,
            "parameter_count": _number(final_hardware.get("parameter_count")) if isinstance(final_hardware, Mapping) else None,
            "case_count": _number(final_hardware.get("case_count")) if isinstance(final_hardware, Mapping) else None,
        },
        "gpu_runtime_cost_s": sum(gpu_cost_values) if gpu_cost_values else None,
        "rounds": rounds,
    }


def compare_runs(specs: list[str], *, expected_cases: int | None = None) -> dict[str, Any]:
    if not specs:
        raise ValueError("at least one --run NAME=PATH is required")
    summaries = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--run must use NAME=PATH: %s" % spec)
        name, raw_root = spec.split("=", 1)
        if not name.strip() or not raw_root.strip():
            raise ValueError("--run must use a non-empty NAME=PATH: %s" % spec)
        summaries.append(summarize_run(name.strip(), Path(raw_root).expanduser().resolve()))

    recipes = {
        digest
        for item in summaries
        for digest in item["evaluator_recipe_sha256"]
    }
    warnings: list[str] = []
    if len(recipes) > 1:
        warnings.append("evaluator_recipe_sha256 differs across runs")
    revisions = {
        revision
        for item in summaries
        for revision in item["source_revision"]
    }
    if "unknown" in revisions or not revisions:
        warnings.append("source_revision missing from one or more runs")
    elif len(revisions) > 1:
        warnings.append("source_revision differs across runs")
    if expected_cases is not None:
        for item in summaries:
            case_counts = [item["final"].get("case_count")]
            if case_counts[0] is not None and int(case_counts[0]) != expected_cases:
                warnings.append("%s case_count=%s, expected %s" % (item["name"], case_counts[0], expected_cases))
            elif case_counts[0] is None:
                for candidate_count in item.get("candidate_case_counts", []):
                    if candidate_count != expected_cases:
                        warnings.append("%s candidate case_count=%s, expected %s" % (item["name"], candidate_count, expected_cases))
    for item in summaries:
        if item["final_source"] == "missing":
            warnings.append("%s active model metrics unavailable; final result is missing" % item["name"])
    return {
        "protocol_comparable": bool(summaries) and not warnings and all(item["comparable"] for item in summaries),
        "expected_cases": expected_cases,
        "warnings": warnings,
        "runs": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="run specification NAME=OUTPUT_ROOT; repeat for each baseline")
    parser.add_argument("--expected-cases", type=int, help="optional required case count in the final evaluation")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a compact text summary")
    args = parser.parse_args()
    try:
        result = compare_runs(args.run, expected_cases=args.expected_cases)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("protocol_comparable=%s warnings=%d" % (result["protocol_comparable"], len(result["warnings"])))
        for item in result["runs"]:
            final = item["final"]
            print(
                "%s rounds=%d promoted=%d failures=%d quality=%s latency_s=%s memory_gb=%s size_gb=%s gpu_cost_s=%s"
                % (
                    item["name"],
                    item["round_count"],
                    len(item["promoted_rounds"]),
                    item["failure_count"],
                    final["quality_score"],
                    final["latency_s"],
                    final["peak_memory_gb"],
                    final["model_size_gb"],
                    item["gpu_runtime_cost_s"],
                )
            )
        for warning in result["warnings"]:
            print("warning: " + warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
