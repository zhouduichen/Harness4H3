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
    return _number(hardware.get(name))


def _active_metrics(root: Path) -> Mapping[str, Any] | None:
    pointer = _read_json(root / "models" / "active.json")
    if not pointer:
        return None
    active_id = pointer.get("active_model_id", pointer.get("active"))
    if not isinstance(active_id, str) or not active_id:
        return None
    candidate = _read_json(root / "models" / "candidates" / (active_id + ".json"))
    if not candidate:
        return None
    state = candidate.get("state")
    if not isinstance(state, Mapping):
        return None
    metrics = state.get("measured_metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else None


def summarize_run(name: str, root: Path) -> dict[str, Any]:
    rounds: list[Mapping[str, Any]] = []
    evaluations: list[Mapping[str, Any]] = []
    recipe_digests: set[str] = set()
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
            digest = evaluation.get("evaluator_recipe_sha256")
            if isinstance(digest, str) and digest:
                recipe_digests.add(digest)
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

    active = _active_metrics(root)
    # The initial M0000 record intentionally contains only provenance.  Do not
    # let that empty baseline hide the latest measured rejected candidate.
    active_is_measured = isinstance(active, Mapping) and (
        "quality_score" in active or isinstance(active.get("hardware_metrics"), Mapping)
    )
    final = active if active_is_measured else (evaluations[-1] if evaluations else None)
    gpu_cost = sum(
        value
        for evaluation in evaluations
        for value in [_metric(evaluation, "gpu_runtime_cost_s")]
        if value is not None
    )
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
        "active_model_metrics_available": active_is_measured,
        "evaluator_recipe_sha256": sorted(recipe_digests),
        "comparable": len(recipe_digests) <= 1 and bool(evaluations),
        "final": {
            "quality_score": _number(final.get("quality_score")) if isinstance(final, Mapping) else None,
            "quality_delta": _number(final.get("quality_delta")) if isinstance(final, Mapping) else None,
            "latency_s": _number(final_hardware.get("inference_wall_time_s_mean")) if isinstance(final_hardware, Mapping) else None,
            "peak_memory_gb": _number(final_hardware.get("peak_memory_gb_max")) if isinstance(final_hardware, Mapping) else None,
            "model_size_gb": _number(final_hardware.get("model_size_gb")) if isinstance(final_hardware, Mapping) else None,
            "parameter_count": _number(final_hardware.get("parameter_count")) if isinstance(final_hardware, Mapping) else None,
            "case_count": _number(final_hardware.get("case_count")) if isinstance(final_hardware, Mapping) else None,
        },
        "gpu_runtime_cost_s": gpu_cost if evaluations else None,
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
    if expected_cases is not None:
        for item in summaries:
            case_counts = [item["final"].get("case_count")]
            if case_counts[0] is not None and int(case_counts[0]) != expected_cases:
                warnings.append("%s case_count=%s, expected %s" % (item["name"], case_counts[0], expected_cases))
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
