"""Stage-aware promotion policy for native Wan 32-to-16 evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def _visual_cases(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = result.get("visual_cases")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return [dict(item) for item in direct if isinstance(item, Mapping)]
    cases = []
    for item in result.get("cases", []):
        if not isinstance(item, Mapping):
            continue
        quality = item.get("quality") if isinstance(item.get("quality"), Mapping) else {}
        gate = quality.get("visual_quality_gate") if isinstance(quality.get("visual_quality_gate"), Mapping) else {}
        validity = item.get("validity") if isinstance(item.get("validity"), Mapping) else {}
        failures = list(gate.get("failures", []))
        cases.append(
            {
                "case_id": str(item.get("case_id", "")),
                "seed": int(item.get("seed", 0)),
                "passed": bool(gate.get("enabled", False)) and bool(gate.get("passed", False)) and bool(validity.get("valid", True)),
                "reason": failures[0] if failures else ("visual_gate_disabled" if not gate.get("enabled", False) else "visual_gate_failed"),
                "failures": failures,
            }
        )
    return cases


def append_failure_memory(path: Path, *, stage: str, failures: Sequence[Mapping[str, Any]]) -> None:
    if not failures:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(
                json.dumps(
                    {"timestamp": time.time(), "stage": str(stage), "failure": dict(failure)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def evaluate_native_promotion(
    result: Mapping[str, Any],
    *,
    stage: str,
    base_wan32_quality: float,
    frozen_r2_quality: float,
    record_valid: bool,
    scheduler_parity_ok: bool,
    expected_nfe: int = 192,
    failure_memory_path: Path | None = None,
    required_artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return an auditable, fail-closed F1/F2/F3 promotion decision."""

    stage = str(stage).strip()
    if stage not in {"F1", "F2", "F3"}:
        raise ValueError("native promotion stage must be F1, F2, or F3")
    cases = _visual_cases(result)
    failures: list[dict[str, Any]] = []
    for case in cases:
        if not bool(case.get("passed", False)):
            failures.append({"case_id": case.get("case_id", ""), "seed": case.get("seed", 0), "reason": case.get("reason", "visual_failed"), "failures": list(case.get("failures", []))})
    for artifact in required_artifacts:
        if not bool(artifact.get("present", False)):
            failures.append({"case_id": artifact.get("case_id", ""), "seed": artifact.get("seed", 0), "reason": artifact.get("reason", "required_artifact_missing")})
    all_visual_pass = bool(cases) and not failures
    quality = float(result.get("quality_score", result.get("quality", 0.0)))
    hardware = result.get("hardware_metrics") if isinstance(result.get("hardware_metrics"), Mapping) else {}
    nfe = int(result.get("nfe", hardware.get("inference_steps_total", hardware.get("transformer_nfe_proxy", 0))))
    pure_seconds = float(result.get("pure_seconds", hardware.get("inference_wall_time_s_mean", 0.0)))
    peak_vram_gib = float(result.get("peak_vram_gib", hardware.get("peak_memory_gb_max", 0.0)))
    quality_ok = quality >= float(base_wan32_quality) - 0.005
    r2_ok = quality > float(frozen_r2_quality)
    resource_ok = nfe == int(expected_nfe) and pure_seconds > 0.0 and peak_vram_gib > 0.0
    stage_ok = stage == "F3" and bool(record_valid) and bool(scheduler_parity_ok)
    promotion_allowed = bool(stage_ok and all_visual_pass and quality_ok and r2_ok and resource_ok)
    if failure_memory_path is not None:
        append_failure_memory(Path(failure_memory_path), stage=stage, failures=failures)
    return {
        "stage": stage,
        "quality": quality,
        "base_wan32_quality": float(base_wan32_quality),
        "frozen_r2_quality": float(frozen_r2_quality),
        "all_visual_pass": all_visual_pass,
        "visual_cases": cases,
        "failure_memory": failures,
        "record_valid": bool(record_valid),
        "scheduler_parity_ok": bool(scheduler_parity_ok),
        "nfe": nfe,
        "expected_nfe": int(expected_nfe),
        "pure_seconds": pure_seconds,
        "peak_vram_gib": peak_vram_gib,
        "quality_ok": quality_ok,
        "r2_ok": r2_ok,
        "resource_ok": resource_ok,
        "promotion_allowed": promotion_allowed,
        "controller_search_enabled": False,
    }


__all__ = ["append_failure_memory", "evaluate_native_promotion"]
