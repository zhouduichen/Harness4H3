#!/usr/bin/env python3
"""Authorize a bounded Controller campaign from immutable Wan/H3 evidence.

The release is deliberately separate from fixed-seed promotion: the current
16-step child may use the audited adaptive visual-retry contract, while the
legacy R2 checkpoint remains permanently ineligible for promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _passed(value: bool, name: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"name": name, "passed": bool(value)})


def build_release(
    *,
    correctness_path: Path,
    strict_path: Path,
    adaptive_path: Path,
    base32_path: Path,
) -> dict[str, Any]:
    correctness = _read(correctness_path)
    strict = _read(strict_path)
    adaptive = _read(adaptive_path)
    base32 = _read(base32_path)
    checks: list[dict[str, Any]] = []

    _passed(correctness.get("status") == "passed", "p0_correctness_passed", checks)
    _passed(correctness.get("offline_simulation") is False, "p0_real_evidence", checks)
    _passed(strict.get("status") == "success", "strict_evaluation_completed", checks)
    _passed(adaptive.get("status") == "success", "adaptive_evaluation_completed", checks)
    _passed(adaptive.get("offline_simulation") is False, "adaptive_real_evaluation", checks)
    _passed(adaptive.get("contract_mode") == "adaptive_visual_retry", "adaptive_contract_explicit", checks)
    _passed(adaptive.get("adaptive_retry_verified") is True, "adaptive_retry_verified", checks)
    _passed(adaptive.get("promotable") is True, "adaptive_visual_gate_passed", checks)
    _passed(adaptive.get("fixed_seed_promotable") is False, "strict_gate_remains_distinct", checks)
    _passed(base32.get("status") == "success" and base32.get("promotable") is True, "base32_reference_valid", checks)

    strict_digest = str(strict.get("student_checkpoint_sha256", ""))
    adaptive_digest = str(adaptive.get("student_checkpoint_sha256", ""))
    _passed(bool(strict_digest) and strict_digest == adaptive_digest, "strict_adaptive_same_checkpoint", checks)

    base_quality = float(base32.get("quality_score", 0.0))
    adaptive_quality = float(adaptive.get("quality_score", 0.0))
    base_hardware = base32.get("hardware_metrics") if isinstance(base32.get("hardware_metrics"), Mapping) else {}
    adaptive_hardware = adaptive.get("hardware_metrics") if isinstance(adaptive.get("hardware_metrics"), Mapping) else {}
    base_wall = float(base_hardware.get("generation_wall_time_s_mean", float("inf")))
    adaptive_wall = float(adaptive_hardware.get("generation_wall_time_s_mean", float("inf")))
    base_nfe = int(base_hardware.get("transformer_nfe_proxy", 0))
    adaptive_nfe = int(adaptive_hardware.get("transformer_nfe_proxy", 0))
    base_vram = float(base_hardware.get("peak_memory_gb_max", float("inf")))
    adaptive_vram = float(adaptive_hardware.get("peak_memory_gb_max", float("inf")))
    _passed(adaptive_quality >= base_quality, "adaptive_quality_not_below_base32", checks)
    _passed(adaptive_wall < base_wall, "adaptive_latency_below_base32", checks)
    _passed(adaptive_nfe < base_nfe, "adaptive_nfe_below_base32", checks)
    _passed(adaptive_vram <= base_vram + 1e-6, "adaptive_vram_not_above_base32", checks)

    passed = all(bool(item["passed"]) for item in checks)
    return {
        "status": "passed" if passed else "blocked",
        "controller_search_authorized": passed,
        "release_contract": "adaptive_visual_retry",
        "legacy_r2_promotion_frozen": True,
        "fixed_seed_promotion_authorized": False,
        "checks": checks,
        "checkpoint_sha256": adaptive_digest,
        "comparisons": {
            "base32_quality": base_quality,
            "adaptive_quality": adaptive_quality,
            "base32_wall_time_s_mean": base_wall,
            "adaptive_wall_time_s_mean": adaptive_wall,
            "base32_nfe": base_nfe,
            "adaptive_nfe": adaptive_nfe,
            "base32_peak_vram_gb": base_vram,
            "adaptive_peak_vram_gb": adaptive_vram,
        },
        "evidence": {
            "correctness": {"path": str(correctness_path.resolve()), "sha256": _sha256(correctness_path)},
            "strict": {"path": str(strict_path.resolve()), "sha256": _sha256(strict_path)},
            "adaptive": {"path": str(adaptive_path.resolve()), "sha256": _sha256(adaptive_path)},
            "base32": {"path": str(base32_path.resolve()), "sha256": _sha256(base32_path)},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--strict", required=True, type=Path)
    parser.add_argument("--adaptive", required=True, type=Path)
    parser.add_argument("--base32", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_release(
        correctness_path=args.correctness,
        strict_path=args.strict,
        adaptive_path=args.adaptive,
        base32_path=args.base32,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["controller_search_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
