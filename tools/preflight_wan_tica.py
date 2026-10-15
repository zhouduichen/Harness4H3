#!/usr/bin/env python3
"""Read-only Wan/TICA checkpoint and split preflight.

This command never loads Diffusers, Torch, a pipeline, or a GPU.  It checks
the immutable checkpoint shell and the Calibration/Validation/Test manifest
provenance so a GPU host can run the runtime FFN/hook probe separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harness4h3.adapters.wan.diffusers import DiffusersWanAdapter
from harness4h3.evolution.config import CompressionConfig
from harness4h3.evolution.tica_calibration import TICAEvaluationSplits
from harness4h3.evolution import tica_calibration as _tica_calibration


def _manifest(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for item in raw.get("specs", []):
        normalized = dict(item)
        for name in ("latent_path", "encoder_hidden_states_path"):
            if normalized.get(name) and not Path(str(normalized[name])).is_absolute():
                normalized[name] = str((path.parent / str(normalized[name])).resolve())
        specs.append(normalized)
    return _tica_calibration.TICACalibrationManifest.from_dict({**raw, "specs": specs})


def preflight(config_path: Path) -> dict[str, Any]:
    config = CompressionConfig.from_path(config_path)
    if config.backend != "diffusers":
        raise ValueError("Wan TICA preflight requires backend=diffusers")
    if not config.tica_calibration_manifest or not config.tica_validation_manifest or not config.tica_test_manifest:
        raise ValueError("Wan TICA preflight requires all three split manifests")
    calibration = _manifest(config.tica_calibration_manifest)
    validation = _manifest(config.tica_validation_manifest)
    test = _manifest(config.tica_test_manifest)
    splits = TICAEvaluationSplits(calibration=calibration, validation=validation, test=test)
    for item in (calibration, validation, test):
        item.verify_cached_sources()
    checkpoint = DiffusersWanAdapter.preflight_checkpoint(config.checkpoint)
    return {
        "status": "ready_for_runtime_probe",
        "loads_model": False,
        "starts_gpu": False,
        "phase": config.phase,
        "checkpoint": checkpoint,
        "splits": {
            "calibration": calibration.to_dict(),
            "validation": validation.to_dict(),
            "test": test.to_dict(),
            "digest": splits.digest(),
        },
        "runtime_probe": {
            "required": True,
            "checks": ["all FFN paths", "activation hook output width", "global channel alignment", "loaded checkpoint digest"],
            "status": "not_run",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, dest="compression_config")
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = preflight(Path(args.compression_config).resolve())
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["output"] = str(output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=not args.json, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
