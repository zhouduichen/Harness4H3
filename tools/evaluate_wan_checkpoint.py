#!/usr/bin/env python3
"""Run an independent Wan video evaluation for two existing checkpoints.

This command deliberately does not mutate an EvolutionLoop, model store, or
Pareto front.  It reuses the trusted Wan adapter and WanVideoEvaluator so a
Dev-run candidate can be evaluated on the independent Test manifest without
silently creating a new compression round.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from harness4h3.adapters.wan import CheckpointArtifact, build_wan_adapter
from harness4h3.evolution import CompressionConfig, WanVideoEvaluationConfig, WanVideoEvaluator


def _artifact_from_path(adapter: Any, path: Path, *, device: str, dtype: str) -> CheckpointArtifact:
    """Inspect one existing checkpoint and return a content-addressed artifact."""

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    model = adapter.load_model(str(resolved), device=device, dtype=dtype)
    try:
        structure = adapter.inspect(model)
        return CheckpointArtifact(
            str(resolved),
            structure.checkpoint_sha256,
            structure.architecture,
            structure,
            {"offline_simulation": False, "independent_evaluation": True},
        )
    finally:
        del model
        gc.collect()


def evaluate_checkpoints(
    *,
    config_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = CompressionConfig.from_path(config_path)
    if config.backend != "diffusers":
        raise ValueError("independent Wan evaluation requires backend=diffusers")
    if str(config.evaluator.get("backend", "")).strip().lower() != "wan_video":
        raise ValueError("independent Wan evaluation requires evaluator.backend=wan_video")
    adapter = build_wan_adapter("diffusers", local_files_only=config.local_files_only)
    baseline = _artifact_from_path(adapter, baseline_path, device=config.device, dtype=config.dtype)
    candidate = _artifact_from_path(adapter, candidate_path, device=config.device, dtype=config.dtype)
    if baseline.sha256 == candidate.sha256:
        raise ValueError("baseline and candidate checkpoint content hashes are identical")
    evaluator_config = WanVideoEvaluationConfig.from_mapping(
        config.evaluator,
        Path(output_root).resolve() / "real-evaluation",
    )
    evaluator = WanVideoEvaluator(evaluator_config, baseline=baseline)
    measured = dict(evaluator(candidate, {}))
    result = {
        "status": "success",
        "evaluation_kind": "independent_test",
        "config_path": str(config_path.resolve()),
        "output_root": str(Path(output_root).resolve()),
        "source_revision": measured.get("source_revision", evaluator_config.source_revision),
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
        "evaluation": measured,
    }
    target = Path(output_root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / "independent-evaluation.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target / "independent-evaluation.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Wan compression config with the independent evaluator manifest")
    parser.add_argument("--baseline", required=True, type=Path, help="existing baseline checkpoint directory")
    parser.add_argument("--candidate", required=True, type=Path, help="existing candidate checkpoint directory")
    parser.add_argument("--output-root", required=True, type=Path, help="new output directory for independent evidence")
    parser.add_argument("--json", action="store_true", help="emit the complete result as JSON")
    args = parser.parse_args()
    result = evaluate_checkpoints(
        config_path=args.config,
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True) if args.json else json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
