#!/usr/bin/env python3
"""Run one real Wan artifact through the existing video evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness4h3.adapters.wan.contracts import CheckpointArtifact, ModelStructure
from harness4h3.evolution.config import CompressionConfig
from harness4h3.evolution.evaluator import WanVideoEvaluationConfig, WanVideoEvaluator
from harness4h3.profiler.report import ProfileReport


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", type=Path, help="complete real-CUDA profile; defaults to evaluator.execution_profile")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = CompressionConfig.from_path(config_path)
    if config.backend != "diffusers" or str(config.evaluator.get("backend", "")).strip() != "wan_video":
        raise ValueError("one-shot Wan evaluation requires backend=diffusers and evaluator.backend=wan_video")
    profile_path = args.profile or Path(str(config.evaluator.get("execution_profile", ""))).expanduser()
    if not str(profile_path).strip():
        raise ValueError("a real-CUDA execution_profile is required to identify the checkpoint")
    if not profile_path.is_absolute():
        profile_path = (config_path.parent / profile_path).resolve()
    report = ProfileReport.read_json(profile_path)
    execution = dict(report.execution)
    cases = execution.get("cases")
    if (
        execution.get("available") is not True
        or execution.get("offline_simulation") is not False
        or execution.get("partial") is not False
        or not isinstance(cases, list)
        or execution.get("expected_case_count") != len(cases)
        or execution.get("observed_case_count") != len(cases)
    ):
        raise ValueError("execution_profile must explicitly contain a complete real-CUDA measurement")
    if str(execution.get("checkpoint_sha256", "")) != str(report.model.get("checkpoint_sha256", "")):
        raise ValueError("execution_profile checkpoint and model records disagree")

    model = dict(report.model)
    structure = ModelStructure(
        architecture=str(model["architecture"]),
        block_count=int(model["block_count"]),
        hidden_size=int(model["hidden_size"]),
        ffn_width=int(model["ffn_width"]),
        parameter_count=int(model["parameter_count"]),
        checkpoint_sha256=str(model["checkpoint_sha256"]),
        metadata=dict(model.get("metadata") or {}),
    )
    artifact = CheckpointArtifact(
        path=str(config.checkpoint),
        sha256=structure.checkpoint_sha256,
        architecture=structure.architecture,
        structure=structure,
        metadata={"offline_simulation": False, "source_profile": str(profile_path)},
    )
    evaluator_config = WanVideoEvaluationConfig.from_mapping(
        config.evaluator,
        config.output_root / "real-evaluation",
    )
    evaluator = WanVideoEvaluator(evaluator_config, baseline=artifact)
    result = evaluator._evaluate_artifact(artifact)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
