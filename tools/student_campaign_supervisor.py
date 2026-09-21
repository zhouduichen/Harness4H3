#!/usr/bin/env python3
"""Remote-side detached supervisor for the autonomous Student campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness4h3.student.campaign import StudentCampaign, build_student_proposal_provider
from harness4h3.student.compiler import StudentCompiler
from harness4h3.student.config import load_student_campaign_config
from harness4h3.student.evaluation_manifest import build_manifest
from harness4h3.student.remote import (
    RemoteStudentBaseline,
    RemoteStudentEvaluator,
    RemoteStudentRetention,
    RemoteStudentWorker,
)
from harness4h3.remote.ssh import LocalCommandClient


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one detached autonomous Student campaign")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_student_campaign_config(Path(args.config))
    client = LocalCommandClient(config.remote)
    provider = build_student_proposal_provider(
        config.controller.provider,
        config.controller.model,
        config.target,
        base_url=config.controller.base_url,
        timeout_s=config.controller.timeout_s,
    )
    output_root = Path(config.remote_campaign_root)
    result_path = output_root / "campaign-result.json"
    try:
        manifest_path = Path(config.evaluation_manifest)
        if not manifest_path.is_file():
            build_manifest(Path(config.h3_cache_dir), manifest_path)
        campaign_args = (
            provider,
            StudentCompiler(config.target),
            RemoteStudentWorker(config, client),
            RemoteStudentEvaluator(config, client),
        )
        campaign_kwargs = dict(
            goal=config.goal,
            target=config.target,
            output_root=output_root,
            experience_path=output_root / "experience.jsonl",
            max_failures=config.max_failures,
            min_rounds_before_success=config.min_rounds_before_success,
            retention_handler=RemoteStudentRetention(config, client).retain,
        )
        if config.quality_backend == "clip_temporal":
            teacher_baseline = RemoteStudentBaseline(config, client).run()
            campaign_kwargs.update(
                teacher_baseline=teacher_baseline,
                quality_policy={
                    "quality_floor_ratio": config.quality_floor_ratio,
                    "min_reward_delta": config.min_reward_delta,
                    "material_efficiency_gain": config.material_efficiency_gain,
                    "max_metric_regression": config.max_metric_regression,
                    "no_improvement_patience": config.no_improvement_patience,
                },
            )
        campaign = StudentCampaign(*campaign_args, **campaign_kwargs)
        result = campaign.run(max_rounds=args.max_rounds or config.max_rounds)
        exit_code = 0 if result.status == "success" else 1
        payload = result.to_dict()
    except Exception as exc:
        exit_code = 1
        payload = {"status": "supervisor_failed", "failure_code": "supervisor_failed", "message": str(exc), "rounds": []}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
