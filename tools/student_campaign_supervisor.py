#!/usr/bin/env python3
"""Remote-side detached supervisor for the autonomous Student campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness4h3.student.campaign import OllamaStudentProposalProvider, StudentCampaign
from harness4h3.student.compiler import StudentCompiler
from harness4h3.student.config import load_student_campaign_config
from harness4h3.student.remote import (
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
    provider = OllamaStudentProposalProvider(
        config.controller.model,
        config.target,
        base_url=config.controller.base_url,
        timeout_s=config.controller.timeout_s,
    )
    output_root = Path(config.remote_campaign_root)
    campaign = StudentCampaign(
        provider,
        StudentCompiler(config.target),
        RemoteStudentWorker(config, client),
        RemoteStudentEvaluator(config, client),
        goal=config.goal,
        target=config.target,
        output_root=output_root,
        experience_path=output_root / "experience.jsonl",
        max_failures=config.max_failures,
        retention_handler=RemoteStudentRetention(config, client).retain,
    )
    result_path = output_root / "campaign-result.json"
    try:
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
