#!/usr/bin/env python3
"""Evaluate one immutable Wan prompt shard on one dedicated GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wan_vllm_campaign as campaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True)
    args = parser.parse_args(argv)
    config = campaign.load_loop_config(args.config)
    config["evaluation_manifest"] = str(args.manifest.resolve())
    config["evaluation_parallel_devices"] = []
    config["adaptive_visual_retry_enabled"] = False
    config["device"] = str(args.device)
    config["max_cases"] = 1
    result = campaign._evaluate_checkpoint_strict_single(config, args.checkpoint.resolve(), args.output.resolve())
    print(json.dumps({key: result.get(key) for key in ("status", "valid", "promotable", "quality_score", "case_count")}, sort_keys=True))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
