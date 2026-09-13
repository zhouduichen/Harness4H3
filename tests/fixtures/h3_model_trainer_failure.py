from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    Path(args.result).write_text(
        json.dumps(
            {
                "status": "failed",
                "failure_type": "invalid_training_config",
                "message": "the trainer rejected the bounded configuration",
                "metrics": {"real_worker": True},
            }
        ),
        encoding="utf-8",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
