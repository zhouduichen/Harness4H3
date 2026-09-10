from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    parent = request["parent"]
    source = Path(parent["checkpoint_path"])
    child = Path(request["artifacts_dir"]) / "trainer-child.safetensors"
    shutil.copy2(source, child)
    state = dict(parent["state"])
    state.update(
        model_id=request["child_model_id"],
        parent_model_id=parent["id"],
        checkpoint_path=str(child),
        parameter_count=int(state.get("parameter_count") or 1) // 2,
        provenance={"fixture_trainer": True},
    )
    metrics = dict(state.get("measured_metrics") or {})
    metrics["model_size_gb"] = float(metrics.get("model_size_gb", 1.0)) * 0.5
    state["measured_metrics"] = metrics
    Path(args.result).write_text(
        json.dumps(
            {
                "status": "success",
                "output_state": state,
                "cost": {"wall_time_s": 0.02, "gpu_hours": 0.01, "controller_calls": 0},
                "metrics": {"fixture_trainer": True},
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
