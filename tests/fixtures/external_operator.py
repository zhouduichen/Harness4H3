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
    child_path = Path(request["artifacts_dir"]) / "child.safetensors"
    shutil.copyfile(source, child_path)
    state = dict(parent["state"])
    state.update(
        model_id=request["child_model_id"],
        parent_model_id=parent["id"],
        checkpoint_path=str(child_path),
        dtype="int4",
        quantization={"bits": 4, "scheme": "fixture"},
        provenance={"operator": request["operator"], "parent_model_id": parent["id"]},
    )
    metrics = dict(state.get("measured_metrics") or {})
    metrics["model_size_gb"] = metrics.get("model_size_gb", 1.0) * 0.5
    state["measured_metrics"] = metrics
    result = {
        "status": "success",
        "output_state": state,
        "cost": {"wall_time_s": 0.01, "gpu_hours": 0.0, "controller_calls": 0},
        "metrics": {"fixture": True},
    }
    Path(args.result).write_text(json.dumps(result), encoding="utf-8")
    print("created", child_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
