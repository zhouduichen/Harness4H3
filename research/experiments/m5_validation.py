from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from harness4h3.backends.comfyui import MiniMaxH3Adapter
from harness4h3.benchmark.h3 import H3BenchmarkRunner
from harness4h3.benchmark.validation import M5ValidationRunner
from harness4h3.config import load_config
from harness4h3.evaluator.evaluator import SubprocessEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.harness.loop import load_workflow
from harness4h3.harness.state import Task, load_tasks
from harness4h3.target.profile import load_target_profile


def _state(model_id: str, checkpoint: str, bits: int, size_gb: float) -> ModelState:
    return ModelState(
        model_id=model_id,
        parent_model_id=None if model_id == "M0000" else "M0000",
        checkpoint_path=checkpoint,
        architecture_name="MiniMax-H3",
        parameter_count=11_681_874_744,
        num_blocks=50,
        hidden_size=2688,
        num_attention_heads=32,
        ffn_width=10752,
        dtype="mixed",
        quantization={"bits": bits, "scheme": "convrot-int8" if bits == 8 else "nvfp4"},
        sampling_steps=20,
        components={"text_encoder": "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"},
        runtime_state={"backend": "comfyui", "m5_stage": "m5.5", "metrics_stale": False},
        measured_metrics={"model_size_gb": size_gb},
        provenance={"source": "remote Windows RTX 5080"},
    )


def _extra_seeds(base: Task) -> List[Task]:
    return [replace(base, id=f"{base.id}-seed-{seed}", seed=seed) for seed in (7, 123, 999)]


def main() -> int:
    parser = argparse.ArgumentParser(description="M5.5 reproducibility/generalization validation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target", default="configs/targets/rtx5080_example.yaml")
    parser.add_argument("--base-url", default=os.environ.get("COMFYUI_BASE_URL", "http://100.88.143.10:8188"))
    parser.add_argument("--output", default="var/m5-controlled/m5.5-validation.json")
    parser.add_argument("--benchmark-output", default="var/m5-controlled/m5.5-runs")
    parser.add_argument("--sanity-repetitions", type=int, default=2)
    parser.add_argument(
        "--splits",
        default="sanity,dev,heldout,multiseed",
        help="comma-separated validation groups; completed groups are retained in the output JSON",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / args.config)
    target = load_target_profile(root / args.target)
    tasks = load_tasks(config.runtime.tasks_path)
    by_split = {split: [task for task in tasks if task.split == split] for split in ("sanity", "dev", "heldout")}
    if not by_split["sanity"]:
        raise ValueError("task manifest has no sanity task")
    multi_seed = _extra_seeds(by_split["sanity"][0])
    parent = _state(
        "M0000",
        r"D:\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        8,
        20.970379616,
    )
    child = _state(
        "M0001",
        r"D:\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_nvfp4.safetensors",
        4,
        12.5286368,
    )
    runner = H3BenchmarkRunner(
        MiniMaxH3Adapter(args.base_url, request_timeout_s=30, poll_interval_s=2, task_timeout_s=1800),
        SubprocessEvaluator(config.evaluator.command, config.evaluator.timeout_s),
        load_workflow(config.workflow.template),
        config.workflow,
        root / args.benchmark_output,
        system_sample_interval_s=1.0,
    )
    validation = M5ValidationRunner(runner)
    attribution = {
        "primary_intervention": "quantization",
        "secondary_changes": [],
        "controlled_variables": [
            "sampling_steps", "seed", "prompt", "scheduler", "resolution", "cfg",
            "vae", "text_encoder", "lora", "cache_reset_before_each_run",
        ],
        "rationale": "Size/latency are the measured bottlenecks; hold the generation recipe fixed before changing optimization layers.",
    }
    output = root / args.output
    results: Dict[str, Any] = {}
    if output.exists():
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
            if previous.get("target_profile_id") == target.id and isinstance(previous.get("results"), dict):
                results.update(previous["results"])
        except (OSError, ValueError, TypeError):
            pass
    groups = (
        ("sanity", by_split["sanity"], args.sanity_repetitions),
        ("dev", by_split["dev"], 1),
        ("heldout", by_split["heldout"], 1),
        ("multiseed", multi_seed, 1),
    )
    requested = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not requested:
        raise ValueError("--splits must name at least one validation group")
    known = {item[0] for item in groups}
    unknown = [label for label in requested if label not in known]
    if unknown:
        raise ValueError("unknown validation split(s): " + ", ".join(unknown))

    def persist() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps({"target_profile_id": target.id, "results": results}, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)

    for label, selected, repetitions in groups:
        if label not in requested:
            continue
        if not selected:
            continue
        result = validation.run(
            parent,
            child,
            selected,
            label=label,
            repetitions=repetitions,
            target=target,
            operator_attribution=attribution,
            black_frame_rate_threshold=0.0,
        )
        results[label] = result.to_dict()
        persist()
        print(json.dumps({"label": label, "validated": result.validated, "pairwise": result.pairwise}, ensure_ascii=False))
    persist()
    print("RESULT_PATH", output.resolve())
    return 0 if all(results[label]["validated"] for label in requested if label in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
