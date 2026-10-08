#!/usr/bin/env python3
"""Generate a real approximately-30-second Wan Student video."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness4h3.student.wan import load_pretrained_wan, load_wan_spec, write_wan_video
from harness4h3.student.wan_long_video import (
    WanLongVideoError,
    WanLongVideoSpec,
    generate_wan_long_video,
    load_wan_transformer_checkpoint,
    validate_wan_long_video,
)
from harness4h3.student.wan_training import sha256_path


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompt", required=True, action="append")
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--window-count", type=int, default=8)
    parser.add_argument("--overlap-latent-frames", type=int, default=7)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    spec = load_wan_spec(args.config)
    inference_steps = int(args.inference_steps or spec.num_inference_steps)
    contract = WanLongVideoSpec(
        window_frames=spec.num_frames,
        overlap_latent_frames=int(args.overlap_latent_frames),
        window_count=int(args.window_count),
        native_fps=spec.native_fps,
        output_fps=spec.output_fps,
        width=spec.width,
        height=spec.height,
        inference_steps=inference_steps,
        guidance_scale=spec.guidance_scale,
    )
    seeds = list(args.seed if args.seed is not None else [20261008])
    if not seeds:
        raise ValueError("at least one seed is required")
    pipeline, device = load_pretrained_wan(spec, device=args.device, local_files_only=bool(args.local_files_only))
    parent_digest = sha256_path(Path(spec.pretrained_path)) if spec.pretrained_path else None
    checkpoint_digest = None
    if args.checkpoint is not None:
        checkpoint_digest = load_wan_transformer_checkpoint(pipeline, args.checkpoint)
    cases = []
    for prompt_index, prompt in enumerate(args.prompt):
        for seed in seeds:
            case_dir = args.output.resolve() / ("prompt_%03d_seed_%d" % (prompt_index, int(seed)))
            video_path = case_dir / "wan_long.mp4"
            started = time.perf_counter()
            frames, generation = generate_wan_long_video(
                pipeline,
                spec,
                contract,
                prompt,
                seed=int(seed),
                device=device,
            )
            write_wan_video(frames, video_path, contract.output_fps)
            validity = validate_wan_long_video(video_path, contract)
            cases.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": str(prompt),
                    "seed": int(seed),
                    "video_path": str(video_path),
                    "generation_wall_time_s": float(time.perf_counter() - started),
                    "generation": generation,
                    "validity": validity,
                    "hardware": {
                        "device": str(device),
                        "pretrained": True,
                        "offline_simulation": False,
                        "long_video": True,
                        "continuation_verified": bool(generation.get("continuation_verified")),
                        "video_contract_digest": contract.digest,
                    },
                }
            )
    return {
        "status": "success",
        "pretrained": True,
        "offline_simulation": False,
        "long_video": True,
        "continuation_verified": all(bool(case["generation"].get("continuation_verified")) for case in cases),
        "lineage_root": spec.lineage_root,
        "model_id": spec.model_id,
        "pretrained_path": spec.pretrained_path,
        "parent_model_digest": parent_digest,
        "student_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "student_checkpoint_sha256": checkpoint_digest,
        "device": str(device),
        "contract": contract.to_dict(),
        "contract_digest": contract.digest,
        "case_count": len(cases),
        "cases": cases,
    }


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
        _write(args.result.resolve(), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (WanLongVideoError, OSError, RuntimeError, TypeError, ValueError) as exc:
        failure = {"status": "failed", "error_code": getattr(exc, "code", "wan_long_worker_failed"), "message": str(exc)}
        _write(args.result.resolve(), failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
