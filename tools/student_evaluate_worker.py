#!/usr/bin/env python3
"""Fixed Student checkpoint→latent sampling→H3 VAE→video evaluator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

try:
    from harness4h3.student.evaluator import StudentEvaluation, StudentEvaluator
    from harness4h3.student.inference import StudentGenerationError, generate_video
    from harness4h3.student.proposal import StudentProposal, StudentTarget
except ModuleNotFoundError:  # direct invocation from the repository root
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness4h3.student.evaluator import StudentEvaluation, StudentEvaluator
    from harness4h3.student.inference import StudentGenerationError, generate_video
    from harness4h3.student.proposal import StudentProposal, StudentTarget


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _failure(path: Path, code: str, message: str) -> StudentEvaluation:
    return StudentEvaluation(
        valid=False,
        promotable=False,
        failure_code=code,
        message=message,
        video_path=str(path),
        validity={},
        quality_metrics={},
        hardware={},
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate and evaluate one Student video")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--comfyui-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--vae-name", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--black-frame-ratio-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)
    output_dir = Path(args.output).resolve()
    result_path = Path(args.result).resolve()
    video_path = output_dir / "student-generation.mp4"
    started = time.perf_counter()
    try:
        manifest = json.loads((output_dir / "compile_manifest.json").read_text(encoding="utf-8"))
        proposal = StudentProposal.from_dict(manifest["proposal"])
        target = StudentTarget(**dict(manifest["target"]))
        generation = generate_video(
            proposal,
            Path(args.checkpoint),
            target,
            Path(args.comfyui_root),
            Path(args.cache_dir),
            args.vae_name,
            video_path,
            device=args.device,
            seed=args.seed,
        )
        training = {}
        training_path = output_dir / "training-result.json"
        if training_path.is_file():
            raw_training = json.loads(training_path.read_text(encoding="utf-8"))
            if isinstance(raw_training, Mapping):
                training = dict(raw_training)
        # This is deliberately labelled a structural proxy, not a semantic
        # video-quality model. It gives the campaign a reproducible quality
        # signal while keeping generation validity as the hard gate.
        evaluator = StudentEvaluator(black_frame_ratio_threshold=args.black_frame_ratio_threshold)
        hardware = {
            **generation,
            "generation_wall_time_s": time.perf_counter() - started,
            "training_peak_memory_gb": training.get("peak_memory_gb"),
            "quantization": training.get("quantization"),
        }
        preliminary = evaluator.evaluate(video_path, hardware=hardware)
        sample_mean = float(preliminary.validity.get("sample_mean", 0.0))
        black_ratio = float(preliminary.validity.get("black_frame_ratio", 1.0))
        structural_score = max(0.0, min(1.0, 0.6 * (1.0 - black_ratio) + 0.4 * min(sample_mean / 128.0, 1.0)))
        quality = {
            "score": structural_score,
            "score_type": "structural_proxy",
            "sample_mean": sample_mean,
            "black_frame_ratio": black_ratio,
        }
        evaluation = evaluator.evaluate(video_path, quality=quality, hardware=hardware)
    except StudentGenerationError as exc:
        evaluation = _failure(video_path, exc.code, exc.message)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        evaluation = _failure(video_path, "student_evaluator_failed", str(exc))
    _write(result_path, evaluation.to_dict())
    return 0 if evaluation.promotable else 1


if __name__ == "__main__":
    raise SystemExit(main())
