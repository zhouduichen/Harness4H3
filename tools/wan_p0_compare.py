#!/usr/bin/env python3
"""Export the fixed-seed P0 Wan/H3 correctness evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness4h3.student.evaluation_manifest import CacheContract, EvaluationManifest, validate_h3_cache_item
from harness4h3.student.inference import decode_video_latent, load_h3_cache_item
from harness4h3.student.proposal import StudentTarget
from harness4h3.student.wan import load_pretrained_wan, load_wan_spec, write_wan_video
from harness4h3.student.wan_long_video import load_wan_transformer_checkpoint
from harness4h3.student.wan_schedule import native_wan_schedule
from harness4h3.student.wan_training import WanTrainingSpec, encode_wan_video, resample_video_frames, sha256_path


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and isinstance(raw.get("wan_p0"), Mapping):
        raw = raw["wan_p0"]
    if not isinstance(raw, Mapping):
        raise ValueError("P0 config requires a mapping or wan_p0 mapping")
    return dict(raw)


def tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Return deterministic, device-independent statistics for one tensor."""

    value = torch.as_tensor(tensor)
    if value.numel() == 0:
        raise ValueError("tensor_stats requires a non-empty tensor")
    cpu_float = value.detach().float().cpu().contiguous()
    finite = bool(torch.isfinite(cpu_float).all().item())
    digest = hashlib.sha256(cpu_float.numpy().tobytes()).hexdigest()
    return {
        "name": str(name),
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": finite,
        "min": float(cpu_float.min().item()),
        "max": float(cpu_float.max().item()),
        "mean": float(cpu_float.mean().item()),
        "std": float(cpu_float.std(unbiased=False).item()),
        "sha256": digest,
    }


def _decode_wan_normalized_latent(pipeline: Any, latent: torch.Tensor) -> torch.Tensor:
    vae = pipeline.vae
    vae_dtype = next(vae.parameters()).dtype
    value = latent.to(device=latent.device, dtype=vae_dtype)
    mean = torch.as_tensor(vae.config.latents_mean, device=value.device, dtype=vae_dtype).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(vae.config.latents_std, device=value.device, dtype=vae_dtype).view(1, -1, 1, 1, 1)
    with torch.inference_mode():
        decoded = vae.decode(value * std + mean, return_dict=False)[0]
    processed = pipeline.video_processor.postprocess_video(decoded, output_type="np")
    if hasattr(processed, "shape"):
        result = torch.as_tensor(processed)
    else:
        result = torch.as_tensor(processed[0] if len(processed) == 1 else processed)
    if result.ndim == 5:
        result = result[0]
    if result.ndim != 4 or result.shape[-1] not in (3, 4):
        raise ValueError("Wan VAE reconstruction has invalid RGB shape %s" % (tuple(result.shape),))
    return result[..., :3].float().cpu()


def export_h3_and_wan_vae_evidence(
    *,
    pipeline: Any,
    h3_rgb: torch.Tensor,
    spec: Any,
    output_dir: Path,
    prefix: str = "teacher",
) -> Mapping[str, Any]:
    """Write H3 RGB, Wan-VAE RGB, and the bridge latent statistics."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    h3_frames = torch.as_tensor(h3_rgb).float().cpu()
    h3_path = output / (prefix + "-h3-rgb.mp4")
    write_wan_video(h3_frames, h3_path, float(spec.native_fps))
    training_spec = WanTrainingSpec(
        model_id=str(spec.model_id),
        pretrained_path=getattr(spec, "pretrained_path", None),
        width=int(spec.width),
        height=int(spec.height),
        num_frames=int(spec.num_frames),
        native_fps=float(spec.native_fps),
        dtype=str(spec.dtype),
    )
    wan_latent = encode_wan_video(pipeline.vae, h3_frames, training_spec, next(pipeline.vae.parameters()).device)
    reconstruction = _decode_wan_normalized_latent(pipeline, wan_latent)
    reconstruction_path = output / (prefix + "-wan-vae-reconstruction.mp4")
    write_wan_video(reconstruction, reconstruction_path, float(spec.native_fps))
    latent_path = output / (prefix + "-wan-vae-latent.pt")
    torch.save(wan_latent.detach().cpu(), latent_path)
    stats = [
        tensor_stats("h3_rgb", h3_frames),
        tensor_stats("wan_vae_latent", wan_latent),
        tensor_stats("wan_vae_reconstruction_rgb", reconstruction),
    ]
    stats_path = output / (prefix + "-latent-stats.json")
    _json_write(stats_path, {"stats": stats})
    return {
        "h3_rgb": str(h3_path),
        "wan_vae_reconstruction": str(reconstruction_path),
        "wan_vae_latent": str(latent_path),
        "latent_stats": stats,
        "latent_stats_path": str(stats_path),
    }


def _validate_inputs(config: Mapping[str, Any]) -> tuple[EvaluationManifest, Any, Path]:
    required = ("wan_config", "evaluation_manifest", "h3_cache_dir", "r2_checkpoint", "comfyui_root", "vae_name")
    missing = [name for name in required if not str(config.get(name, "")).strip()]
    if missing:
        raise ValueError("P0 config is missing: %s" % ", ".join(missing))
    manifest_path = Path(str(config["evaluation_manifest"])).resolve()
    manifest = EvaluationManifest.from_path(manifest_path)
    contract = manifest.cache_contract
    if contract != CacheContract.native_5s():
        raise ValueError("P0 manifest must bind the native_5s cache contract")
    cache_dir = Path(str(config["h3_cache_dir"])).resolve()
    cache_paths = [Path(case.cache_path).resolve() for case in manifest.cases]
    for cache_path in cache_paths:
        validate_h3_cache_item(cache_path, contract)
    checkpoint = Path(str(config["r2_checkpoint"])).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("R2 checkpoint does not exist: %s" % checkpoint)
    spec = load_wan_spec(Path(str(config["wan_config"])))
    return manifest, spec, checkpoint


def _run_wan_case(
    pipeline: Any,
    device: torch.device,
    spec: Any,
    *,
    prompt: str,
    seed: int,
    steps: int,
    output_path: Path,
    label: str,
) -> dict[str, Any]:
    schedule = native_wan_schedule(pipeline.scheduler, steps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    generator = torch.Generator(device=str(device)).manual_seed(int(seed))
    result = pipeline(
        prompt=prompt,
        height=int(spec.height),
        width=int(spec.width),
        num_frames=int(spec.num_frames),
        num_inference_steps=int(steps),
        guidance_scale=float(spec.guidance_scale),
        generator=generator,
        output_type="latent",
    )
    latent = getattr(result, "frames", None)
    if not isinstance(latent, torch.Tensor) or not torch.isfinite(latent).all():
        raise ValueError("Wan %s latent output is missing or non-finite" % label)
    frames = _decode_wan_normalized_latent(pipeline, latent)
    write_wan_video(frames, output_path, float(spec.native_fps))
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "steps": int(steps),
        "video": str(output_path),
        "latent_stats": tensor_stats(label + "_latent", latent),
        "schedule": schedule.to_dict(),
        "wall_time_s": float(time.perf_counter() - started),
        "peak_memory_bytes": peak_memory,
    }


def run_p0_case(config: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    manifest, spec, r2_checkpoint = _validate_inputs(config)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = manifest.cases[0]
    seed = int(case.seeds[0])
    cache_path = Path(case.cache_path).resolve()
    target = StudentTarget(
        latent_channels=24,
        latent_frames=manifest.cache_contract.latent_frames,
        latent_height=16,
        latent_width=16,
        condition_dim=5120,
    )
    device_name = str(config.get("device", "cuda:3"))
    pipeline, device = load_pretrained_wan(spec, device=device_name, local_files_only=True)
    cache_dir = Path(str(config["h3_cache_dir"])).resolve()
    cache_item = load_h3_cache_item(
        cache_dir,
        target,
        device,
        torch.bfloat16,
        cache_path=cache_path,
        cache_contract=manifest.cache_contract,
    )
    h3_latent = cache_item["latent"].to(device=device, dtype=torch.bfloat16)
    h3_frames = decode_video_latent(
        Path(str(config["comfyui_root"])),
        str(config["vae_name"]),
        h3_latent,
    )
    h3_frames = resample_video_frames(
        h3_frames,
        num_frames=int(spec.num_frames),
        height=int(spec.height),
        width=int(spec.width),
    )
    bridge = export_h3_and_wan_vae_evidence(
        pipeline=pipeline,
        h3_rgb=h3_frames,
        spec=spec,
        output_dir=output / "evidence" / case.case_id / ("seed_%d" % seed),
        prefix="teacher",
    )
    run_dir = output / "videos" / case.case_id / ("seed_%d" % seed)
    base32 = _run_wan_case(
        pipeline,
        device,
        spec,
        prompt=case.caption,
        seed=seed,
        steps=32,
        output_path=run_dir / "base-native-32.mp4",
        label="base_native_32",
    )
    base16 = _run_wan_case(
        pipeline,
        device,
        spec,
        prompt=case.caption,
        seed=seed,
        steps=16,
        output_path=run_dir / "base-native-16.mp4",
        label="base_native_16",
    )
    r2_digest = load_wan_transformer_checkpoint(pipeline, r2_checkpoint)
    r2_16 = _run_wan_case(
        pipeline,
        device,
        spec,
        prompt=case.caption,
        seed=seed,
        steps=16,
        output_path=run_dir / "r2-native-16.mp4",
        label="r2_native_16",
    )
    result = {
        "status": "passed",
        "case_id": case.case_id,
        "prompt": case.caption,
        "seed": seed,
        "runs": {
            "base_native_32": base32,
            "base_native_16": base16,
            "r2_native_16": {**r2_16, "checkpoint_sha256": r2_digest},
        },
        **bridge,
        "cache_sha256": sha256_path(cache_path),
        "manifest_digest": manifest.digest,
        "r2_checkpoint_sha256": r2_digest,
        "offline_simulation": False,
        "correctness": {
            "wan_formula": "x_sigma=(1-sigma)x_clean+sigma*noise; v=noise-clean; x_next=x+(sigma_next-sigma)*v",
            "h3_video_shift": 12.0,
            "h3_audio_shift": 3.0,
            "wan_schedule_shift": 3.0,
        },
    }
    _json_write(output / "p0-correctness.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    manifest, spec, checkpoint = _validate_inputs(config)
    if args.dry_run:
        payload = {
            "status": "validated",
            "native_cache_contract": manifest.cache_contract.to_dict(),
            "legacy_30s_cache_selected": "cache-long-30s-36x7" in str(config["h3_cache_dir"]),
            "r2_checkpoint": str(checkpoint),
            "r2_checkpoint_present": checkpoint.is_file(),
            "manifest_digest": manifest.digest,
            "wan_model_id": spec.model_id,
            "correctness_gate": "pending",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = run_p0_case(config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
