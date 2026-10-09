#!/usr/bin/env python3
"""Audit native H3 cache latents by decoding the stored Teacher videos to RGB."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness4h3.student.evaluation_manifest import CacheContract, validate_h3_cache_item
from harness4h3.student.inference import decode_video_latent, load_h3_cache_item
from harness4h3.student.proposal import StudentTarget


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(value: torch.Tensor) -> dict[str, Any]:
    tensor = torch.as_tensor(value).detach().float().cpu()
    finite = bool(torch.isfinite(tensor).all())
    result: dict[str, Any] = {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "finite": finite,
    }
    if finite:
        result.update(
            {
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--comfyui-root", required=True, type=Path)
    parser.add_argument("--vae-name", default="minimax_h3_video_vae_fp16.safetensors")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir.resolve()
    device = torch.device(args.device)
    target = StudentTarget(latent_channels=24, latent_frames=24, latent_height=16, latent_width=16, condition_dim=5120)
    contract = CacheContract.native_5s()
    records: list[dict[str, Any]] = []
    paths = sorted(cache_dir.glob("*.pt"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no cache items under %s" % cache_dir)
    try:
        for path in paths:
            record: dict[str, Any] = {
                "cache": str(path),
                "cache_sha256": _sha256(path),
            }
            try:
                raw = validate_h3_cache_item(path, contract)
                item = load_h3_cache_item(
                    cache_dir,
                    target,
                    device,
                    torch.bfloat16,
                    cache_path=path,
                    cache_contract=contract,
                )
                latent = item["latent"].to(device=device, dtype=torch.bfloat16)
                record["latent"] = _stats(latent)
                frames = decode_video_latent(args.comfyui_root, args.vae_name, latent)
                record["rgb"] = _stats(frames)
                record["caption"] = str(raw.get("caption", ""))
                record["teacher_generated"] = bool(raw.get("teacher_generated", False))
                record["teacher_sampling_steps"] = int(raw.get("teacher_sampling_steps", 0))
                record["offline_simulation"] = bool(raw.get("offline_simulation", True))
                record["status"] = "passed" if record["latent"]["finite"] and record["rgb"]["finite"] else "failed_nonfinite"
            except Exception as exc:  # audit records every item and continues
                record["status"] = "failed"
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)
            records.append(record)
            try:
                from harness4h3.student import inference

                inference._VAE_CACHE.clear()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        try:
            from harness4h3.student import inference

            inference._VAE_CACHE.clear()
        except Exception:
            pass
    result = {
        "schema_version": 1,
        "status": "passed" if all(item.get("status") == "passed" for item in records) else "failed",
        "cache_contract": contract.to_dict(),
        "cache_dir": str(cache_dir),
        "records": records,
        "finite_count": sum(item.get("status") == "passed" for item in records),
        "item_count": len(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
