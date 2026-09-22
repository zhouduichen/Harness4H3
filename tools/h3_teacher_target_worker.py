#!/usr/bin/env python3
"""Run one real distributed H3 teacher forward and persist Student targets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
try:
    from h3_real_train_worker import (
        _construct_model,
        _init_distributed,
        _import_comfy,
        _install_termination_handlers,
        _make_batch,
        _prepare_cache,
        _wrap_fsdp,
    )
except ModuleNotFoundError:
    from tools.h3_real_train_worker import (
        _construct_model,
        _init_distributed,
        _import_comfy,
        _install_termination_handlers,
        _make_batch,
        _prepare_cache,
        _wrap_fsdp,
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _construct_teacher_model(
    api,
    checkpoint: Path,
    rank: int,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Use the shared H3 constructor with rank-0 checkpoint loading."""

    return _construct_model(
        api,
        Path(checkpoint).resolve(),
        device or torch.device("cuda", int(rank)),
        load_checkpoint=(int(rank) == 0),
    )


def _run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    rank = 0
    initialized = False
    try:
        _install_termination_handlers()
        rank, world_size, local_rank = _init_distributed(float(args.distributed_timeout_s))
        initialized = True
        if world_size != int(args.world_size):
            raise RuntimeError("teacher world size mismatch: expected %d, got %d" % (args.world_size, world_size))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        api = _import_comfy(Path(args.comfyui_root).resolve())
        api["comfy_model_management"].in_training = True
        checkpoint = Path(args.checkpoint).resolve()
        output_dir = Path(args.output).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("teacher checkpoint does not exist: %s" % checkpoint)
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in output_dir.glob("*.pt"):
                path.unlink()
        dist.barrier()
        item = _prepare_cache(Path(args.cache_dir).resolve(), int(args.seed))
        model = _construct_teacher_model(api, checkpoint, rank)
        fsdp_model = _wrap_fsdp(model, api, device, train_heads=False, sync_module_states=True)
        fsdp_model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        batch = _make_batch(item, device, float(args.sigma), int(args.seed), api)
        with torch.inference_mode():
            (video, audio), timestep, context, layout, _target_video, _target_audio = batch
            raw_video, _raw_audio = fsdp_model(
                [video, audio],
                timestep,
                context,
                transformer_options={},
                minimax_payload={"layout": layout, "audio_scale": 1.0},
            )
        payload = {
            "latent": batch[0][0].detach().to(device="cpu", dtype=torch.bfloat16),
            "conditioning": batch[2].detach().to(device="cpu", dtype=torch.bfloat16),
            "timestep": (batch[1] / 1000.0).detach().to(device="cpu", dtype=torch.float32),
            # Student learns clean-minus-noise velocity; native H3 is opposite.
            "target": (-raw_video).detach().to(device="cpu", dtype=torch.bfloat16),
            "teacher_world_size": world_size,
            "teacher_checkpoint": str(checkpoint),
            "sigma": float(args.sigma),
        }
        dist.barrier()
        if rank == 0:
            target_path = output_dir / "00000000.pt"
            temporary = target_path.with_suffix(target_path.suffix + ".part")
            torch.save(payload, temporary)
            temporary.replace(target_path)
            _write_json(
                Path(args.result),
                {
                    "status": "success",
                    "target_path": str(target_path),
                    "world_size": world_size,
                    "peak_memory_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
                    "wall_time_s": time.monotonic() - started,
                    "offline_simulation": False,
                },
            )
        dist.barrier()
        del fsdp_model, model
        torch.cuda.empty_cache()
        return 0
    except BaseException as exc:
        if rank == 0:
            try:
                _write_json(
                    Path(args.result),
                    {
                        "status": "failed",
                        "failure_code": "teacher_target_failed",
                        "message": str(exc),
                        "wall_time_s": time.monotonic() - started,
                        "offline_simulation": False,
                    },
                )
            except OSError:
                pass
        raise
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--comfyui-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--distributed-timeout-s", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.world_size < 2 or args.world_size > 4:
        parser.error("--world-size must be between 2 and 4")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
