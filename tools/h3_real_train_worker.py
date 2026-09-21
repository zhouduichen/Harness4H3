#!/usr/bin/env python3
"""Authentic MiniMax-H3 recovery trainer using the ComfyUI model implementation."""

from __future__ import annotations

import argparse
from datetime import timedelta
import faulthandler
import fcntl
import gc
import hashlib
import json
import math
import os
import signal
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:  # pragma: no cover - older torch fallback
    def record(function):
        return function

try:
    from h3_real_support import (
        TRAINABLE_TENSOR_NAMES,
        make_smoke_cache,
        patch_safetensors_tensors,
        read_safetensors_header,
    )
except ModuleNotFoundError:
    from tools.h3_real_support import (
        TRAINABLE_TENSOR_NAMES,
        make_smoke_cache,
        patch_safetensors_tensors,
        read_safetensors_header,
    )


FAILURE_TYPES = frozenset(
    {
        "unsupported_training_operator",
        "invalid_training_config",
        "cache_corrupt",
        "nonfinite_loss",
        "zero_gradient",
        "training_oom",
        "checkpoint_corrupt",
        "parent_modified",
        "unchanged_child",
        "frozen_tensor_changed",
        "child_reload_failed",
        "device_unavailable",
        "training_interrupted",
        "distributed_failure",
    }
)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON object required: %s" % path)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_sha256(path: Path, cache_path: Optional[Path]) -> tuple[str, str]:
    """Use a prior full hash only when the immutable file stat still matches."""

    stat = path.stat()
    key = str(path.resolve())
    if cache_path is not None and cache_path.is_file():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            entry = raw.get(key) if isinstance(raw, Mapping) else None
            if (
                isinstance(entry, Mapping)
                and int(entry.get("size", -1)) == int(stat.st_size)
                and int(entry.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
                and isinstance(entry.get("sha256"), str)
                and len(entry["sha256"]) == 64
            ):
                return entry["sha256"], "cached_full_hash"
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    digest = _sha256_file(path)
    _record_sha256_cache(path, digest, cache_path)
    return digest, "full_hash"


def _record_sha256_cache(path: Path, digest: str, cache_path: Optional[Path]) -> None:
    """Record a digest using the file stat observed for that digest."""

    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    stat = path.stat()
    raw[str(path.resolve())] = {
        "sha256": str(digest),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    temporary.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(cache_path)


def validate_config(config: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    required = ("comfyui_root", "model_checkpoint", "cache_dir", "output_dir")
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError("missing worker config key(s): %s" % ", ".join(missing))
    root = Path(str(config["comfyui_root"])).resolve()
    parent = Path(str(config["model_checkpoint"])).resolve()
    cache = Path(str(config["cache_dir"])).resolve()
    output = Path(str(config["output_dir"])).resolve()
    request_artifacts = request.get("artifacts_dir")
    if request_artifacts is not None:
        if not isinstance(request_artifacts, str) or not request_artifacts.strip():
            raise ValueError("request artifacts_dir must be a non-empty path")
        # The external-operator contract gives each experiment an isolated
        # artifact directory.  Prefer it over the static smoke output root so
        # concurrent controller experiments cannot collide.
        output = Path(request_artifacts).resolve()
    if not root.is_dir():
        raise ValueError("ComfyUI root does not exist: %s" % root)
    if not parent.is_file() or parent.suffix.lower() != ".safetensors":
        raise ValueError("MiniMax-H3 parent safetensors does not exist: %s" % parent)
    request_parent = request.get("parent")
    request_checkpoint = request_parent.get("checkpoint_path") if isinstance(request_parent, Mapping) else None
    if request_checkpoint and Path(str(request_checkpoint)).resolve() != parent:
        raise ValueError("request parent does not match fixed worker parent")
    operator = str(request.get("operator", "recovery_finetune"))
    if operator not in {"recovery_finetune", "distill", "step_distill", "dmd2"}:
        raise ValueError("unsupported_training_operator: %s" % operator)
    source_parent_raw = config.get("source_parent_checkpoint")
    source_parent = None
    if source_parent_raw:
        source_parent = Path(str(source_parent_raw)).resolve()
    if operator == "distill":
        if source_parent is None:
            raise ValueError("distill requires source_parent_checkpoint")
        if not source_parent.is_file() or source_parent.suffix.lower() != ".safetensors":
            raise ValueError("distill source teacher checkpoint does not exist: %s" % source_parent)
        if source_parent == parent:
            raise ValueError("distill source_parent_checkpoint must differ from the student parent")
    world_size = int(config.get("world_size", 4))
    if world_size < 2 or world_size > 4:
        raise ValueError("real H3 distributed training requires world_size in [2, 4]")
    max_steps = int(config.get("max_steps", 1))
    if max_steps <= 0 or max_steps > 50:
        raise ValueError("real H3 max_steps must be between 1 and 50")
    operator_args = request.get("operator_args")
    if not isinstance(operator_args, Mapping):
        operator_args = {}
    source_steps = int(config.get("source_steps", 32))
    target_steps = int(operator_args.get("target_steps", config.get("target_steps", source_steps // 2)))
    if operator == "step_distill":
        if source_steps <= 0 or target_steps <= 0 or source_steps != 2 * target_steps:
            raise ValueError("step_distill requires source_steps=2*target_steps")
    generator_update_interval = int(operator_args.get("generator_update_interval", 2))
    if operator == "dmd2" and not 1 <= generator_update_interval <= max_steps:
        raise ValueError("dmd2.generator_update_interval must be in [1, max_steps]")
    dataset_fraction = float(operator_args.get("dataset_fraction", config.get("dataset_fraction", 1.0)))
    if operator == "distill":
        # This worker currently has one deterministic cached H3 sample, not a
        # dataset loader.  Refuse a smaller fraction instead of silently
        # claiming that only part of a dataset was consumed.
        if not math.isfinite(dataset_fraction) or abs(dataset_fraction - 1.0) > 1e-8:
            raise ValueError("distill requires dataset_fraction=1.0 for the configured cache")
    learning_rate = float(config.get("learning_rate", 1e-6))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    full_frozen_scan = config.get("full_frozen_scan", True)
    if not isinstance(full_frozen_scan, bool):
        raise ValueError("full_frozen_scan must be boolean")
    distributed_timeout_s = float(config.get("distributed_timeout_s", 180.0))
    if not math.isfinite(distributed_timeout_s) or distributed_timeout_s <= 0:
        raise ValueError("distributed_timeout_s must be finite and positive")
    parent_hash_timeout_s = float(config.get("parent_hash_timeout_s", 3600.0))
    if not math.isfinite(parent_hash_timeout_s) or parent_hash_timeout_s <= 0:
        raise ValueError("parent_hash_timeout_s must be finite and positive")
    parent_hash_cache = config.get("parent_hash_cache")
    if parent_hash_cache is not None and (not isinstance(parent_hash_cache, str) or not parent_hash_cache.strip()):
        raise ValueError("parent_hash_cache must be a non-empty path when configured")
    checkpoint_stage_dir = config.get("checkpoint_stage_dir")
    if checkpoint_stage_dir is not None and (
        not isinstance(checkpoint_stage_dir, str) or not checkpoint_stage_dir.strip()
    ):
        raise ValueError("checkpoint_stage_dir must be a non-empty path when configured")
    rank0_only_load = config.get("rank0_only_load", True)
    if not isinstance(rank0_only_load, bool):
        raise ValueError("rank0_only_load must be boolean")
    return {
        "comfyui_root": root,
        "model_checkpoint": parent,
        "cache_dir": cache,
        "output_dir": output,
        "world_size": world_size,
        "max_steps": max_steps,
        "operator": operator,
        "source_parent_checkpoint": source_parent,
        "source_steps": source_steps,
        "target_steps": target_steps,
        "generator_update_interval": generator_update_interval,
        "dataset_fraction": dataset_fraction,
        "learning_rate": learning_rate,
        "seed": int(config.get("seed", 20260913)),
        "sigma": float(config.get("sigma", 0.5)),
        "full_frozen_scan": full_frozen_scan,
        "distributed_timeout_s": distributed_timeout_s,
        "parent_hash_timeout_s": parent_hash_timeout_s,
        "parent_hash_cache": Path(parent_hash_cache).resolve() if parent_hash_cache else None,
        "checkpoint_stage_dir": Path(checkpoint_stage_dir).resolve() if checkpoint_stage_dir else None,
        "rank0_only_load": rank0_only_load,
    }


def _write_failure(result_path: Path, failure_type: str, message: str, wall_time_s: float = 0.0) -> None:
    if failure_type not in FAILURE_TYPES:
        failure_type = "invalid_training_config"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "failure_type": failure_type,
                "message": message,
                "cost": {"wall_time_s": float(wall_time_s), "gpu_hours": 0.0, "controller_calls": 0},
                "metrics": {"real_worker": True, "offline_simulation": False},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _classify_failure(message: str) -> str:
    """Map a worker exception to the stable failure vocabulary."""

    lowered = str(message).lower()
    if "unsupported_training_operator" in lowered:
        return "unsupported_training_operator"
    if "only tensors of floating point" in lowered or "loading state_dict" in lowered:
        return "checkpoint_corrupt"
    if "parent" in lowered and "hash" in lowered:
        return "parent_modified"
    if "gradient" in lowered:
        return "zero_gradient"
    if "loss" in lowered and "finite" in lowered:
        return "nonfinite_loss"
    if "frozen tensor" in lowered:
        return "frozen_tensor_changed"
    if "child" in lowered and ("reload" in lowered or "reloaded" in lowered):
        return "child_reload_failed"
    if "trainable tensor changed" in lowered or "child hash equals" in lowered:
        return "unchanged_child"
    if "checkpoint" in lowered:
        return "checkpoint_corrupt"
    if any(
        token in lowered
        for token in ("nccl", "process group", "collective", "distributed", "timed out", "timeout")
    ):
        return "distributed_failure"
    return "invalid_training_config"


def _init_distributed(timeout_s: float) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 2 or world_size > 4:
        raise ValueError("torchrun world size must be in [2, 4], got %d" % world_size)
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("%d CUDA devices are required for distributed H3 training" % world_size)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # The NCCL default can leave the surviving ranks blocked for many minutes
    # after one worker disappears.  A bounded timeout turns that into a
    # reportable worker failure and lets the parent Harness cleanly reject the
    # candidate instead of leaking three GPU processes.
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_s))
    return rank, world_size, local_rank


def _install_termination_handlers() -> None:
    """Convert operator interrupts into the normal cleanup path."""

    def _raise_keyboard_interrupt(signum, _frame):
        raise KeyboardInterrupt("received signal %d" % signum)

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def _exchange_parent_hash(
    parent: Path,
    child_model_id: str,
    rank: int,
    distributed_timeout_s: float,
    parent_hash_timeout_s: float,
    parent_hash_cache: Optional[Path] = None,
) -> tuple[str, str]:
    """Exchange the large-parent hash through the rendezvous store.

    The parent checkpoint lives on a slow NFS volume.  Rank 0 must hash it
    before training, but using an NCCL object broadcast for that hand-off lets
    the other ranks enter a lazy NCCL communicator setup while rank 0 is still
    reading tens of gigabytes.  The rendezvous store is already available
    after ``init_process_group`` and does not require a NCCL collective, so it
    can safely wait for the NFS read without consuming the short collective
    timeout.
    """

    store = dist.distributed_c10d._get_default_store()
    run_id = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT", "default")
    key = "h3-parent-hash/%s/%s" % (child_model_id, run_id)
    store.set_timeout(timedelta(seconds=parent_hash_timeout_s))
    try:
        if rank == 0:
            try:
                parent_hash, source = _cached_sha256(parent, parent_hash_cache)
            except Exception as exc:
                # Wake waiting ranks with the real root cause instead of
                # making them wait for the store timeout after rank 0 exits.
                store.set(key, "ERROR: %s" % exc)
                raise
            store.set(key, json.dumps({"sha256": parent_hash, "source": source}))
            return parent_hash, source

        raw_hash = store.get(key)
        encoded = bytes(raw_hash).decode("utf-8")
        if encoded.startswith("ERROR:"):
            raise RuntimeError("rank 0 could not hash parent checkpoint: %s" % encoded[6:].strip())
        payload = json.loads(encoded)
        return str(payload["sha256"]), str(payload.get("source", "unknown"))
    finally:
        # Keep the normal NCCL collectives fail-fast after the slow file
        # exchange has completed.
        store.set_timeout(timedelta(seconds=distributed_timeout_s))


def _load_model_config(checkpoint: Path) -> dict[str, Any]:
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw = metadata.get("config")
    if not raw:
        raise ValueError("parent checkpoint has no ComfyUI config metadata")
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("transformer"), Mapping):
        raise ValueError("parent checkpoint metadata has no transformer config")
    return dict(parsed["transformer"])


def _checkpoint_has_quantized_sidecars(checkpoint: Path) -> bool:
    """Inspect tensor names without materializing the large checkpoint."""

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        return any(
            name.endswith(".comfy_quant") or name.endswith(".weight_scale")
            for name in handle.keys()
        )


class _TrainableLinear(torch.nn.Linear):
    """Autograd-safe output head for mixed-precision ComfyUI models."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype
        output = torch.nn.functional.linear(
            input.to(dtype=self.weight.dtype),
            self.weight,
            self.bias,
        )
        return output.to(dtype=input_dtype) if output.dtype != input_dtype else output


def _replace_trainable_output_heads(model: torch.nn.Module) -> int:
    """Use normal Linear heads so CPU offload does not detach gradients.

    This runs after the checkpoint loader has installed all source weights.  In
    particular, ``assign=True`` and ComfyUI's quantized loader cannot replace
    the module object after this point and silently turn the selected heads
    back into a frozen manual-cast layer.
    """

    final_layer = getattr(model, "final_layer", None)
    if final_layer is None:
        return 0
    replaced = 0
    for name in ("video_out", "audio_out"):
        old = getattr(final_layer, name, None)
        if old is None or not hasattr(old, "in_features") or not hasattr(old, "out_features"):
            continue
        old_weight = getattr(old, "weight", None)
        old_bias = getattr(old, "bias", None)
        if old_weight is not None:
            if hasattr(old_weight, "dequantize"):
                old_weight = old_weight.dequantize()
            if not isinstance(old_weight, torch.Tensor) or not old_weight.is_floating_point():
                raise RuntimeError("%s output head has no floating-point weight" % name)
            target_dtype = old_weight.dtype
        else:
            # With rank0_only_load, non-zero ranks intentionally construct the
            # model without reading the checkpoint.  mixed_precision_ops.Linear
            # leaves weight unset until its state-dict loader runs, but FSDP
            # still needs a concrete, same-shaped module before it can sync the
            # rank-0 parameters.  The checkpoint heads are an FP32 island, so
            # use FP32 for this placeholder; rank 0 supplies the real values.
            target_dtype = torch.float32
        replacement = _TrainableLinear(
            int(old.in_features),
            int(old.out_features),
            bias=old_bias is not None,
            device="cpu",
            dtype=target_dtype,
        )
        if old_weight is not None:
            with torch.no_grad():
                replacement.weight.copy_(old_weight.detach().to(dtype=target_dtype, device="cpu"))
                if old_bias is not None:
                    replacement.bias.copy_(old_bias.detach().to(dtype=target_dtype, device="cpu"))
        setattr(final_layer, name, replacement)
        replaced += 1
    return replaced


def _import_comfy(root: Path):
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    import comfy.model_management
    import comfy.ops
    from comfy.ldm.minimax.model import DiTBlock, MiniMaxH3Model, PackedLayout, time_shift_sigma
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    return {
        "comfy_model_management": comfy.model_management,
        "ops": comfy.ops,
        "DiTBlock": DiTBlock,
        "MiniMaxH3Model": MiniMaxH3Model,
        "QuantizedTensor": comfy.ops.QuantizedTensor,
        "PackedLayout": PackedLayout,
        "time_shift_sigma": time_shift_sigma,
        "FSDP": FSDP,
        "MixedPrecision": MixedPrecision,
        "ShardingStrategy": ShardingStrategy,
        "transformer_auto_wrap_policy": transformer_auto_wrap_policy,
    }


def _materialize_quantized_weights(model: torch.nn.Module, api: Mapping[str, Any]) -> int:
    """Convert ComfyUI quantized weights to BF16 before FSDP training.

    ComfyUI's mixed-precision loader understands the ``.comfy_quant`` sidecar
    records and ConvRot scales. A plain ``load_state_dict`` would instead try
    to install INT8 tensors as trainable parameters, which PyTorch rejects.
    FSDP also needs floating-point parameters for its flat representation, so
    materialize ComfyUI's exact dequantization result once on CPU and keep the
    registered training scope as the only parameters requiring gradients.
    """

    quantized_type = api.get("QuantizedTensor")
    if quantized_type is None:
        return 0
    converted = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, quantized_type):
            continue
        dequantized = weight.dequantize()
        if not isinstance(dequantized, torch.Tensor) or not dequantized.is_floating_point():
            raise RuntimeError("quantized H3 weight did not dequantize to a floating tensor")
        module.weight = torch.nn.Parameter(
            dequantized.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
            requires_grad=False,
        )
        # The sidecars are consumed by the quantized loader only. Removing
        # them prevents FSDP from sharding stale scale parameters after the
        # weight has become an ordinary frozen BF16 tensor.
        for name in ("weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale"):
            if name in getattr(module, "_parameters", {}):
                del module._parameters[name]
            if name in getattr(module, "_buffers", {}):
                del module._buffers[name]
        for name, value in (
            ("quant_format", None),
            ("layout_type", None),
            ("_full_precision_mm", False),
            ("_full_precision_mm_config", False),
        ):
            if hasattr(module, name):
                setattr(module, name, value)
        converted += 1
    return converted


def _construct_model(
    api: Mapping[str, Any],
    checkpoint: Path,
    device: torch.device,
    *,
    load_checkpoint: bool = True,
) -> torch.nn.Module:
    config = _load_model_config(checkpoint)
    model_cls = api["MiniMaxH3Model"]
    mixed_precision_factory = getattr(api["ops"], "mixed_precision_ops", None)
    operations = (
        mixed_precision_factory(compute_dtype=torch.bfloat16, full_precision_mm=True)
        if callable(mixed_precision_factory)
        else api["ops"].disable_weight_init
    )
    with torch.device("cpu"):
        model = model_cls(
            **config,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            operations=operations,
        )
    if load_checkpoint:
        state = load_file(str(checkpoint), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        del state
        if missing or unexpected:
            raise RuntimeError("parent/model key mismatch; missing=%s unexpected=%s" % (missing[:3], unexpected[:3]))
    replaced_heads = _replace_trainable_output_heads(model)
    if replaced_heads != 2:
        raise RuntimeError("expected two trainable H3 output heads, replaced=%d" % replaced_heads)
    _materialize_quantized_weights(model, api)
    return model


def _stage_checkpoint(checkpoint: Path, stage_dir: Optional[Path], rank: int) -> Path:
    """Copy a shared/NFS checkpoint once to a fast local staging directory."""

    if stage_dir is None:
        return checkpoint
    source = checkpoint.resolve()
    stage_dir = stage_dir.resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    target = stage_dir / (
        "%s-%d-%d.safetensors" % (source.stem, int(source_stat.st_size), int(source_stat.st_mtime_ns))
    )
    if target == source:
        return source
    if rank == 0:
        # The campaign may already be prefetching this exact source.  Share a
        # lock with h3_checkpoint_stage.py so the worker waits for the one
        # copy instead of racing it or reading a partial file.
        lock_path = stage_dir / ".h3-checkpoint-stage.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                partial = target.with_name(target.name + ".part")
                if not target.is_file() or target.stat().st_size != source_stat.st_size:
                    print("[0] staging checkpoint to %s" % target, flush=True)
                    shutil.copyfile(source, partial)
                    if partial.stat().st_size != source_stat.st_size:
                        raise IOError("staged checkpoint size mismatch: %s" % partial)
                    with partial.open("r+b") as handle:
                        os.fsync(handle.fileno())
                    partial.replace(target)
                    print("[0] checkpoint staging complete", flush=True)
                # This directory is a local read-through cache for one
                # single-flight campaign, not a second model archive.  Keep
                # only the exact source needed by this worker.
                for candidate in stage_dir.glob("*.safetensors"):
                    if candidate == target:
                        continue
                    try:
                        candidate.unlink()
                    except OSError as exc:
                        print("[0] unable to prune staged checkpoint %s: %s" % (candidate, exc), flush=True)
                for candidate in stage_dir.glob("*.safetensors.part"):
                    try:
                        candidate.unlink()
                    except OSError as exc:
                        print("[0] unable to prune staged partial %s: %s" % (candidate, exc), flush=True)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    dist.barrier()
    if not target.is_file() or target.stat().st_size != source_stat.st_size:
        raise IOError("staged checkpoint is unavailable after distributed barrier: %s" % target)
    return target


def _prepare_cache(cache_dir: Path, seed: int) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "00000000.pt"
    if not cache_path.is_file():
        make_smoke_cache(cache_dir, seed)
    try:
        item = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("unable to load H3 cache: %s" % exc) from exc
    if not isinstance(item, Mapping):
        raise ValueError("H3 cache item must be a mapping")
    required = ("video", "audio", "prompt", "height", "width", "latent_frames", "audio_frames")
    if any(name not in item for name in required):
        raise ValueError("H3 cache item is missing required fields")
    if tuple(item["video"].shape) != (320, 96):
        raise ValueError("unexpected video cache shape: %s" % (tuple(item["video"].shape),))
    if tuple(item["audio"].shape) != (74, 32):
        raise ValueError("unexpected audio cache shape: %s" % (tuple(item["audio"].shape),))
    if item["prompt"].ndim != 2 or item["prompt"].shape[1] != 5120:
        raise ValueError("unexpected prompt cache shape: %s" % (tuple(item["prompt"].shape),))
    return dict(item)


def _unpatch_video(rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (
        rows.reshape(5, 8, 8, 24, 1, 2, 2)
        .permute(3, 0, 4, 1, 5, 2, 6)
        .reshape(1, 24, 5, 16, 16)
        .to(device=device, dtype=torch.bfloat16)
    )


def _unpack_audio(rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    return rows.reshape(2, 37, 32).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)


def _make_batch(item: Mapping[str, Any], device: torch.device, sigma: float, seed: int, api: Mapping[str, Any]):
    clean_video = _unpatch_video(item["video"].to(torch.float32), device)
    clean_audio = _unpack_audio(item["audio"].to(torch.float32), device)
    video_generator = torch.Generator(device="cuda").manual_seed(seed)
    audio_generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    noise_video = torch.randn(clean_video.shape, generator=video_generator, device=device, dtype=torch.bfloat16)
    noise_audio = torch.randn(clean_audio.shape, generator=audio_generator, device=device, dtype=torch.bfloat16)
    sigma_v = torch.tensor(float(sigma), device=device, dtype=torch.float32)
    sigma_a = api["time_shift_sigma"](sigma_v, 12.0, 3.0)
    noisy_video = (1.0 - sigma_v).to(torch.bfloat16) * clean_video + sigma_v.to(torch.bfloat16) * noise_video
    noisy_audio = (1.0 - sigma_a).to(torch.bfloat16) * clean_audio + sigma_a.to(torch.bfloat16) * noise_audio
    layout = api["PackedLayout"](int(item["prompt"].shape[0]), 5, 16, 16, 37)
    # ComfyUI's H3 forward expects a batched tensor here and indexes context[0]
    # internally; it is not the list-of-tensors convention used by some
    # Diffusers pipelines.
    context = item["prompt"].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    timestep = (sigma_v * 1000.0).reshape(1)
    target_video = -(clean_video - noise_video)
    target_audio = -(clean_audio - noise_audio)
    return (noisy_video, noisy_audio), timestep, context, layout, target_video, target_audio


def shifted_sigma(sigma: float, shift: float) -> float:
    """Map the unit flow grid to ComfyUI's shifted sigma schedule."""
    base = float(sigma)
    return float(shift * base / (1.0 + (shift - 1.0) * base))


def binary_sigma_schedule(nfe: int, video_shift: float = 12.0, audio_shift: float = 3.0) -> tuple[list[float], list[float]]:
    """Return aligned video/audio schedules for one binary distillation stage."""
    if nfe <= 0:
        raise ValueError("nfe must be positive")
    base = [1.0 - index / float(nfe) for index in range(nfe + 1)]
    video = [shifted_sigma(value, video_shift) for value in base]
    audio = [
        shifted_sigma(value, audio_shift)
        for value in [
            value / (video_shift + value * (1.0 - video_shift))
            for value in video
        ]
    ]
    return video, audio


def _make_distill_inputs(item: Mapping[str, Any], device: torch.device, seed: int, api: Mapping[str, Any]):
    clean_video = _unpatch_video(item["video"].to(torch.float32), device)
    clean_audio = _unpack_audio(item["audio"].to(torch.float32), device)
    video_generator = torch.Generator(device="cuda").manual_seed(seed)
    audio_generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    noise_video = torch.randn(clean_video.shape, generator=video_generator, device=device, dtype=torch.bfloat16)
    noise_audio = torch.randn(clean_audio.shape, generator=audio_generator, device=device, dtype=torch.bfloat16)
    context = item["prompt"].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    layout = api["PackedLayout"](int(item["prompt"].shape[0]), 5, 16, 16, 37)
    return clean_video, clean_audio, noise_video, noise_audio, context, layout


def _mix_at_sigma(clean: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma_value = torch.tensor(float(sigma), device=clean.device, dtype=torch.bfloat16)
    return (1.0 - sigma_value) * clean + sigma_value * noise


def _native_prediction(
    model: torch.nn.Module,
    video: torch.Tensor,
    audio: torch.Tensor,
    sigma_video: float,
    context: torch.Tensor,
    layout: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive clean-minus-noise velocity from ComfyUI's raw output."""
    raw_video, raw_audio = model(
        [video, audio],
        torch.tensor([float(sigma_video) * 1000.0], device=video.device, dtype=torch.float32),
        context,
        transformer_options={},
        minimax_payload={"layout": layout, "audio_scale": 1.0},
    )
    return -raw_video, -raw_audio


def _distill_target(
    model: torch.nn.Module,
    clean_video: torch.Tensor,
    clean_audio: torch.Tensor,
    noise_video: torch.Tensor,
    noise_audio: torch.Tensor,
    context: torch.Tensor,
    layout: Any,
    teacher_video: list[float],
    teacher_audio: list[float],
    interval_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate two frozen-teacher steps over one student interval."""
    teacher_index = interval_index * 2
    start_video = _mix_at_sigma(clean_video, noise_video, teacher_video[teacher_index])
    start_audio = _mix_at_sigma(clean_audio, noise_audio, teacher_audio[teacher_index])
    first_video, first_audio = _native_prediction(
        model, start_video, start_audio, teacher_video[teacher_index], context, layout
    )
    middle_video = start_video + (teacher_video[teacher_index] - teacher_video[teacher_index + 1]) * first_video
    middle_audio = start_audio + (teacher_audio[teacher_index] - teacher_audio[teacher_index + 1]) * first_audio
    second_video, second_audio = _native_prediction(
        model, middle_video, middle_audio, teacher_video[teacher_index + 1], context, layout
    )
    return (
        middle_video + (teacher_video[teacher_index + 1] - teacher_video[teacher_index + 2]) * second_video,
        middle_audio + (teacher_audio[teacher_index + 1] - teacher_audio[teacher_index + 2]) * second_audio,
    )


def _distill_loss(
    model: torch.nn.Module,
    clean_video: torch.Tensor,
    clean_audio: torch.Tensor,
    noise_video: torch.Tensor,
    noise_audio: torch.Tensor,
    context: torch.Tensor,
    layout: Any,
    student_video: list[float],
    student_audio: list[float],
    target_video: torch.Tensor,
    target_audio: torch.Tensor,
    interval_index: int,
) -> torch.Tensor:
    start_video = _mix_at_sigma(clean_video, noise_video, student_video[interval_index])
    start_audio = _mix_at_sigma(clean_audio, noise_audio, student_audio[interval_index])
    prediction_video, prediction_audio = _native_prediction(
        model, start_video, start_audio, student_video[interval_index], context, layout
    )
    endpoint_video = start_video + (student_video[interval_index] - student_video[interval_index + 1]) * prediction_video
    endpoint_audio = start_audio + (student_audio[interval_index] - student_audio[interval_index + 1]) * prediction_audio
    loss = torch.nn.functional.mse_loss(endpoint_video.float(), target_video.float())
    loss = loss + torch.nn.functional.mse_loss(endpoint_audio.float(), target_audio.float())
    if not torch.isfinite(loss):
        raise FloatingPointError("H3 distillation loss is non-finite")
    return loss


def _raw_prediction(
    model: torch.nn.Module,
    batch: Any,
    api: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the model's raw velocity prediction for output distillation."""
    (video, audio), timestep, context, layout, _, _ = batch
    with torch.no_grad():
        predicted_video, predicted_audio = model(
            [video, audio],
            timestep,
            context,
            transformer_options={},
            minimax_payload={"layout": layout, "audio_scale": 1.0},
        )
    return predicted_video.detach(), predicted_audio.detach()


def _output_distill_loss(
    model: torch.nn.Module,
    batch: Any,
    teacher_video: torch.Tensor,
    teacher_audio: torch.Tensor,
    api: Mapping[str, Any],
) -> torch.Tensor:
    """Match a frozen parent prediction while updating only the student heads."""
    (video, audio), timestep, context, layout, _, _ = batch
    predicted_video, predicted_audio = model(
        [video, audio],
        timestep,
        context,
        transformer_options={},
        minimax_payload={"layout": layout, "audio_scale": 1.0},
    )
    loss = torch.nn.functional.mse_loss(predicted_video.float(), teacher_video.float())
    loss = loss + torch.nn.functional.mse_loss(predicted_audio.float(), teacher_audio.float())
    if not torch.isfinite(loss):
        raise FloatingPointError("H3 output distillation loss is non-finite")
    return loss


class H3DMD2Critic(torch.nn.Module):
    """Small latent-space critic used by the real H3 DMD2 worker.

    The full MiniMax-H3 teacher is FSDP-sharded and frozen.  A second full
    diffusion transformer as a trainable critic would exceed the available
    memory and would not be an auditable bounded experiment.  This critic
    predicts the clean-minus-noise velocity in latent space with modality-
    preserving 1x1 projections and an explicit sigma embedding.  It is a
    real trainable critic, not a scalar or offline metric surrogate.
    """

    def __init__(self, video_channels: int = 24, audio_channels: int = 32) -> None:
        super().__init__()
        self.video_projection = torch.nn.Conv3d(video_channels, video_channels, kernel_size=1)
        self.audio_projection = torch.nn.Conv2d(audio_channels, audio_channels, kernel_size=1)
        self.video_sigma = torch.nn.Linear(1, video_channels)
        self.audio_sigma = torch.nn.Linear(1, audio_channels)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_value = torch.full(
            (video.shape[0], 1),
            float(sigma),
            device=video.device,
            dtype=video.dtype,
        )
        video_bias = self.video_sigma(sigma_value).reshape(video.shape[0], -1, 1, 1, 1)
        audio_bias = self.audio_sigma(sigma_value).reshape(audio.shape[0], -1, 1, 1)
        return self.video_projection(video) + video_bias, self.audio_projection(audio) + audio_bias


def _head_parameters(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    selected = [(name, parameter) for name, parameter in module.named_parameters() if name in TRAINABLE_TENSOR_NAMES]
    names = {name for name, _ in selected}
    if names != set(TRAINABLE_TENSOR_NAMES):
        raise RuntimeError("model trainable scope mismatch: %s" % sorted(set(TRAINABLE_TENSOR_NAMES) - names))
    return selected


def _wrap_fsdp(
    model: torch.nn.Module,
    api: Mapping[str, Any],
    device: torch.device,
    train_heads: bool = True,
    sync_module_states: bool = True,
) -> torch.nn.Module:
    model.requires_grad_(False)
    if train_heads:
        for name, parameter in model.named_parameters():
            if name in TRAINABLE_TENSOR_NAMES:
                parameter.requires_grad_(True)
    layer_policy = lambda module, recurse, nonwrapped_numel: api["transformer_auto_wrap_policy"](
        module, recurse, nonwrapped_numel, {api["DiTBlock"]}
    )
    mixed = api["MixedPrecision"](
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    # The released ComfyUI checkpoint intentionally keeps these small input
    # and timestep projections in FP32. FSDP flat parameters must have one
    # dtype, so keep the frozen FP32 islands replicated while sharding the
    # BF16 transformer body. The two output heads are also replicated because
    # their gradients are reduced explicitly below.
    ignored = [
        model.video_patch_proj,
        model.audio_patch_proj,
        model.time_embedder,
        model.final_layer.video_out,
        model.final_layer.audio_out,
    ]
    fsdp_model = api["FSDP"](
        model,
        sharding_strategy=api["ShardingStrategy"].FULL_SHARD,
        auto_wrap_policy=layer_policy,
        ignored_modules=ignored,
        mixed_precision=mixed,
        use_orig_params=True,
        device_id=device,
        sync_module_states=sync_module_states,
        limit_all_gathers=True,
    )
    # FSDP deliberately skips moving ignored modules. They are replicated
    # rather than sharded, so place their frozen weights on this rank now.
    for module in ignored:
        module.to(device)
    if train_heads:
        # Keep this post-wrap assignment explicit: ignored modules retain
        # their original Parameter objects, while FSDP may change the body
        # parameter views during construction.
        for name, parameter in fsdp_model.named_parameters():
            normalized_name = name.split("_fsdp_wrapped_module.", 1)[-1]
            if normalized_name in TRAINABLE_TENSOR_NAMES:
                parameter.requires_grad_(True)
    return fsdp_model


def _forward_loss(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    api: Mapping[str, Any],
    requires_grad: bool,
) -> torch.Tensor:
    (video, audio), timestep, context, layout, target_video, target_audio = batch
    payload = {"layout": layout, "audio_scale": 1.0}
    options: dict[str, Any] = {}
    grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
    with grad_context:
        predicted_video, predicted_audio = model(
            [video, audio],
            timestep,
            context,
            transformer_options=options,
            minimax_payload=payload,
        )
        loss_video = torch.nn.functional.mse_loss(predicted_video.float(), target_video.float())
        loss_audio = torch.nn.functional.mse_loss(predicted_audio.float(), target_audio.float())
        loss = loss_video + loss_audio
    if not torch.isfinite(loss):
        raise FloatingPointError("H3 loss is non-finite")
    return loss


def _gradient_diagnostic(
    model: torch.nn.Module,
    selected: list[tuple[str, torch.nn.Parameter]],
    loss: torch.Tensor,
) -> str:
    """Describe the trainable path when a worker loses its autograd graph."""

    wrapped = getattr(model, "module", model)
    heads = []
    final_layer = getattr(wrapped, "final_layer", None)
    for name in ("video_out", "audio_out"):
        head = getattr(final_layer, name, None) if final_layer is not None else None
        heads.append(
            {
                "name": name,
                "type": type(head).__name__ if head is not None else None,
                "weight_requires_grad": bool(
                    getattr(getattr(head, "weight", None), "requires_grad", False)
                ),
            }
        )
    selected_state = [
        {"name": name, "requires_grad": bool(parameter.requires_grad), "shape": list(parameter.shape)}
        for name, parameter in selected
    ]
    return "loss has no grad_fn; loss_requires_grad=%s heads=%s selected=%s" % (
        bool(loss.requires_grad),
        heads,
        selected_state,
    )


def _assert_trainable_heads(
    model: torch.nn.Module,
    selected: list[tuple[str, torch.nn.Parameter]],
    device: torch.device,
) -> None:
    """Fail before the expensive teacher pass if heads are not trainable."""

    if any(not parameter.requires_grad for _, parameter in selected):
        raise RuntimeError(_gradient_diagnostic(model, selected, torch.zeros((), device=device)))
    wrapped = getattr(model, "module", model)
    final_layer = getattr(wrapped, "final_layer", None)
    if final_layer is None:
        raise RuntimeError("H3 model has no final_layer for head training")
    with torch.enable_grad():
        probe_input = torch.zeros(
            (1, int(final_layer.video_out.in_features)),
            device=device,
            dtype=torch.bfloat16,
        )
        probe = final_layer.video_out(probe_input)
        if not probe.requires_grad:
            raise RuntimeError(_gradient_diagnostic(model, selected, probe.sum()))
    del probe_input, probe


def _reduce_head_gradients(
    selected: list[tuple[str, torch.nn.Parameter]], world_size: int, device: torch.device
) -> float:
    """Synchronize the replicated output heads and return their global norm."""
    for name, parameter in selected:
        if parameter.grad is None:
            raise RuntimeError("zero gradient: %s" % name)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)
    grad_sq = torch.zeros((), device=device, dtype=torch.float64)
    for _, parameter in selected:
        grad_sq += parameter.grad.detach().double().square().sum()
    dist.all_reduce(grad_sq, op=dist.ReduceOp.SUM)
    gradient_norm = float(grad_sq.sqrt().item() / math.sqrt(world_size))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("zero or non-finite gradient norm")
    return gradient_norm


def _recovery_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    selected: list[tuple[str, torch.nn.Parameter]],
    batch: Any,
    world_size: int,
    device: torch.device,
    api: Mapping[str, Any],
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss_tensor = _forward_loss(model, batch, device, api, True)
    loss = float(loss_tensor.detach().cpu())
    if not loss_tensor.requires_grad:
        raise RuntimeError(_gradient_diagnostic(model, selected, loss_tensor))
    loss_tensor.backward()
    gradient_norm = _reduce_head_gradients(selected, world_size, device)
    torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss, gradient_norm


def _distill_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    selected: list[tuple[str, torch.nn.Parameter]],
    loss_fn,
    world_size: int,
    device: torch.device,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss_tensor = loss_fn()
    loss = float(loss_tensor.detach().cpu())
    if not loss_tensor.requires_grad:
        raise RuntimeError(_gradient_diagnostic(model, selected, loss_tensor))
    loss_tensor.backward()
    gradient_norm = _reduce_head_gradients(selected, world_size, device)
    torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss, gradient_norm


def _reduce_module_gradients(module: torch.nn.Module, world_size: int, device: torch.device) -> float:
    """Synchronize gradients for the replicated lightweight DMD2 critic."""
    grad_sq = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is None:
            raise RuntimeError("zero gradient in DMD2 critic")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)
        grad_sq += parameter.grad.detach().double().square().sum()
    dist.all_reduce(grad_sq, op=dist.ReduceOp.SUM)
    gradient_norm = float(grad_sq.sqrt().item() / math.sqrt(world_size))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("zero or non-finite DMD2 critic gradient norm")
    return gradient_norm


def _seeded_noise_like(value: torch.Tensor, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    return torch.randn(value.shape, generator=generator, device=device, dtype=value.dtype)


def _h3_clean_sample(
    model: torch.nn.Module,
    batch: Any,
    api: Mapping[str, Any],
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Generate one Euler clean-latent sample from a native H3 velocity."""
    (video, audio), timestep, context, layout, _, _ = batch
    sigma = float(timestep.reshape(-1)[0].detach().cpu()) / 1000.0
    grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
    with grad_context:
        velocity_video, velocity_audio = _native_prediction(
            model, video, audio, sigma, context, layout
        )
        clean_video = video + sigma * velocity_video
        clean_audio = audio + sigma * velocity_audio
    return clean_video, clean_audio, sigma


def _dmd2_update(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    critic: H3DMD2Critic,
    student_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    selected: list[tuple[str, torch.nn.Parameter]],
    batch: Any,
    world_size: int,
    device: torch.device,
    api: Mapping[str, Any],
    step_index: int,
    generator_update_interval: int,
    seed: int,
) -> tuple[float, float, Optional[float], Optional[float], bool]:
    """Run one real-H3 DMD2 critic update and optional student update."""
    update_student = step_index % generator_update_interval == 0
    generated_video, generated_audio, sigma = _h3_clean_sample(student, batch, api, update_student)
    detached_video, detached_audio = generated_video.detach(), generated_audio.detach()

    critic_noise_video = _seeded_noise_like(detached_video, seed + 100_000 + step_index * 2, device)
    critic_noise_audio = _seeded_noise_like(detached_audio, seed + 100_001 + step_index * 2, device)
    critic_noisy_video = _mix_at_sigma(detached_video, critic_noise_video, sigma)
    critic_noisy_audio = _mix_at_sigma(detached_audio, critic_noise_audio, sigma)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_prediction_video, critic_prediction_audio = critic(
        critic_noisy_video, critic_noisy_audio, sigma
    )
    critic_loss = torch.nn.functional.mse_loss(
        critic_prediction_video.float(), (detached_video - critic_noise_video).float()
    ) + torch.nn.functional.mse_loss(
        critic_prediction_audio.float(), (detached_audio - critic_noise_audio).float()
    )
    if not torch.isfinite(critic_loss):
        raise FloatingPointError("H3 DMD2 critic loss is non-finite")
    critic_loss_value = float(critic_loss.detach().cpu())
    critic_loss.backward()
    critic_gradient_norm = _reduce_module_gradients(critic, world_size, device)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optimizer.step()
    critic_optimizer.zero_grad(set_to_none=True)

    student_loss_value: Optional[float] = None
    student_gradient_norm: Optional[float] = None
    if update_student:
        score_noise_video = _seeded_noise_like(detached_video, seed + 200_000 + step_index * 2, device)
        score_noise_audio = _seeded_noise_like(detached_audio, seed + 200_001 + step_index * 2, device)
        score_noisy_video = _mix_at_sigma(detached_video, score_noise_video, sigma)
        score_noisy_audio = _mix_at_sigma(detached_audio, score_noise_audio, sigma)
        with torch.no_grad():
            teacher_video, teacher_audio = _native_prediction(
                teacher, score_noisy_video, score_noisy_audio, sigma, batch[2], batch[3]
            )
            fake_video, fake_audio = critic(score_noisy_video, score_noisy_audio, sigma)
            teacher_clean_video = score_noisy_video + sigma * teacher_video
            teacher_clean_audio = score_noisy_audio + sigma * teacher_audio
            fake_clean_video = score_noisy_video + sigma * fake_video
            fake_clean_audio = score_noisy_audio + sigma * fake_audio
            difference_video = teacher_clean_video - fake_clean_video
            difference_audio = teacher_clean_audio - fake_clean_audio
            scale = torch.cat(
                [difference_video.float().reshape(-1), difference_audio.float().reshape(-1)]
            ).abs().mean().clamp_min(1e-6)
            pseudo_video = detached_video + (difference_video / scale).clamp(-10.0, 10.0)
            pseudo_audio = detached_audio + (difference_audio / scale).clamp(-10.0, 10.0)
        student_optimizer.zero_grad(set_to_none=True)
        student_loss = torch.nn.functional.mse_loss(generated_video.float(), pseudo_video.float()) + torch.nn.functional.mse_loss(
            generated_audio.float(), pseudo_audio.float()
        )
        if not torch.isfinite(student_loss):
            raise FloatingPointError("H3 DMD2 student loss is non-finite")
        student_loss_value = float(student_loss.detach().cpu())
        student_loss.backward()
        student_gradient_norm = _reduce_head_gradients(selected, world_size, device)
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected], 1.0)
        student_optimizer.step()
        student_optimizer.zero_grad(set_to_none=True)
    return critic_loss_value, critic_gradient_norm, student_loss_value, student_gradient_norm, update_student


def _range_equal(path_a: Path, path_b: Path, start: int, length: int) -> bool:
    with path_a.open("rb") as first, path_b.open("rb") as second:
        first.seek(start)
        second.seek(start)
        remaining = length
        while remaining:
            block = min(1024 * 1024, remaining)
            if first.read(block) != second.read(block):
                return False
            remaining -= block
    return True


def _verify_child_ranges(
    parent: Path,
    child: Path,
    replacements: Mapping[str, torch.Tensor],
    full_frozen_scan: bool = True,
) -> tuple[int, int]:
    parent_header, data_start = read_safetensors_header(parent)
    child_header, child_data_start = read_safetensors_header(child)
    if parent_header != child_header or data_start != child_data_start:
        raise ValueError("child safetensors header changed")
    if parent.stat().st_size != child.stat().st_size:
        raise ValueError("child safetensors size changed")
    changed = 0
    frozen = sum(1 for name in parent_header if name not in replacements and name != "__metadata__")
    for name, entry in parent_header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        if name in replacements:
            same = _range_equal(parent, child, data_start + start, int(end - start))
            changed += int(not same)
        elif full_frozen_scan:
            # A full scan is useful on local storage, but is prohibitively
            # slow for a child on the experiment NFS volume. The fast path is
            # still safe because patch_safetensors_tensors first makes a
            # byte-for-byte copy and then writes only fixed replacement ranges.
            same = _range_equal(parent, child, data_start + start, int(end - start))
            if not same:
                raise RuntimeError("frozen tensor changed: %s" % name)
    return changed, frozen


def _reload_child(api: Mapping[str, Any], child: Path) -> int:
    model = _construct_model(api, child, torch.device("cpu"))
    state = model.state_dict()
    if not TRAINABLE_TENSOR_NAMES.issubset(state):
        raise RuntimeError("reloaded child is missing output heads")
    for name in TRAINABLE_TENSOR_NAMES:
        value = state[name]
        if not torch.isfinite(value).all():
            raise RuntimeError("reloaded child has non-finite tensor: %s" % name)
    count = len(state)
    del state, model
    gc.collect()
    return count


def _run(request: Mapping[str, Any], config: Mapping[str, Any], result_path: Path) -> int:
    started = time.monotonic()
    rank = 0
    distributed = False
    partial: Optional[Path] = None
    try:
        paths = validate_config(config, request)
        _install_termination_handlers()
        rank, world_size, local_rank = _init_distributed(paths["distributed_timeout_s"])
        distributed = True
        device = torch.device("cuda", local_rank)
        api = _import_comfy(paths["comfyui_root"])
        api["comfy_model_management"].in_training = True
        quantized_parent = _checkpoint_has_quantized_sidecars(paths["model_checkpoint"])
        torch.manual_seed(paths["seed"])
        torch.cuda.manual_seed_all(paths["seed"])
        phase_started = time.monotonic()
        parent_hash, parent_hash_source = _exchange_parent_hash(
            paths["model_checkpoint"],
            str(request["child_model_id"]),
            rank,
            paths["distributed_timeout_s"],
            paths["parent_hash_timeout_s"],
            paths["parent_hash_cache"],
        )
        print("[%d] parent hash verified (%s)" % (rank, parent_hash_source), flush=True)
        print("[%d] phase=parent_hash elapsed_s=%.1f" % (rank, time.monotonic() - phase_started), flush=True)
        paths["model_checkpoint"] = _stage_checkpoint(
            paths["model_checkpoint"], paths["checkpoint_stage_dir"], rank
        )
        print("[%d] checkpoint=%s" % (rank, paths["model_checkpoint"]), flush=True)
        teacher_hash = None
        if paths["operator"] == "distill":
            teacher_stage_dir = (
                paths["checkpoint_stage_dir"] / "teacher"
                if paths["checkpoint_stage_dir"] is not None
                else None
            )
            paths["teacher_checkpoint"] = _stage_checkpoint(
                paths["source_parent_checkpoint"], teacher_stage_dir, rank
            )
            teacher_hash, _ = _exchange_parent_hash(
                paths["teacher_checkpoint"],
                str(request["child_model_id"]) + "-teacher",
                rank,
                paths["distributed_timeout_s"],
                paths["parent_hash_timeout_s"],
                paths["parent_hash_cache"],
            )
            print("[%d] teacher checkpoint=%s hash=%s" % (rank, paths["teacher_checkpoint"], teacher_hash), flush=True)
        if rank == 0:
            stale_partial = paths["output_dir"] / (
                str(request["child_model_id"]) + ".safetensors.part"
            )
            if stale_partial.exists():
                stale_partial.unlink()
        dist.barrier()
        item = _prepare_cache(paths["cache_dir"], paths["seed"])
        print("[%d] phase=cache_ready elapsed_s=%.1f" % (rank, time.monotonic() - phase_started), flush=True)
        model = _construct_model(
            api,
            paths["model_checkpoint"],
            device,
            load_checkpoint=(rank == 0 or not paths["rank0_only_load"]),
        )
        print(
            "[%d] model structure ready load_checkpoint=%s elapsed_s=%.1f"
            % (rank, rank == 0, time.monotonic() - phase_started),
            flush=True,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        selected = _head_parameters(model)
        trainable_count = sum(parameter.numel() for _, parameter in selected)
        fsdp_model = _wrap_fsdp(
            model,
            api,
            device,
            sync_module_states=paths["rank0_only_load"],
        )
        _assert_trainable_heads(fsdp_model, selected, device)
        print("[%d] FSDP wrapped elapsed_s=%.1f" % (rank, time.monotonic() - phase_started), flush=True)
        teacher_model = None
        teacher_fsdp = None
        critic_model = None
        critic_optimizer = None
        if paths["operator"] in {"distill", "dmd2"}:
            # Distillation and DMD2 both need an independent frozen teacher
            # role.  It is separately FSDP-sharded so each rank holds only
            # its shard of the second H3 model.  Output distillation must not
            # use the student's own initial output as its target: that makes
            # the loss identically zero and consumes a full distributed run
            # before the gradient guard can report the defect.
            teacher_model = _construct_model(
                api,
                paths["teacher_checkpoint"] if paths["operator"] == "distill" else paths["model_checkpoint"],
                device,
                load_checkpoint=(rank == 0 or not paths["rank0_only_load"]),
            )
            teacher_fsdp = _wrap_fsdp(
                teacher_model,
                api,
                device,
                train_heads=False,
                sync_module_states=paths["rank0_only_load"],
            )
            teacher_fsdp.eval()
            if paths["operator"] == "dmd2":
                critic_model = H3DMD2Critic().to(device=device, dtype=torch.bfloat16)
                critic_optimizer = torch.optim.AdamW(
                    critic_model.parameters(),
                    lr=float(paths.get("critic_learning_rate", paths["learning_rate"])),
                    betas=(0.9, 0.95),
                    weight_decay=0.01,
                )
                print("[%d] DMD2 frozen teacher and latent critic ready" % rank, flush=True)
            else:
                print("[%d] output-distill frozen source teacher ready" % rank, flush=True)
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in selected],
            lr=paths["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        print("[%d] training batches ready elapsed_s=%.1f" % (rank, time.monotonic() - phase_started), flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        step_losses: list[float] = []
        step_gradient_norms: list[float] = []
        critic_losses: list[float] = []
        critic_gradient_norms: list[float] = []
        student_losses: list[float] = []
        student_gradient_norms: list[float] = []
        critic_updates = 0
        student_updates = 0
        last_batch = None
        distill_data = None
        distill_targets = None
        student_video = student_audio = None
        if paths["operator"] in {"distill", "step_distill"}:
            distill_data = _make_distill_inputs(item, device, paths["seed"], api)
            clean_video, clean_audio, noise_video, noise_audio, context, layout = distill_data
            if paths["operator"] == "step_distill":
                student_video, student_audio = binary_sigma_schedule(paths["target_steps"])
                teacher_video, teacher_audio = binary_sigma_schedule(paths["source_steps"])
                with torch.inference_mode():
                    distill_targets = tuple(
                        _distill_target(
                            fsdp_model,
                            clean_video,
                            clean_audio,
                            noise_video,
                            noise_audio,
                            context,
                            layout,
                            teacher_video,
                            teacher_audio,
                            index,
                        )
                        for index in range(paths["target_steps"])
                    )
                print("[%d] teacher trajectory cached (%d intervals)" % (rank, paths["target_steps"]), flush=True)
            else:
                # Output distillation freezes the parent prediction before the
                # first optimizer step.  The tensors are detached, so later
                # updates cannot move the teacher target with the student.
                distill_batch = _make_batch(item, device, paths["sigma"], paths["seed"], api)
                distill_targets = _raw_prediction(teacher_fsdp, distill_batch, api)
                distill_data = distill_batch
                print("[%d] frozen source-teacher output cached" % rank, flush=True)
            # The teacher pass performs many FSDP all-gathers. Release
            # allocator blocks before the first gradient-bearing student pass
            # so a transient peak cannot look like a random rank failure.
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        for step_index in range(paths["max_steps"]):
            if paths["operator"] in {"step_distill", "dmd2"}:
                print(
                    "[%d] %s update start (%d/%d)"
                    % (rank, paths["operator"], step_index + 1, paths["max_steps"]),
                    flush=True,
                )
            if paths["operator"] == "dmd2":
                assert teacher_fsdp is not None and critic_model is not None and critic_optimizer is not None
                critic_loss, critic_gradient, student_loss, student_gradient, _ = _dmd2_update(
                    fsdp_model,
                    teacher_fsdp,
                    critic_model,
                    optimizer,
                    critic_optimizer,
                    selected,
                    _make_batch(item, device, paths["sigma"], paths["seed"] + step_index, api),
                    world_size,
                    device,
                    api,
                    step_index,
                    paths["generator_update_interval"],
                    paths["seed"],
                )
                critic_losses.append(critic_loss)
                critic_gradient_norms.append(critic_gradient)
                critic_updates += 1
                if student_loss is not None:
                    student_losses.append(student_loss)
                if student_gradient is not None:
                    student_gradient_norms.append(student_gradient)
                    student_updates += 1
                print(
                    "[%d] dmd2 update complete critic_loss=%.8f critic_grad=%.8f student_loss=%s"
                    % (
                        rank,
                        critic_loss,
                        critic_gradient,
                        "%.8f" % student_loss if student_loss is not None else "skipped",
                    ),
                    flush=True,
                )
            elif paths["operator"] == "recovery_finetune":
                # Keep the smoke sample fixed while changing the noise seed
                # per update. The only campaign variable is step count.
                last_batch = _make_batch(item, device, paths["sigma"], paths["seed"] + step_index, api)
                loss, step_gradient_norm = _recovery_update(
                    fsdp_model, optimizer, selected, last_batch, world_size, device, api
                )
            elif paths["operator"] == "step_distill":
                assert distill_data is not None and distill_targets is not None
                clean_video, clean_audio, noise_video, noise_audio, context, layout = distill_data
                interval_index = step_index % paths["target_steps"]
                target_video, target_audio = distill_targets[interval_index]
                loss, step_gradient_norm = _distill_update(
                    fsdp_model,
                    optimizer,
                    selected,
                    lambda: _distill_loss(
                        fsdp_model,
                        clean_video,
                        clean_audio,
                        noise_video,
                        noise_audio,
                        context,
                        layout,
                        student_video,
                        student_audio,
                        target_video,
                        target_audio,
                        interval_index,
                    ),
                    world_size,
                    device,
                )
            elif paths["operator"] == "distill":
                assert distill_data is not None and distill_targets is not None
                teacher_video, teacher_audio = distill_targets
                loss, step_gradient_norm = _distill_update(
                    fsdp_model,
                    optimizer,
                    selected,
                    lambda: _output_distill_loss(
                        fsdp_model,
                        distill_data,
                        teacher_video,
                        teacher_audio,
                        api,
                    ),
                    world_size,
                    device,
                )
            if paths["operator"] != "dmd2":
                step_losses.append(loss)
                step_gradient_norms.append(step_gradient_norm)
                if paths["operator"] in {"distill", "step_distill"}:
                    print(
                        "[%d] student update complete loss=%.8f grad=%.8f"
                        % (rank, loss, step_gradient_norm),
                        flush=True,
                    )
            # FSDP/NCCL launches collectives asynchronously.  Do not let a
            # rank report a completed optimizer step while a CUDA failure is
            # still pending; synchronizing here makes the failing rank enter
            # the exception path before the other ranks reach finalization.
            torch.cuda.synchronize(device)
        end_event.record()
        end_event.synchronize()
        if paths["operator"] == "dmd2":
            assert critic_losses
            initial_loss = critic_losses[0]
        else:
            initial_loss = step_losses[0]
        final_loss_source = "post_update_forward"
        if paths["operator"] == "recovery_finetune":
            assert last_batch is not None
            final_loss = float(_forward_loss(fsdp_model, last_batch, device, api, False).detach().cpu())
        elif paths["operator"] == "dmd2":
            final_loss = student_losses[-1] if student_losses else critic_losses[-1]
            final_loss_source = "last_dmd2_student_loss" if student_losses else "last_dmd2_critic_loss"
        else:
            # The distillation update already produced a finite, measured
            # loss.  Running another full H3/FSDP forward here adds no
            # evaluator evidence and used to trigger a second unguarded
            # collective immediately after the fragile student update.  The
            # independent evaluator remains responsible for post-update
            # quality; this field records exactly which worker loss is used.
            final_loss = step_losses[-1]
            final_loss_source = "last_student_update_loss"
        print("[%d] training phase complete final_loss=%.8f" % (rank, final_loss), flush=True)
        all_gradient_norms = step_gradient_norms + critic_gradient_norms + student_gradient_norms
        gradient_norm = max(all_gradient_norms)
        if not math.isfinite(final_loss):
            raise FloatingPointError("final H3 loss is non-finite")
        gpu_time_s = float(start_event.elapsed_time(end_event) / 1000.0)
        peak_vram = int(torch.cuda.max_memory_allocated(device))
        peak_vram_by_rank = _gather_int(peak_vram, device)
        gpu_time_by_rank = _gather_float(gpu_time_s, device)
        replacements = {name: parameter.detach().to("cpu").clone() for name, parameter in selected}
        # All NCCL collectives are complete before rank 0 touches the large
        # checkpoint.  Do not leave the other ranks in a CUDA barrier while
        # rank 0 copies/hashes a ~66 GB safetensors file on NFS: the NCCL
        # watchdog treats that normal I/O skew as a deadlock after 10 minutes.
        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            return 0
        child = paths["output_dir"] / (str(request["child_model_id"]) + ".safetensors")
        if child.exists():
            raise FileExistsError("child checkpoint already exists: %s" % child)
        partial = child.with_name(child.name + ".part")
        if partial.exists():
            partial.unlink()
        parent_after, parent_after_source = _cached_sha256(
            paths["model_checkpoint"], paths["parent_hash_cache"]
        )
        if parent_after != parent_hash:
            raise RuntimeError("parent checkpoint hash changed during training")
        # Keep the final filename reserved for a fully written, verified
        # checkpoint.  An interrupted NFS copy may leave the .part file, but
        # it can never be mistaken for a usable child model.
        patch_safetensors_tensors(paths["model_checkpoint"], partial, replacements)
        changed, unchanged_frozen = _verify_child_ranges(
            paths["model_checkpoint"],
            partial,
            replacements,
            full_frozen_scan=paths["full_frozen_scan"],
        )
        if changed <= 0:
            raise RuntimeError("no trainable tensor changed")
        child_hash = _sha256_file(partial)
        if child_hash == parent_hash:
            raise RuntimeError("child hash equals parent hash")
        del optimizer, fsdp_model, model
        if critic_optimizer is not None:
            del critic_optimizer
        if critic_model is not None:
            del critic_model
        if teacher_fsdp is not None:
            del teacher_fsdp
        if teacher_model is not None:
            del teacher_model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded_tensor_count = _reload_child(api, partial)
        partial.replace(child)
        last_gradient_norm = (
            step_gradient_norms[-1]
            if step_gradient_norms
            else student_gradient_norms[-1]
            if student_gradient_norms
            else critic_gradient_norms[-1]
        )
        optimizer_steps = (
            critic_updates + student_updates
            if paths["operator"] == "dmd2"
            else paths["max_steps"]
        )
        loss_history = (
            {"critic": critic_losses, "student": student_losses}
            if paths["operator"] == "dmd2"
            else step_losses
        )
        evidence = {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "final_loss_source": final_loss_source,
                "gradient_norm": gradient_norm,
                "gradient_norm_last": last_gradient_norm,
                "optimizer_steps": optimizer_steps,
                "step_losses": loss_history,
                "critic_updates": critic_updates,
                "student_updates": student_updates,
                "dmd2_role_updates": {
                    "critic": critic_updates,
                    "student": student_updates,
                    "fake_score": student_updates,
                },
                "dmd2_generator_update_interval": paths["generator_update_interval"],
                "trainable_parameter_count": int(trainable_count),
                "parent_sha256": parent_hash,
                "parent_sha256_before": parent_hash,
                "parent_sha256_after": parent_after,
                "parent_hash_source": parent_hash_source,
                "parent_hash_after_source": parent_after_source,
                "child_sha256": child_hash,
                "changed_trainable_tensors": int(changed),
                "unchanged_frozen_tensors": int(unchanged_frozen),
                "frozen_tensor_verification": (
                    "full_range_scan" if paths["full_frozen_scan"] else "byte_for_byte_parent_copy"
                ),
                "child_reloaded": True,
                "reloaded_tensor_count": int(reloaded_tensor_count),
                "peak_vram_per_rank": {str(rank_id): int(value) for rank_id, value in enumerate(peak_vram_by_rank)},
                "peak_vram_gb_per_rank": {
                    str(rank_id): float(value / (1024**3))
                    for rank_id, value in enumerate(peak_vram_by_rank)
                },
                "gpu_time_s_per_rank": gpu_time_by_rank,
                "wall_time_s": time.monotonic() - started,
                "evidence_kind": "real_h3",
                "world_size": world_size,
                "real_worker": True,
                "offline_simulation": False,
            }
        state = {
                "model_id": str(request["child_model_id"]),
                # The external-operator request schema names the parent field
                # ``model_id``.  Keep the emitted ModelState aligned with that
                # contract; using the old ``id`` key here made a completed
                # training update fail only while serializing its evidence.
                "parent_model_id": str(request["parent"]["model_id"]),
                "checkpoint_path": str(child),
                "architecture_name": "MiniMax-H3-FL2VA",
                "parameter_count": int(parameter_count),
                "trainable_parameter_count": int(trainable_count),
                "num_blocks": 50,
                "hidden_size": 5376,
                "num_attention_heads": 56,
                "ffn_width": 14336,
                "dtype": "mixed_quantized" if quantized_parent else "bfloat16",
                "quantization": (
                    {"bits": 8, "scheme": "int8_tensorwise+convrot"}
                    if quantized_parent
                    else {"bits": 16, "scheme": "none"}
                ),
                "sampling_steps": (
                    paths["target_steps"]
                    if paths["operator"] == "step_distill"
                    else int(request.get("parent", {}).get("state", {}).get("sampling_steps", 32))
                ),
                "components": {"comfyui_root": str(paths["comfyui_root"]), "variant": "FL2VA"},
                "algorithm_state": {
                    "operator": paths["operator"],
                    "trainable_scope": "heads",
                    "training_steps": paths["max_steps"],
                    **(
                        {
                            "source_steps": paths["source_steps"],
                            "target_steps": paths["target_steps"],
                            "distillation": "binary_teacher_trajectory",
                        }
                        if paths["operator"] == "step_distill"
                        else {
                            "distillation": "dmd2_distribution_matching",
                            "teacher_checkpoint_sha256": parent_hash,
                            "critic": "latent_1x1_modality_projections",
                            "generator_update_interval": paths["generator_update_interval"],
                            "critic_updates": critic_updates,
                            "student_updates": student_updates,
                            "experimental": True,
                        }
                        if paths["operator"] == "dmd2"
                        else {
                            "distillation": "frozen_source_teacher_output",
                            "dataset_fraction": paths["dataset_fraction"],
                            "teacher_checkpoint_sha256": teacher_hash,
                            "teacher_checkpoint_path": str(paths["source_parent_checkpoint"]),
                        }
                        if paths["operator"] == "distill"
                        else {}
                    ),
                },
                "runtime_state": {"fsdp": "full_shard", "world_size": world_size},
                "measured_metrics": {
                    "quality_score": None,
                    "latency_s": None,
                    "peak_memory_gb": max(evidence["peak_vram_gb_per_rank"].values()),
                    "model_size_gb": float(child.stat().st_size / (1024**3)),
                    "energy_j": None,
                    "quality_measured": False,
                    "metrics_stale": True,
                },
                "provenance": {
                    "real_worker": True,
                    "offline_simulation": False,
                    "parent_sha256": parent_hash,
                    "evidence": str(child) + ".evidence.json",
                },
                "warnings": [
                    "A1-T0 smoke cache uses deterministic latent inputs; no quality claim."
                    if paths["operator"] == "recovery_finetune"
                    else (
                        "Output distillation uses a frozen source-teacher prediction on the deterministic cache; no quality claim."
                        if paths["operator"] == "distill"
                        else (
                            "DMD2 uses a frozen FSDP teacher and a trainable latent critic on the deterministic cache; benchmark required."
                            if paths["operator"] == "dmd2"
                            else "Progressive distillation uses a deterministic cached teacher trajectory; no quality claim."
                        )
                    )
                ],
            }
        evidence_path = child.with_suffix(child.suffix + ".evidence.json")
        _record_sha256_cache(child, child_hash, paths["parent_hash_cache"])
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "output_state": state,
                    "metrics": {**evidence, "child_evidence_manifest": str(evidence_path)},
                    "cost": {
                        "wall_time_s": time.monotonic() - started,
                        "gpu_hours": world_size * gpu_time_s / 3600.0,
                        "controller_calls": 0,
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(evidence, sort_keys=True), flush=True)
        dist.destroy_process_group()
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        print("[rank %d] CUDA OOM: %s" % (rank, exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        if rank == 0 and partial is not None:
            try:
                partial.unlink()
            except OSError:
                pass
        if distributed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        if rank == 0:
            _write_failure(result_path, "training_oom", str(exc), time.monotonic() - started)
        return 1
    except KeyboardInterrupt as exc:
        print("[rank %d] worker interrupted: %s" % (rank, exc), file=sys.stderr, flush=True)
        if rank == 0 and partial is not None:
            try:
                partial.unlink()
            except OSError:
                pass
        if distributed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        if rank == 0:
            _write_failure(result_path, "training_interrupted", str(exc), time.monotonic() - started)
        return 1
    except (FileExistsError, FloatingPointError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print("[rank %d] worker failure: %s" % (rank, exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        if rank == 0 and partial is not None:
            try:
                partial.unlink()
            except OSError:
                pass
        if distributed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        if rank == 0:
            message = str(exc)
            _write_failure(result_path, _classify_failure(message), message, time.monotonic() - started)
        return 1
    finally:
        # SIGKILL cannot run Python cleanup, but normal exceptions and handled
        # SIGTERM/SIGINT do reach here.  Never leave a failed partial file
        # behind when the final child was not published.
        if rank == 0 and partial is not None and partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass


def _gather_int(value: int, device: torch.device) -> list[int]:
    values = [torch.zeros((), device=device, dtype=torch.int64) for _ in range(dist.get_world_size())]
    dist.all_gather(values, torch.tensor(value, device=device, dtype=torch.int64))
    return [int(item.item()) for item in values]


def _gather_float(value: float, device: torch.device) -> list[float]:
    values = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(dist.get_world_size())]
    dist.all_gather(values, torch.tensor(value, device=device, dtype=torch.float64))
    return [float(item.item()) for item in values]


@record
def main(argv: Optional[list[str]] = None) -> int:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    request = _json(args.request)
    config = _json(args.config)
    return _run(request, config, args.result)


if __name__ == "__main__":
    raise SystemExit(main())
