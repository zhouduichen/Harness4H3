#!/usr/bin/env python3
"""Adopt a trusted, prebuilt MiniMax-H3 quantized checkpoint.

Quantized H3 tensors are not ordinary trainable PyTorch parameters.  This
worker therefore does not perform an ad-hoc conversion.  It selects a
pre-registered device-side artifact, verifies that it is the matching H3
architecture and quantized format, copies it into the isolated experiment
artifact directory, and leaves semantic/hardware validation to the
independent benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from safetensors import safe_open


FAILURE_TYPES = frozenset(
    {
        "invalid_quantization_config",
        "unsupported_quantization_variant",
        "checkpoint_corrupt",
        "parent_modified",
        "source_modified",
        "child_reload_failed",
    }
)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON object required: %s" % path)
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_metadata(path: Path) -> tuple[dict[str, Any], int, int, str, bool]:
    """Read only metadata/tensor descriptors; do not materialize 34 GB of weights."""

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = list(handle.keys())
        parameter_count = 0
        block_indices = set()
        has_quantized_payload = False
        for name in keys:
            tensor = handle.get_slice(name)
            shape = tuple(int(value) for value in tensor.get_shape())
            parameter_count += math.prod(shape)
            match = re.match(r"^blocks\.(\d+)\.", name)
            if match:
                block_indices.add(int(match.group(1)))
            if name.endswith(".comfy_quant") or name.endswith(".weight_scale"):
                has_quantized_payload = True
            # The actual weight records in the server's INT8 variant are I8;
            # the companion ``.comfy_quant``/``.weight_scale`` descriptors are
            # enough to prove that without materializing any weight payload.
    raw_config = metadata.get("config")
    if not isinstance(raw_config, str) or not raw_config:
        raise ValueError("quantized checkpoint has no ComfyUI config metadata")
    config = json.loads(raw_config)
    if not isinstance(config, Mapping) or not isinstance(config.get("transformer"), Mapping):
        raise ValueError("quantized checkpoint metadata has no transformer config")
    return dict(config), parameter_count, (max(block_indices) + 1 if block_indices else 0), str(raw_config), has_quantized_payload


def validate_config(config: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    operator = str(request.get("operator", ""))
    if operator != "quantize":
        raise ValueError("unsupported_quantization_variant: %s" % operator)
    parent_raw = request.get("parent")
    if not isinstance(parent_raw, Mapping):
        raise ValueError("parent is required")
    parent = Path(str(parent_raw.get("checkpoint_path", ""))).resolve()
    if not parent.is_file() or parent.suffix.lower() != ".safetensors":
        raise ValueError("parent H3 safetensors does not exist: %s" % parent)
    artifacts = request.get("artifacts_dir")
    if not isinstance(artifacts, str) or not artifacts.strip():
        raise ValueError("request artifacts_dir is required")
    output_dir = Path(artifacts).resolve()
    bits = request.get("operator_args", {}).get("bits") if isinstance(request.get("operator_args"), Mapping) else None
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in {4, 8}:
        raise ValueError("quantize.bits must be 4 or 8")
    variants = config.get("quantized_variants")
    if not isinstance(variants, Mapping):
        raise ValueError("quantized_variants must be configured by the trusted worker")
    source_value = variants.get(bits, variants.get(str(bits)))
    if not isinstance(source_value, str) or not source_value.strip():
        raise ValueError("unsupported_quantization_variant: no configured %d-bit checkpoint" % bits)
    source = Path(source_value).resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise ValueError("configured quantized checkpoint does not exist: %s" % source)
    if source == parent:
        raise ValueError("quantized checkpoint must differ from parent")
    child_id = str(request.get("child_model_id", "")).strip()
    if not child_id or "/" in child_id or "\\" in child_id:
        raise ValueError("child_model_id is invalid")
    return {
        "parent": parent,
        "source": source,
        "output_dir": output_dir,
        "bits": bits,
        "child_id": child_id,
    }


def _failure(result_path: Path, failure_type: str, message: str, wall_time_s: float) -> int:
    if failure_type not in FAILURE_TYPES:
        failure_type = "invalid_quantization_config"
    _write(
        result_path,
        {
            "status": "failed",
            "failure_type": failure_type,
            "message": message,
            "cost": {"wall_time_s": float(wall_time_s), "gpu_hours": 0.0, "controller_calls": 0},
            "metrics": {"real_worker": True, "offline_simulation": False},
        },
    )
    return 1


def _run(request: Mapping[str, Any], config: Mapping[str, Any], result_path: Path) -> int:
    started = time.monotonic()
    try:
        paths = validate_config(config, request)
        parent = paths["parent"]
        source = paths["source"]
        bits = int(paths["bits"])
        output_dir = paths["output_dir"]
        child = output_dir / (paths["child_id"] + ".safetensors")
        partial = child.with_name(child.name + ".part")
        if child.exists() or partial.exists():
            raise FileExistsError("child checkpoint already exists: %s" % child)

        parent_hash = _sha256(parent)
        source_hash = _sha256(source)
        parent_config, _, parent_blocks, _, _ = _checkpoint_metadata(parent)
        source_config, source_parameter_count, source_blocks, _, source_is_quantized = _checkpoint_metadata(source)
        if parent_config.get("transformer") != source_config.get("transformer"):
            raise ValueError("configured quantized checkpoint architecture does not match parent")
        if source_blocks != int(source_config["transformer"].get("num_layers", source_blocks)):
            raise ValueError("configured quantized checkpoint has inconsistent block metadata")
        if parent_blocks != int(parent_config["transformer"].get("num_layers", parent_blocks)):
            raise ValueError("parent checkpoint has inconsistent block metadata")
        if not source_is_quantized:
            raise ValueError("configured source is not a quantized H3 checkpoint")

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, partial)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        child_hash = _sha256(partial)
        if child_hash != source_hash:
            raise RuntimeError("copied child hash does not match trusted source")
        if _sha256(parent) != parent_hash:
            raise RuntimeError("parent checkpoint hash changed during quantization")
        partial.replace(child)

        evidence = {
            "operator": "quantize",
            "quantization_bits": bits,
            "quantization_method": "trusted_prebuilt_variant",
            "source_checkpoint": str(source),
            "source_sha256": source_hash,
            "parent_sha256": parent_hash,
            "parent_sha256_before": parent_hash,
            "parent_sha256_after": _sha256(parent),
            "child_sha256": child_hash,
            "child_size_bytes": child.stat().st_size,
            "source_is_quantized": True,
            "child_copy_verified": True,
            "benchmark_reload_pending": True,
            "real_worker": True,
            "offline_simulation": False,
        }
        evidence_path = child.with_suffix(child.suffix + ".evidence.json")
        _write(evidence_path, evidence)
        transformer = dict(source_config["transformer"])
        state = {
            "model_id": paths["child_id"],
            "parent_model_id": str(request["parent"].get("model_id") or request["parent"].get("id")),
            "checkpoint_path": str(child),
            "architecture_name": "MiniMax-H3-FL2VA",
            "parameter_count": int(source_parameter_count),
            "trainable_parameter_count": 0,
            "num_blocks": int(transformer.get("num_layers", source_blocks)),
            "hidden_size": int(transformer.get("hidden_size", 0)) or None,
            "num_attention_heads": int(transformer.get("num_attention_heads", 0)) or None,
            "ffn_width": int(transformer.get("ffn_hidden_size", 0)) or None,
            "dtype": "mixed_quantized",
            "quantization": {"bits": bits, "scheme": source.stem},
            "sampling_steps": int(request.get("parent", {}).get("state", {}).get("sampling_steps", 32)),
            "components": {"comfyui_root": str(config.get("comfyui_root", "")), "variant": "FL2VA"},
            "algorithm_state": {
                "operator": "quantize",
                "quantization_bits": bits,
                "quantization_method": "trusted_prebuilt_variant",
                "source_checkpoint": str(source),
                "source_sha256": source_hash,
            },
            "runtime_state": {"operator": "quantize", "weights_loaded": False, "metrics_stale": True},
            "measured_metrics": {
                "quality_score": None,
                "latency_s": None,
                "peak_memory_gb": None,
                "model_size_gb": float(child.stat().st_size / (1024 ** 3)),
                "energy_j": None,
                "quality_measured": False,
                "metrics_stale": True,
            },
            "provenance": {
                "real_worker": True,
                "offline_simulation": False,
                "parent_sha256": parent_hash,
                "source_sha256": source_hash,
                "evidence": str(evidence_path),
            },
            "warnings": [
                "The quantized child is a trusted prebuilt variant; benchmark reload and all quality/hardware metrics are pending."
            ],
        }
        _write(
            result_path,
            {
                "status": "success",
                "output_state": state,
                "metrics": {**evidence, "child_evidence_manifest": str(evidence_path)},
                "cost": {"wall_time_s": time.monotonic() - started, "gpu_hours": 0.0, "controller_calls": 0},
            },
        )
        return 0
    except FileExistsError as exc:
        return _failure(result_path, "invalid_quantization_config", str(exc), time.monotonic() - started)
    except (OSError, TypeError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        message = str(exc)
        failure = "parent_modified" if "parent checkpoint hash changed" in message else "checkpoint_corrupt"
        if "source" in message and "hash" in message:
            failure = "source_modified"
        return _failure(result_path, failure, message, time.monotonic() - started)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="trusted MiniMax-H3 prebuilt quantization worker")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    return _run(_json(args.request), _json(args.config), args.result)


if __name__ == "__main__":
    raise SystemExit(main())
