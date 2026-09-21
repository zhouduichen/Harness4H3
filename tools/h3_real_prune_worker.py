#!/usr/bin/env python3
"""Trusted structural MiniMax-H3 block-pruning worker.

The worker is intentionally single-process: pruning changes checkpoint
structure and does not need four-way FSDP.  It removes complete transformer
blocks, rewrites the checkpoint metadata's ``num_layers``, and reloads the
child through the real ComfyUI MiniMax-H3 loader before publishing it.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.(.+)$")


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON object required: %s" % path)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config(checkpoint: Path) -> tuple[dict[str, Any], dict[str, str]]:
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw = metadata.get("config")
    if not raw:
        raise ValueError("parent checkpoint has no ComfyUI config metadata")
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("transformer"), Mapping):
        raise ValueError("parent checkpoint metadata has no transformer config")
    return dict(parsed), {str(key): str(value) for key, value in metadata.items()}


def _importance(state: Mapping[str, torch.Tensor], index: int) -> float:
    """Cheap deterministic block score using scale/norm parameters.

    This avoids converting the full 66 GiB checkpoint to float32 merely to
    rank blocks.  The score is a pruning heuristic, not a semantic quality
    measurement; the benchmark decides whether the resulting child survives.
    """
    suffixes = (
        "adaln_proj.linear.bias",
        "norm1.weight",
        "norm2.weight",
        "attn.q_norm.weight",
        "attn.k_norm.weight",
    )
    values = []
    for suffix in suffixes:
        tensor = state.get("blocks.%d.%s" % (index, suffix))
        if tensor is not None:
            value = float(tensor.float().abs().mean().item())
            if not math.isfinite(value):
                raise ValueError("non-finite block importance: %d" % index)
            values.append(value)
    if not values:
        raise ValueError("block %d has no importance tensors" % index)
    return sum(values) / len(values)


def _load_api(root: Path):
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    import comfy.model_management
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model

    return {
        "model_management": comfy.model_management,
        "ops": comfy.ops,
        "model": MiniMaxH3Model,
        "QuantizedTensor": comfy.ops.QuantizedTensor,
    }


def _materialize_quantized_weights(model: torch.nn.Module, api: Mapping[str, Any]) -> int:
    """Load ComfyUI INT8/ConvRot weights, then validate the child as BF16."""

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


def _reload(api: Mapping[str, Any], checkpoint: Path) -> int:
    config, _ = _config(checkpoint)
    transformer = config["transformer"]
    mixed_precision_factory = getattr(api["ops"], "mixed_precision_ops", None)
    operations = (
        mixed_precision_factory(compute_dtype=torch.bfloat16, full_precision_mm=True)
        if callable(mixed_precision_factory)
        else api["ops"].disable_weight_init
    )
    with torch.device("cpu"):
        model = api["model"](
            **transformer,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            operations=operations,
        )
    state = load_file(str(checkpoint), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError("child/model key mismatch; missing=%s unexpected=%s" % (missing[:3], unexpected[:3]))
    _materialize_quantized_weights(model, api)
    if len(model.blocks) != int(transformer["num_layers"]):
        raise RuntimeError("reloaded child has incorrect block count")
    count = len(state)
    del state, model
    gc.collect()
    return count


def _run(request: Mapping[str, Any], config: Mapping[str, Any], result_path: Path) -> int:
    started = time.monotonic()
    try:
        operator = str(request.get("operator", ""))
        if operator != "prune_blocks":
            raise ValueError("unsupported_pruning_operator: %s" % operator)
        parent_value = request.get("parent", {}).get("checkpoint_path")
        if not isinstance(parent_value, str) or not parent_value:
            raise ValueError("request parent checkpoint is required")
        parent = Path(parent_value).resolve()
        if not parent.is_file() or parent.suffix.lower() != ".safetensors":
            raise ValueError("parent H3 checkpoint does not exist: %s" % parent)
        artifacts_value = request.get("artifacts_dir")
        if not isinstance(artifacts_value, str) or not artifacts_value:
            raise ValueError("request artifacts_dir is required")
        output_dir = Path(artifacts_value).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        child = output_dir / (str(request["child_model_id"]) + ".safetensors")
        partial = child.with_name(child.name + ".part")
        if child.exists() or partial.exists():
            raise FileExistsError("child checkpoint already exists: %s" % child)
        args = request.get("operator_args")
        if not isinstance(args, Mapping):
            raise ValueError("operator_args must be a mapping")
        ratio = float(args.get("ratio", 0.0))
        if not math.isfinite(ratio) or not 0.01 <= ratio <= 0.8:
            raise ValueError("prune_blocks.ratio must be in [0.01, 0.8]")
        parent_hash = _sha256(parent)
        config_raw, metadata = _config(parent)
        transformer = dict(config_raw["transformer"])
        layer_count = int(transformer.get("num_layers", 0))
        if layer_count <= 1:
            raise ValueError("parent must contain at least two transformer blocks")
        state = load_file(str(parent), device="cpu")
        block_indices = sorted({int(match.group(1)) for name in state for match in [BLOCK_KEY.match(name)] if match})
        if block_indices != list(range(layer_count)):
            raise ValueError("checkpoint block index set does not match metadata")
        remove_count = min(layer_count - 1, max(1, int(math.ceil(layer_count * ratio))))
        scores = {index: _importance(state, index) for index in block_indices}
        removed = sorted(block_indices, key=lambda index: (scores[index], index))[:remove_count]
        removed_set = set(removed)
        kept = [index for index in block_indices if index not in removed_set]
        remap = {old: new for new, old in enumerate(kept)}
        child_state = {}
        for name, tensor in state.items():
            match = BLOCK_KEY.match(name)
            if match is None:
                child_state[name] = tensor
                continue
            old = int(match.group(1))
            if old in removed_set:
                continue
            child_state["blocks.%d.%s" % (remap[old], match.group(2))] = tensor
        transformer["num_layers"] = len(kept)
        config_raw["transformer"] = transformer
        metadata["config"] = json.dumps(config_raw, ensure_ascii=False, separators=(",", ":"))
        has_quantized_sidecars = any(
            name.endswith(".comfy_quant") or name.endswith(".weight_scale")
            for name in child_state
        )
        child_quantization = (
            {"bits": 8, "scheme": "int8_tensorwise+convrot"}
            if has_quantized_sidecars
            else {"bits": 16, "scheme": "none"}
        )
        save_file(child_state, str(partial), metadata=metadata)
        del state, child_state
        gc.collect()
        child_hash = _sha256(partial)
        if child_hash == parent_hash:
            raise RuntimeError("child hash equals parent hash")
        api = _load_api(Path(str(config["comfyui_root"])).resolve())
        reloaded_tensor_count = _reload(api, partial)
        partial.replace(child)
        parent_parameters = sum(int(value.numel()) for value in load_file(str(parent), device="cpu").values())
        child_parameters = sum(int(value.numel()) for value in load_file(str(child), device="cpu").values())
        evidence = {
            "operator": operator,
            "pruning_method": "magnitude_structured_block",
            "ratio_requested": ratio,
            "removed_block_indices": removed,
            "kept_block_indices": kept,
            "parent_num_blocks": layer_count,
            "child_num_blocks": len(kept),
            "parent_parameter_count": parent_parameters,
            "child_parameter_count": child_parameters,
            "removed_parameter_count": parent_parameters - child_parameters,
            "structural_change": True,
            "hardware_effect_expected": True,
            "parent_sha256": parent_hash,
            "parent_sha256_before": parent_hash,
            "parent_sha256_after": _sha256(parent),
            "child_sha256": child_hash,
            "child_reloaded": True,
            "reloaded_tensor_count": reloaded_tensor_count,
            "real_worker": True,
            "offline_simulation": False,
        }
        child_evidence = child.with_suffix(child.suffix + ".evidence.json")
        _write_json(child_evidence, evidence)
        state = {
            "model_id": str(request["child_model_id"]),
            "parent_model_id": str(request["parent"].get("model_id") or request["parent"].get("id")),
            "checkpoint_path": str(child),
            "architecture_name": "MiniMax-H3-FL2VA",
            "parameter_count": child_parameters,
            "trainable_parameter_count": 0,
            "num_blocks": len(kept),
            "hidden_size": int(transformer.get("hidden_size", 5376)),
            "num_attention_heads": int(transformer.get("num_attention_heads", 56)),
            "ffn_width": int(transformer.get("ffn_hidden_size", 14336)),
            "dtype": "mixed_quantized" if has_quantized_sidecars else "bfloat16",
            "quantization": child_quantization,
            "sampling_steps": int(request.get("parent", {}).get("state", {}).get("sampling_steps", 32)),
            "components": {"comfyui_root": str(config["comfyui_root"]), "variant": "FL2VA"},
            "algorithm_state": {
                "operator": operator,
                "structured_prune": "magnitude_structured_block",
                "prune_ratio": ratio,
                "removed_block_indices": removed,
                "kept_block_indices": kept,
            },
            "runtime_state": {"structural_pruning": True},
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
                "evidence": str(child_evidence),
            },
            "warnings": [
                "Block importance is a structural heuristic; semantic quality is decided only by the benchmark."
            ],
        }
        _write_json(
            result_path,
            {
                "status": "success",
                "output_state": state,
                "metrics": {**evidence, "child_evidence_manifest": str(child_evidence)},
                "cost": {"wall_time_s": time.monotonic() - started, "gpu_hours": 0.0, "controller_calls": 0},
            },
        )
        return 0
    except (FileExistsError, KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        _write_json(
            result_path,
            {
                "status": "failed",
                "failure_type": "pruning_worker_failure",
                "message": str(exc),
                "cost": {"wall_time_s": time.monotonic() - started, "gpu_hours": 0.0, "controller_calls": 0},
                "metrics": {"real_worker": True, "offline_simulation": False},
            },
        )
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    return _run(_json(args.request), _json(args.config), args.result)


if __name__ == "__main__":
    raise SystemExit(main())
