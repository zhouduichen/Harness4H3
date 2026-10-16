from __future__ import annotations

import hashlib
import json
import os
import shutil
from types import MethodType
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .contracts import BlockRef, CheckpointArtifact, ChannelPruningConfig, ModelStructure, StructuralChange
from .attention import prune_attention_module_heads
from ...operators.wan.token_geometry import merge_token_sequence
from ...operators.wan.runtime_gating import TimestepAttentionRule, normalized_timesteps, rule_matches, rules_to_dict, validate_timestep_attention_plan, validate_timestep_gate_plan
from ...operators.wan.runtime_policy import AttentionWindowPolicy, KVTokenSelectionPolicy, SamplingStepPolicy, SparseRuntimePolicy, TokenMergePolicy, TokenSparsityPolicy, select_sampling_steps


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional deployment dependency
        raise RuntimeError("Diffusers Wan adapter requires torch") from exc
    return torch, nn


def _import_pipeline() -> Any:
    try:
        from diffusers import DiffusionPipeline
    except ImportError as exc:  # pragma: no cover - optional deployment dependency
        raise RuntimeError("Diffusers Wan adapter requires diffusers") from exc
    return DiffusionPipeline


def _param_count(module: Any) -> int:
    return sum(int(item.numel()) for item in module.parameters())


def _transformer_checkpoint_sha256(path: str | Path) -> str:
    """Hash the actual Wan Transformer bytes and its pipeline index.

    The VAE and text encoder are unchanged by EvoCompress.  Including the
    Transformer files plus ``model_index.json`` gives an honest model-content
    identity while avoiding a repeated 27GB scan of unrelated components on a
    network filesystem.  The scope is recorded in artifact metadata.
    """

    root = Path(path)
    candidates = []
    index = root / "model_index.json"
    if index.is_file():
        candidates.append(index)
    transformer = root / "transformer"
    if transformer.is_dir():
        candidates.extend(item for item in transformer.rglob("*") if item.is_file())
    if not candidates:
        raise ValueError("Wan checkpoint has no model_index/transformer content: %s" % root)
    digest = hashlib.sha256()
    for item in sorted(candidates, key=lambda value: value.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _module_at(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part) if not part.isdigit() else value[int(part)]
    return value


def _register_config(module: Any, key: str, value: Any) -> None:
    register = getattr(module, "register_to_config", None)
    if callable(register):
        register(**{key: value})
    else:
        setattr(getattr(module, "config", module), key, value)


def _read_runtime_config(checkpoint: str | Path) -> Mapping[str, Any]:
    """Read EvoCompress runtime metadata from the raw Wan config JSON.

    Diffusers intentionally drops unknown config attributes while constructing
    a model.  Runtime policies are still persisted in the checkpoint, so
    replay must read the source JSON rather than trusting ``transformer.config``.
    """

    path = Path(checkpoint) / "transformer" / "config.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_runtime_config: %s" % exc) from exc
    if not isinstance(value, Mapping):
        raise ValueError("invalid_runtime_config: expected JSON object")
    return value


def _record_runtime_operation(module: Any, operator: str, args: Mapping[str, Any]) -> None:
    """Persist the exact runtime/sampler application order in the transformer config."""

    entry = {"operator": str(operator), "args": dict(args)}
    existing = getattr(module, "_harness4h3_runtime_operations", None)
    if not isinstance(existing, list):
        config_value = getattr(getattr(module, "config", module), "harness4h3_runtime_operations", None)
        existing = list(config_value) if isinstance(config_value, list) else []
    existing.append(entry)
    module._harness4h3_runtime_operations = existing
    _register_config(module, "harness4h3_runtime_operations", existing)


def _slice_rotary(rotary: Any, indices: Any, sequence_length: int) -> Any:
    """Select matching positions from Wan's (cos, sin) rotary tuple."""

    if rotary is None:
        return None
    result = []
    for value in rotary:
        selected = value
        try:
            for dimension, size in enumerate(tuple(value.shape)):
                if int(size) == int(sequence_length):
                    selected = value.index_select(dimension, indices.to(value.device))
                    break
        except (AttributeError, IndexError, RuntimeError, TypeError):
            selected = value
        result.append(selected)
    return tuple(result)


def _merge_token_sequence(hidden_states: Any, keep_ratio: float, torch: Any, grid_shape: Sequence[int] = ()) -> tuple[Any, Any, Any, tuple[int, int, int] | None]:
    """Merge token groups and return (merged, source_groups, anchors, group_shape).

    Adjacent mode uses the historical 1-D TokenMap.  Spatiotemporal mode uses
    a deterministic ``(T, H, W)`` cuboid map.  Both keep the first token in
    every group as the positional/RoPE anchor and average only pre-RoPE hidden
    states.  ``source_groups`` is the unmerge map from every original position
    to its merged destination.
    """

    return merge_token_sequence(hidden_states, keep_ratio, torch, grid_shape=grid_shape)


def _select_kv_indices(hidden_states: Any, keep_ratio: float, score: str, torch: Any) -> Any:
    """Select one shared K/V index set from a batch for RoPE-compatible gather."""

    count = max(1, min(int(hidden_states.shape[1]), int(round(int(hidden_states.shape[1]) * float(keep_ratio)))))
    values = hidden_states.detach().float()
    scores = values.pow(2).mean(dim=(0, 2)) if score == "l2_norm" else values.abs().mean(dim=(0, 2))
    return torch.topk(scores, count).indices.sort().values


def _apply_wan_rotary(hidden_states: Any, rotary_emb: Any) -> Any:
    """Apply Wan's real-valued RoPE tuple to Q/K tensors."""

    if rotary_emb is None:
        return hidden_states
    freqs_cos, freqs_sin = rotary_emb
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = hidden_states.new_empty(hidden_states.shape)
    output[..., 0::2] = x1 * cos - x2 * sin
    output[..., 1::2] = x1 * sin + x2 * cos
    return output.type_as(hidden_states)


def _wan_sparse_self_attention(
    attention: Any,
    hidden_states: Any,
    keys: Any,
    indices: Any,
    rotary_emb: Any,
    torch: Any,
    *,
    query_indices: Any = None,
    sequence_length: Optional[int] = None,
) -> Any:
    """Run Wan self-attention with full Q and selected K/V tokens.

    The stock Wan processor accepts one RoPE tuple for both Q and K.  This
    small adapter keeps its Q/K normalization and output projection while
    slicing K's positions independently, so reduced K/V work does not drop
    positional semantics.
    """

    fused = bool(getattr(attention, "fused_projections", False))
    if fused:
        query = attention.to_qkv(hidden_states).chunk(3, dim=-1)[0]
        key, value = attention.to_qkv(keys).chunk(3, dim=-1)[1:]
    else:
        query = attention.to_q(hidden_states)
        key = attention.to_k(keys)
        value = attention.to_v(keys)
    norm_q = getattr(attention, "norm_q", None)
    norm_k = getattr(attention, "norm_k", None)
    if callable(norm_q):
        query = norm_q(query)
    if callable(norm_k):
        key = norm_k(key)
    heads = int(attention.heads)
    query = query.unflatten(2, (heads, -1))
    key = key.unflatten(2, (heads, -1))
    value = value.unflatten(2, (heads, -1))
    total_length = int(sequence_length or hidden_states.shape[1])
    query_rotary = _slice_rotary(rotary_emb, query_indices, total_length) if query_indices is not None else rotary_emb
    query = _apply_wan_rotary(query, query_rotary)
    key = _apply_wan_rotary(key, _slice_rotary(rotary_emb, indices, total_length))
    output = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0, is_causal=False
    ).transpose(1, 2)
    output = output.flatten(2, 3).type_as(query)
    output = attention.to_out[0](output)
    return attention.to_out[1](output)


def _copy_shared_components(source: Path, target: Path) -> bool:
    """Materialize an unchanged pipeline shell without copying its weights.

    EvoCompress only mutates ``transformer``.  Child checkpoints remain
    reloadable because unchanged component files are hard-linked when possible
    and otherwise reference the immutable local HF snapshot with absolute
    symlinks.  The sharing policy is recorded in artifact metadata.
    """

    if not source.is_dir() or not (source / "model_index.json").is_file():
        return False
    try:
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns("transformer"),
            copy_function=_hardlink_or_external_symlink,
        )
    except (OSError, shutil.Error):
        if target.exists():
            shutil.rmtree(target)
        return False
    return True


def _hardlink_or_external_symlink(source: str | Path, target: str | Path) -> None:
    """Share files without copying immutable HF components across filesystems."""

    source_path = Path(source)
    target_path = Path(target)
    if source_path.is_symlink():
        target_path.symlink_to(source_path.resolve(strict=True))
        return
    try:
        os.link(source_path, target_path)
    except OSError:
        # Parent checkpoints may live on /home while experiment artifacts use
        # /data.  The source is an immutable local HF snapshot, so an absolute
        # reference is reloadable on this host and avoids copying many GB.
        target_path.symlink_to(source_path.resolve(strict=True))


def _copy_full_checkpoint(source: Path, target: Path) -> bool:
    """Copy a complete pipeline when hardlinks cannot cross filesystems."""

    if not source.is_dir() or not (source / "model_index.json").is_file():
        return False
    try:
        shutil.copytree(source, target, copy_function=shutil.copy2)
    except (OSError, shutil.Error):
        if target.exists():
            shutil.rmtree(target)
        return False
    return True


class DiffusersWanAdapter:
    """Best-effort structural boundary for real Wan Diffusers checkpoints.

    Diffusers has changed Wan internals across releases.  This adapter probes
    known module-list names and refuses unknown layouts instead of silently
    applying a partial mutation.
    """

    architecture_name = "WanDiffusersDiT"
    # Wan reloadable checkpoints require the FFN projection width to stay
    # aligned across every remaining transformer block.  Composite plans may
    # therefore normalize a channel step to all current positional blocks.
    channel_pruning_requires_all_blocks = True
    _block_paths = ("transformer_blocks", "blocks", "single_blocks")

    def __init__(self, local_files_only: bool = True) -> None:
        self.local_files_only = bool(local_files_only)
        self._device = "cpu"
        self._dtype = "float32"

    @staticmethod
    def preflight_checkpoint(checkpoint: str | Path) -> Mapping[str, Any]:
        """Check a Wan checkpoint shell without loading weights or CUDA.

        This is intentionally a file/layout check only.  The returned
        ``runtime_probe_required`` flag stays true until a GPU host calls
        :meth:`preflight_loaded_model` on the actually loaded Diffusers
        pipeline.
        """

        root = Path(checkpoint).expanduser()
        if not root.is_dir():
            raise ValueError("wan_preflight_checkpoint_missing: %s" % root)
        index = root / "model_index.json"
        transformer = root / "transformer"
        config_path = transformer / "config.json"
        missing = [str(path.relative_to(root)) for path in (index, transformer, config_path) if not path.exists()]
        if missing:
            raise ValueError("wan_preflight_layout_missing: %s" % ", ".join(missing))
        try:
            model_index = json.loads(index.read_text(encoding="utf-8"))
            transformer_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("wan_preflight_metadata_invalid: %s" % exc) from exc
        if not isinstance(model_index, Mapping) or not isinstance(transformer_config, Mapping):
            raise ValueError("wan_preflight_metadata_invalid: expected JSON objects")
        return {
            "valid": True,
            "checkpoint": str(root),
            "model_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            "transformer_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "transformer_config": {
                key: transformer_config.get(key)
                for key in ("num_layers", "num_single_layers", "hidden_size", "inner_dim", "ffn_dim", "in_channels", "text_dim")
                if key in transformer_config
            },
            "runtime_probe_required": True,
            "cuda_loaded": False,
        }

    def preflight_loaded_model(self, model: Any) -> Mapping[str, Any]:
        """Validate every real FFN path and global channel shape after load."""

        path, blocks = self._blocks(model)
        widths: list[int] = []
        activation_paths: list[str] = []
        for index, block in enumerate(blocks):
            components = self.ffn_components(block)
            if components is None:
                raise ValueError("wan_preflight_ffn_path_unavailable: block.%d" % index)
            up, activation, down = components
            width = int(getattr(up, "out_features", 0))
            if width <= 1 or int(getattr(down, "in_features", 0)) != width:
                raise ValueError("wan_preflight_ffn_width_mismatch: block.%d" % index)
            if not callable(getattr(activation, "register_forward_hook", None)):
                raise ValueError("wan_preflight_activation_hook_unavailable: block.%d" % index)
            widths.append(width)
            activation_paths.append(type(activation).__name__)
        if len(set(widths)) != 1:
            raise ValueError("wan_preflight_global_ffn_alignment_failed")
        return {
            "valid": True,
            "block_path": path,
            "block_count": len(blocks),
            "ffn_width": widths[0],
            "ffn_widths": widths,
            "activation_modules": activation_paths,
            "runtime_probe_required": False,
            "cuda_loaded": str(self._device).startswith("cuda"),
        }

    def _replay_runtime_config(self, model: Any, config: Mapping[str, Any]) -> None:
        """Install the persisted runtime policy sequence on a loaded model."""

        transformer = self._transformer(model)
        runtime_operations = config.get("harness4h3_runtime_operations")
        if isinstance(runtime_operations, list):
            # Rebuild the in-memory record while replaying so loading does not
            # append a second copy of the persisted operation sequence.
            transformer._harness4h3_runtime_operations = []
            _register_config(transformer, "harness4h3_runtime_operations", [])
            for item in runtime_operations:
                if not isinstance(item, Mapping) or not isinstance(item.get("args", {}), Mapping):
                    raise ValueError("invalid_runtime_operation_manifest")
                operator = str(item.get("operator", ""))
                args = dict(item.get("args") or {})
                if operator == "timestep_block_gate":
                    self.apply_timestep_block_gating(model, args.get("rules", []))
                elif operator == "timestep_attention_policy":
                    self.apply_timestep_attention_policy(model, args)
                elif operator == "prune_tokens":
                    self.apply_token_sparsity(model, args)
                elif operator == "token_merge":
                    self.apply_token_merge(model, args)
                elif operator == "kv_token_select":
                    self.apply_kv_token_selection(model, args)
                elif operator == "sparse_runtime":
                    self.apply_sparse_runtime(model, args)
                elif operator == "attention_window":
                    self.apply_attention_window(model, args)
                elif operator == "sampling_step_skip":
                    self.apply_sampling_step_policy(model, args)
                else:
                    raise ValueError("unsupported_runtime_operation: %s" % operator)
            return

        # Legacy checkpoints predate the ordered manifest.  Their individual
        # config fields are replayed in the historical order.
        runtime_rules = config.get("harness4h3_timestep_block_gating")
        if runtime_rules:
            self.apply_timestep_block_gating(model, runtime_rules)
        runtime_attention_rules = config.get("harness4h3_timestep_attention_policy")
        if runtime_attention_rules:
            self.apply_timestep_attention_policy(model, runtime_attention_rules)
        token_policy = config.get("harness4h3_token_sparsity")
        if token_policy:
            self.apply_token_sparsity(model, token_policy)
        token_merge_policy = config.get("harness4h3_token_merge")
        if token_merge_policy:
            self.apply_token_merge(model, token_merge_policy)
        kv_policy = config.get("harness4h3_kv_token_selection")
        if kv_policy:
            self.apply_kv_token_selection(model, kv_policy)
        sparse_policy = config.get("harness4h3_sparse_runtime")
        if sparse_policy:
            self.apply_sparse_runtime(model, sparse_policy)
        window_policy = config.get("harness4h3_attention_window")
        if window_policy:
            self.apply_attention_window(model, window_policy)
        sampling_policy = config.get("harness4h3_sampling_step")
        if sampling_policy:
            self.apply_sampling_step_policy(model, sampling_policy)

    def replay_runtime_operations(self, model: Any, checkpoint: str | Path) -> Any:
        """Replay runtime policies using the checkpoint's raw manifest.

        This public hook is also used by the independent video evaluator,
        which loads a full Diffusers pipeline instead of calling ``load_model``.
        """

        config = _read_runtime_config(checkpoint)
        if config:
            self._replay_runtime_config(model, config)
        return model

    def load_model(self, checkpoint: str, *, device: Optional[str] = None, dtype: Optional[str] = None) -> Any:
        torch, _ = _torch_modules()
        pipeline_cls = _import_pipeline()
        device = str(self._device if device is None else device)
        dtype = str(self._dtype if dtype is None else dtype)
        pipe = pipeline_cls.from_pretrained(checkpoint, local_files_only=self.local_files_only, torch_dtype=getattr(torch, dtype))
        if not hasattr(pipe, "transformer"):
            raise ValueError("unsupported_model_layout: Wan pipeline has no transformer")
        self._replay_runtime_config(pipe, _read_runtime_config(checkpoint))
        # Structural exploration only needs the DiT transformer.  Moving the
        # T5 encoder and VAE as well can exceed one GPU and also triggers
        # allocator/NVML issues on otherwise usable multi-GPU hosts.
        pipe.transformer.to(device)
        pipe.transformer.eval()
        self._device = str(device)
        self._dtype = str(dtype)
        pipe._harness4h3_checkpoint_root = str(checkpoint)
        pipe._harness4h3_checkpoint_sha256 = _transformer_checkpoint_sha256(checkpoint)
        return pipe

    @classmethod
    def _transformer(cls, model: Any) -> Any:
        return getattr(model, "transformer", model)

    @classmethod
    def _blocks(cls, model: Any) -> tuple[str, Any]:
        transformer = cls._transformer(model)
        for path in cls._block_paths:
            blocks = getattr(transformer, path, None)
            if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
                return path, blocks
        raise ValueError("unsupported_model_layout: no recognized Wan block list")

    def inspect(self, model: Any) -> ModelStructure:
        path, blocks = self._blocks(model)
        transformer = self._transformer(model)
        hidden = int(getattr(transformer.config, "hidden_size", getattr(transformer, "inner_dim", 0)))
        if hidden <= 0:
            hidden = int(getattr(transformer.config, "num_attention_heads", 0)) * int(getattr(transformer.config, "attention_head_dim", 0))
        widths = [self._ffn_pair(block)[0].out_features for block in blocks if self._ffn_pair(block) is not None]
        if not widths or hidden <= 0:
            raise ValueError("unsupported_model_layout: no paired FFN projections")
        digest = getattr(model, "_harness4h3_checkpoint_sha256", None)
        if not digest:
            digest = hashlib.sha256(json.dumps(sorted(transformer.state_dict().keys())).encode("utf-8")).hexdigest()
        return ModelStructure(
            self.architecture_name,
            len(blocks),
            hidden,
            max(widths),
            _param_count(transformer),
            digest,
            {
                "block_path": path,
                "ffn_widths": widths,
                "offline_simulation": False,
                "profile_device": self._device,
                "profile_dtype": self._dtype,
                "token_sparsity_policy": dict(getattr(transformer, "_harness4h3_token_sparsity_policy", {}) or {}),
                "timestep_attention_policy": dict(getattr(transformer, "_harness4h3_timestep_attention_policy", {}) or {}),
                "timestep_attention_stats": dict(getattr(transformer, "_harness4h3_timestep_attention_stats", {}) or {}),
                "token_merge_policy": dict(getattr(transformer, "_harness4h3_token_merge_policy", {}) or {}),
                "kv_token_selection_policy": dict(getattr(transformer, "_harness4h3_kv_token_selection_policy", {}) or {}),
                "sparse_runtime_policy": dict(getattr(transformer, "_harness4h3_sparse_runtime_policy", {}) or {}),
                "attention_window_policy": dict(getattr(transformer, "_harness4h3_attention_window_policy", {}) or {}),
                "sampling_step_policy": dict(getattr(transformer, "_harness4h3_sampling_step_policy", {}) or {}),
                "runtime_operations": [dict(item) for item in getattr(transformer, "_harness4h3_runtime_operations", []) or []],
            },
        )

    def get_blocks(self, model: Any) -> tuple[BlockRef, ...]:
        path, blocks = self._blocks(model)
        prefix = "transformer." if hasattr(model, "transformer") else ""
        return tuple(BlockRef("block.%d" % index, index, _param_count(block), "%s%s.%d" % (prefix, path, index)) for index, block in enumerate(blocks))

    @staticmethod
    def ffn_components(block: Any) -> tuple[Any, Any, Any] | None:
        """Resolve the actual Wan FFN activation between paired projections."""

        candidates = (
            ("up_proj", "activation", "down_proj"),
            ("ffn.up_proj", "ffn.activation", "ffn.down_proj"),
            ("ffn.net.0", "ffn.net.1", "ffn.net.2"),
            ("ffn.net.0.proj", "ffn.net.1", "ffn.net.2"),
            ("ffn.proj_in", "ffn.activation", "ffn.proj_out"),
        )
        for up_path, activation_path, down_path in candidates:
            try:
                up = _module_at(block, up_path)
                activation = _module_at(block, activation_path)
                down = _module_at(block, down_path)
            except (AttributeError, IndexError, TypeError):
                continue
            if all(hasattr(item, attribute) for item in (up, down) for attribute in ("in_features", "out_features")):
                return up, activation, down
        return None

    @staticmethod
    def _ffn_pair(block: Any) -> tuple[Any, Any] | None:
        candidates = (("up_proj", "down_proj"), ("ffn.up_proj", "ffn.down_proj"), ("ffn.net.0", "ffn.net.2"), ("ffn.net.0.proj", "ffn.net.2"), ("ffn.proj_in", "ffn.proj_out"))
        for up_path, down_path in candidates:
            try:
                up = _module_at(block, up_path)
                down = _module_at(block, down_path)
            except (AttributeError, IndexError, TypeError):
                continue
            if hasattr(up, "in_features") and hasattr(up, "out_features") and hasattr(down, "in_features") and hasattr(down, "out_features"):
                return up, down
        return None

    def remove_blocks(self, model: Any, block_ids: Sequence[str]) -> StructuralChange:
        path, blocks = self._blocks(model)
        selected = tuple(str(item) for item in block_ids)
        valid = {"block.%d" % index for index in range(len(blocks))}
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= valid or len(selected) >= len(blocks):
            raise ValueError("invalid_block_selection")
        _, nn = _torch_modules()
        kept = [block for index, block in enumerate(blocks) if "block.%d" % index not in selected]
        transformer = self._transformer(model)
        setattr(transformer, path, nn.ModuleList(kept))
        _update_config(transformer, num_layers=len(kept))
        return StructuralChange("block_pruning", {"removed_block_ids": list(selected), "original_blocks": len(blocks), "retained_blocks": len(kept), "module_path": path})

    def prune_channels(self, model: Any, config: ChannelPruningConfig) -> StructuralChange:
        _, blocks = self._blocks(model)
        torch, nn = _torch_modules()
        if set(config.block_ids) != {"block.%d" % index for index in range(len(blocks))}:
            raise ValueError("unsupported_model_layout: channel pruning requires all Wan blocks for reloadable checkpoints")
        original: dict[str, int] = {}
        for index, block in enumerate(blocks):
            block_id = "block.%d" % index
            if block_id not in config.block_ids:
                continue
            pair = self._ffn_pair(block)
            if pair is None:
                raise ValueError("unsupported_model_layout: FFN projections are not paired linear layers")
            up, down = pair
            retained = tuple(config.retained_indices)
            if not retained or len(set(retained)) != len(retained) or len(retained) >= int(up.out_features) or any(item < 0 or item >= int(up.out_features) for item in retained):
                raise ValueError("invalid_channel_selection")
            original[block_id] = int(up.out_features)
            new_up = nn.Linear(int(up.in_features), len(retained), bias=up.bias is not None, device=up.weight.device, dtype=up.weight.dtype)
            new_down = nn.Linear(len(retained), int(down.out_features), bias=down.bias is not None, device=down.weight.device, dtype=down.weight.dtype)
            with torch.no_grad():
                indices = torch.tensor(retained, device=up.weight.device)
                new_up.weight.copy_(up.weight[indices])
                if new_up.bias is not None:
                    new_up.bias.copy_(up.bias[indices])
                new_down.weight.copy_(down.weight[:, indices])
                if new_down.bias is not None:
                    new_down.bias.copy_(down.bias)
            if hasattr(block, "up_proj"):
                block.up_proj, block.down_proj = new_up, new_down
            elif hasattr(block, "ffn") and hasattr(block.ffn, "up_proj"):
                block.ffn.up_proj, block.ffn.down_proj = new_up, new_down
            elif hasattr(block, "ffn") and hasattr(block.ffn, "net") and hasattr(block.ffn.net[0], "proj"):
                block.ffn.net[0].proj, block.ffn.net[2] = new_up, new_down
            else:
                raise ValueError("unsupported_model_layout: cannot replace FFN projections")
        _update_config(self._transformer(model), ffn_dim=len(config.retained_indices))
        return StructuralChange("ffn_channel_pruning", {"block_ids": list(config.block_ids), "retained_indices": list(config.retained_indices), "original_widths": original, "retained_width": len(config.retained_indices)})

    def prune_attention_heads(self, model: Any, removed_heads: Sequence[int], attention_names: Sequence[str] = ("attn1", "attn2")) -> StructuralChange:
        _path, blocks = self._blocks(model)
        names = tuple(str(item) for item in attention_names)
        if not names:
            raise ValueError("invalid_attention_selection")
        available_names = tuple(name for name in ("attn1", "attn2") if hasattr(blocks[0], name))
        if set(names) != set(available_names):
            raise ValueError("global_attention_alignment_required")
        before = _param_count(self._transformer(model))
        changes = []
        for index, block in enumerate(blocks):
            for name in names:
                attention = getattr(block, name, None)
                if attention is None:
                    raise ValueError("unsupported_attention_layout: %s" % name)
                changes.append({"block_id": "block.%d" % index, "attention": name, **prune_attention_module_heads(attention, removed_heads)})
        remaining_heads = int(getattr(getattr(blocks[0], names[0]), "heads", 0))
        _update_config(self._transformer(model), num_attention_heads=remaining_heads)
        after = _param_count(self._transformer(model))
        return StructuralChange("attention_head_pruning", {"attention_names": list(names), "head_indices": list(removed_heads), "changes": changes, "remaining_heads": remaining_heads, "parameter_reduction_ratio": 1.0 - float(after) / max(float(before), 1.0)})

    def apply_token_sparsity(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        torch, _ = _torch_modules()
        parsed = TokenSparsityPolicy.from_mapping(policy)
        _path, blocks = self._blocks(model)
        known = {"block.%d" % index for index in range(len(blocks))}
        unknown = sorted(set(parsed.block_ids) - known)
        if unknown:
            raise ValueError("invalid_token_selection: unknown block %s" % unknown[0])
        transformer = self._transformer(model)
        _register_config(transformer, "harness4h3_token_sparsity", parsed.to_dict())
        transformer._harness4h3_token_sparsity_policy = parsed.to_dict()
        transformer._harness4h3_token_sparsity_stats = {"selected_tokens": 0, "skipped_tokens": 0}
        for index, block in enumerate(blocks):
            block_id = "block.%d" % index
            if hasattr(block, "_harness4h3_token_base_forward"):
                continue
            base = block.forward

            def token_forward(block_self: Any, hidden_states: Any, encoder_hidden_states: Any, temb: Any, rotary_emb: Any, *args: Any, _block_id: str = block_id, _base: Any = base, **kwargs: Any) -> Any:
                active = getattr(transformer, "_harness4h3_token_sparsity_policy", parsed.to_dict())
                if hidden_states.ndim < 3 or hidden_states.shape[1] <= 1 or (active.get("block_ids") and _block_id not in active["block_ids"]):
                    return _base(hidden_states, encoder_hidden_states, temb, rotary_emb, *args, **kwargs)
                count = max(1, min(int(hidden_states.shape[1]), int(round(float(hidden_states.shape[1]) * float(active["keep_ratio"])))) )
                scores = hidden_states.detach().float().pow(2).mean(dim=(0, 2))
                indices = torch.topk(scores, count).indices.sort().values
                reduced = hidden_states.index_select(1, indices)
                reduced_rotary = _slice_rotary(rotary_emb, indices, int(hidden_states.shape[1]))
                output = _base(reduced, encoder_hidden_states, temb, reduced_rotary, *args, **kwargs)
                restored = hidden_states.clone()
                restored.index_copy_(1, indices, output)
                transformer._harness4h3_token_sparsity_stats["selected_tokens"] += int(count)
                transformer._harness4h3_token_sparsity_stats["skipped_tokens"] += int(hidden_states.shape[1] - count)
                return restored

            block._harness4h3_token_base_forward = base
            block.forward = MethodType(token_forward, block)
        _record_runtime_operation(transformer, "prune_tokens", parsed.to_dict())
        return StructuralChange("runtime_token_sparsity", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0})

    def apply_token_merge(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        torch, _ = _torch_modules()
        parsed = TokenMergePolicy.from_mapping(policy)
        transformer = self._transformer(model)
        _path, blocks = self._blocks(model)
        known = {"block.%d" % index for index in range(len(blocks))}
        unknown = sorted(set(parsed.block_ids) - known)
        if unknown:
            raise ValueError("invalid_token_selection: unknown block %s" % unknown[0])
        _register_config(transformer, "harness4h3_token_merge", parsed.to_dict())
        transformer._harness4h3_token_merge_policy = parsed.to_dict()
        transformer._harness4h3_token_merge_stats = {
            "input_tokens": 0,
            "merged_tokens": 0,
            "groups": 0,
            "calls": 0,
            "last_anchors": [],
            "grid_shape": [],
            "group_shape": None,
        }
        for index, block in enumerate(blocks):
            block_id = "block.%d" % index
            attention = getattr(block, "attn1", None)
            if attention is None or hasattr(attention, "_harness4h3_token_merge_base_forward"):
                continue
            base = attention.forward

            def merge_forward(attention_self: Any, hidden_states: Any, encoder_hidden_states: Any = None, attention_mask: Any = None, rotary_emb: Any = None, *args: Any, _block_id: str = block_id, _base: Any = base, **kwargs: Any) -> Any:
                active = getattr(transformer, "_harness4h3_token_merge_policy", parsed.to_dict())
                if (
                    encoder_hidden_states is not None
                    or attention_mask is not None
                    or hidden_states.ndim < 3
                    or hidden_states.shape[1] <= 1
                    or (active.get("block_ids") and _block_id not in active["block_ids"])
                ):
                    return _base(hidden_states, encoder_hidden_states, attention_mask, rotary_emb, *args, **kwargs)
                mode = str(active.get("mode", "adjacent"))
                grid_shape = tuple(active.get("grid_shape", ())) if mode == "spatiotemporal" else ()
                merged, source_groups, anchors, group_shape = _merge_token_sequence(hidden_states, float(active["keep_ratio"]), torch, grid_shape)
                if source_groups is None:
                    return _base(hidden_states, encoder_hidden_states, attention_mask, rotary_emb, *args, **kwargs)
                merged_rotary = _slice_rotary(rotary_emb, anchors, int(hidden_states.shape[1]))
                output = _base(merged, None, None, merged_rotary, *args, **kwargs)
                restored = output.index_select(1, source_groups)
                stats = transformer._harness4h3_token_merge_stats
                stats["input_tokens"] += int(hidden_states.shape[1])
                stats["merged_tokens"] += int(merged.shape[1])
                stats["groups"] = int(merged.shape[1])
                stats["calls"] += 1
                stats["last_anchors"] = anchors.detach().cpu().tolist()
                stats["grid_shape"] = list(grid_shape)
                stats["group_shape"] = list(group_shape) if group_shape is not None else None
                return restored

            attention._harness4h3_token_merge_base_forward = base
            attention.forward = MethodType(merge_forward, attention)
        _record_runtime_operation(transformer, "token_merge", parsed.to_dict())
        return StructuralChange("runtime_token_merge", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0, "unmerge": True})

    def apply_kv_token_selection(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        torch, _ = _torch_modules()
        parsed = KVTokenSelectionPolicy.from_mapping(policy)
        transformer = self._transformer(model)
        if getattr(transformer, "_harness4h3_token_merge_policy", None):
            raise ValueError("incompatible_runtime_attention_composition: kv_token_select and token_merge")
        if getattr(transformer, "_harness4h3_attention_window_policy", None):
            raise ValueError("incompatible_runtime_attention_composition: kv_token_select and attention_window")
        _register_config(transformer, "harness4h3_kv_token_selection", parsed.to_dict())
        transformer._harness4h3_kv_token_selection_policy = parsed.to_dict()
        transformer._harness4h3_kv_token_selection_stats = {
            "query_tokens": 0,
            "full_kv_tokens": 0,
            "selected_kv_tokens": 0,
            "calls": 0,
            "last_indices": [],
        }
        _path, blocks = self._blocks(model)
        # If sparse_runtime was installed first, its wrapper is the combined
        # execution point.  It reads this policy and applies one K/V gather;
        # adding another wrapper would silently replace the operation order.
        if getattr(transformer, "_harness4h3_sparse_runtime_policy", None):
            _record_runtime_operation(transformer, "kv_token_select", parsed.to_dict())
            return StructuralChange("runtime_kv_token_selection", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0, "full_query_length_preserved": True, "combined_with_sparse_runtime": True})
        for index, block in enumerate(blocks):
            block_id = "block.%d" % index
            attention = getattr(block, "attn1", None)
            if attention is None or hasattr(attention, "_harness4h3_kv_token_base_forward"):
                continue
            base = attention.forward

            def kv_forward(attention_self: Any, hidden_states: Any, encoder_hidden_states: Any = None, attention_mask: Any = None, rotary_emb: Any = None, *args: Any, _block_id: str = block_id, _base: Any = base, **kwargs: Any) -> Any:
                active = getattr(transformer, "_harness4h3_kv_token_selection_policy", parsed.to_dict())
                if (
                    encoder_hidden_states is not None
                    or attention_mask is not None
                    or hidden_states.ndim < 3
                    or hidden_states.shape[1] <= 1
                    or (active.get("block_ids") and _block_id not in active["block_ids"])
                ):
                    return _base(hidden_states, encoder_hidden_states, attention_mask, rotary_emb, *args, **kwargs)
                indices = _select_kv_indices(hidden_states, float(active["keep_ratio"]), str(active.get("score", "l2_norm")), torch)
                keys = hidden_states.index_select(1, indices)
                output = _wan_sparse_self_attention(attention_self, hidden_states, keys, indices, rotary_emb, torch)
                stats = transformer._harness4h3_kv_token_selection_stats
                stats["query_tokens"] += int(hidden_states.shape[1])
                stats["full_kv_tokens"] += int(hidden_states.shape[1])
                stats["selected_kv_tokens"] += int(indices.numel())
                stats["calls"] += 1
                stats["last_indices"] = indices.detach().cpu().tolist()
                return output

            attention._harness4h3_kv_token_base_forward = base
            attention.forward = MethodType(kv_forward, attention)
        _record_runtime_operation(transformer, "kv_token_select", parsed.to_dict())
        return StructuralChange("runtime_kv_token_selection", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0, "full_query_length_preserved": True})

    def apply_sparse_runtime(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        torch, _ = _torch_modules()
        parsed = SparseRuntimePolicy.from_mapping(policy)
        transformer = self._transformer(model)
        if getattr(transformer, "_harness4h3_attention_window_policy", None):
            raise ValueError("incompatible_runtime_attention_composition: attention_window and sparse_runtime")
        _register_config(transformer, "harness4h3_sparse_runtime", parsed.to_dict())
        transformer._harness4h3_sparse_runtime_policy = parsed.to_dict()
        transformer._harness4h3_sparse_runtime_stats = {"full_tokens": 0, "selected_tokens": 0}
        _path, blocks = self._blocks(model)
        for block in blocks:
            attention = getattr(block, "attn1", None)
            if attention is None or hasattr(attention, "_harness4h3_sparse_base_forward"):
                continue
            base = attention.forward

            def sparse_forward(attention_self: Any, hidden_states: Any, encoder_hidden_states: Any = None, attention_mask: Any = None, rotary_emb: Any = None, *args: Any, _base: Any = base, **kwargs: Any) -> Any:
                active = getattr(transformer, "_harness4h3_sparse_runtime_policy", parsed.to_dict())
                if encoder_hidden_states is not None or hidden_states.ndim < 3 or hidden_states.shape[1] <= 1:
                    return _base(hidden_states, encoder_hidden_states, attention_mask, rotary_emb, *args, **kwargs)
                kv_policy = getattr(transformer, "_harness4h3_kv_token_selection_policy", None)
                keep_ratio = min(float(active["keep_ratio"]), float(kv_policy["keep_ratio"])) if kv_policy else float(active["keep_ratio"])
                score = str(kv_policy.get("score", active.get("score", "l2_norm"))) if kv_policy else str(active.get("score", "l2_norm"))
                indices = _select_kv_indices(hidden_states, keep_ratio, score, torch)
                keys = hidden_states.index_select(1, indices)
                # Keep full Q, reduce K/V, and slice K's original RoPE
                # positions independently.  Calling the stock processor with
                # a shared tuple would either misalign positions or disable
                # RoPE for the sparse branch.
                result = _wan_sparse_self_attention(attention_self, hidden_states, keys, indices, rotary_emb, torch)
                transformer._harness4h3_sparse_runtime_stats["full_tokens"] += int(hidden_states.shape[1])
                transformer._harness4h3_sparse_runtime_stats["selected_tokens"] += int(indices.numel())
                return result

            attention._harness4h3_sparse_base_forward = base
            attention.forward = MethodType(sparse_forward, attention)
        _record_runtime_operation(transformer, "sparse_runtime", parsed.to_dict())
        return StructuralChange("runtime_sparse_attention", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0})

    def apply_attention_window(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        torch, _ = _torch_modules()
        parsed = AttentionWindowPolicy.from_mapping(policy)
        transformer = self._transformer(model)
        if getattr(transformer, "_harness4h3_sparse_runtime_policy", None):
            raise ValueError("incompatible_runtime_attention_composition: sparse_runtime and attention_window")
        _register_config(transformer, "harness4h3_attention_window", parsed.to_dict())
        transformer._harness4h3_attention_window_policy = parsed.to_dict()
        transformer._harness4h3_attention_window_stats = {
            "query_tokens": 0,
            "selected_kv_tokens": 0,
            "windows": 0,
            "grid_shape": [],
            "window_radius": [],
        }
        _path, blocks = self._blocks(model)
        for block in blocks:
            attention = getattr(block, "attn1", None)
            if attention is None or hasattr(attention, "_harness4h3_window_base_forward"):
                continue
            base = attention.forward

            def window_forward(attention_self: Any, hidden_states: Any, encoder_hidden_states: Any = None, attention_mask: Any = None, rotary_emb: Any = None, *args: Any, _base: Any = base, **kwargs: Any) -> Any:
                active = getattr(transformer, "_harness4h3_attention_window_policy", parsed.to_dict())
                if encoder_hidden_states is not None or hidden_states.ndim < 3 or hidden_states.shape[1] <= 1:
                    return _base(hidden_states, encoder_hidden_states, attention_mask, rotary_emb, *args, **kwargs)
                grid_shape = tuple(active.get("grid_shape", ()))
                if grid_shape:
                    if grid_shape[0] * grid_shape[1] * grid_shape[2] != int(hidden_states.shape[1]):
                        raise ValueError("attention_window.grid_shape product must equal sequence length")
                    radius = torch.tensor(tuple(active["window_radius"]), device=hidden_states.device)
                    positions = torch.arange(int(hidden_states.shape[1]), device=hidden_states.device)
                    coordinates = torch.stack(
                        (
                            positions // (grid_shape[1] * grid_shape[2]),
                            (positions // grid_shape[2]) % grid_shape[1],
                            positions % grid_shape[2],
                        ),
                        dim=1,
                    )
                    pieces = []
                    selected = 0
                    for query_index in range(int(hidden_states.shape[1])):
                        query_indices = positions[query_index:query_index + 1]
                        key_indices = torch.nonzero((coordinates - coordinates[query_index]).abs().le(radius).all(dim=1), as_tuple=False).flatten()
                        query = hidden_states.index_select(1, query_indices)
                        keys = hidden_states.index_select(1, key_indices)
                        pieces.append(
                            _wan_sparse_self_attention(
                                attention_self,
                                query,
                                keys,
                                key_indices,
                                rotary_emb,
                                torch,
                                query_indices=query_indices,
                                sequence_length=int(hidden_states.shape[1]),
                            )
                        )
                        selected += int(key_indices.numel())
                    transformer._harness4h3_attention_window_stats["query_tokens"] += int(hidden_states.shape[1])
                    transformer._harness4h3_attention_window_stats["selected_kv_tokens"] += int(selected)
                    transformer._harness4h3_attention_window_stats["windows"] += int(hidden_states.shape[1])
                    transformer._harness4h3_attention_window_stats["grid_shape"] = list(grid_shape)
                    transformer._harness4h3_attention_window_stats["window_radius"] = list(active["window_radius"])
                    return torch.cat(pieces, dim=1)
                width = int(active["window_size"])
                pieces = []
                selected = 0
                for start in range(0, int(hidden_states.shape[1]), width):
                    stop = min(int(hidden_states.shape[1]), start + width)
                    query = hidden_states[:, start:stop]
                    # Wan's processor applies one RoPE tuple to both Q and K.
                    # Use non-overlapping local blocks so the same slice is
                    # valid for both tensors and preserves the original
                    # flattened video-token positions.
                    keys = hidden_states[:, start:stop]
                    local_rotary = rotary_emb
                    if isinstance(rotary_emb, (tuple, list)) and len(rotary_emb) == 2:
                        sliced = []
                        for frequencies in rotary_emb:
                            if getattr(frequencies, "ndim", 0) >= 2 and int(frequencies.shape[1]) >= stop:
                                sliced.append(frequencies[:, start:stop, ...])
                            else:
                                sliced = []
                                break
                        local_rotary = tuple(sliced) if len(sliced) == 2 else None
                    pieces.append(_base(query, keys, None, local_rotary, *args, **kwargs))
                    selected += stop - start
                transformer._harness4h3_attention_window_stats["query_tokens"] += int(hidden_states.shape[1])
                transformer._harness4h3_attention_window_stats["selected_kv_tokens"] += int(selected)
                transformer._harness4h3_attention_window_stats["windows"] += len(pieces)
                return torch.cat(pieces, dim=1)

            attention._harness4h3_window_base_forward = base
            attention.forward = MethodType(window_forward, attention)
        _record_runtime_operation(transformer, "attention_window", parsed.to_dict())
        return StructuralChange("runtime_attention_window", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0})

    def apply_sampling_step_policy(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        parsed = SamplingStepPolicy.from_mapping(policy)
        transformer = self._transformer(model)
        _register_config(transformer, "harness4h3_sampling_step", parsed.to_dict())
        transformer._harness4h3_sampling_step_policy = parsed.to_dict()
        setattr(model, "_harness4h3_sampling_step_policy", parsed.to_dict())
        scheduler = getattr(model, "scheduler", None)
        if scheduler is not None and callable(getattr(scheduler, "set_timesteps", None)) and not hasattr(scheduler, "_harness4h3_sampling_base_set_timesteps"):
            base_set_timesteps = scheduler.set_timesteps

            def controlled_set_timesteps(scheduler_self: Any, num_inference_steps: Any = None, *args: Any, _base: Any = base_set_timesteps, **kwargs: Any) -> Any:
                active = getattr(transformer, "_harness4h3_sampling_step_policy", parsed.to_dict())
                requested = None if num_inference_steps is None else int(num_inference_steps)
                effective = requested
                if requested is not None:
                    effective = min(requested, int(active["target_steps"]))
                transformer._harness4h3_sampling_step_stats = {
                    "requested_steps": requested,
                    "effective_steps": effective,
                    "strategy": str(active.get("strategy", "uniform")),
                }
                return _base(effective, *args, **kwargs)

            scheduler._harness4h3_sampling_base_set_timesteps = base_set_timesteps
            scheduler.set_timesteps = MethodType(controlled_set_timesteps, scheduler)
        _record_runtime_operation(transformer, "sampling_step_skip", parsed.to_dict())
        return StructuralChange("runtime_sampling_step_skip", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0})

    def sampling_steps(self, model: Any, timesteps: Sequence[Any]) -> tuple[Any, ...]:
        transformer = self._transformer(model)
        policy = getattr(transformer, "_harness4h3_sampling_step_policy", {})
        selected = select_sampling_steps(timesteps, policy) if policy else tuple(timesteps)
        if policy:
            transformer._harness4h3_sampling_step_stats = {
                "requested_steps": len(tuple(timesteps)),
                "effective_steps": len(selected),
                "strategy": str(policy.get("strategy", "uniform")),
            }
        return selected

    def estimate_flops(self, model: Any) -> int:
        """Estimate Wan transformer work from the inspected live modules.

        This is a structural multiply-add estimate, not a hardware timing
        claim.  Runtime policies scale the attention work estimate so the
        algorithm evaluator can compare static and dynamic candidates before
        the independent GPU evaluator supplies wall-clock evidence.
        """

        transformer = self._transformer(model)
        _path, blocks = self._blocks(model)
        total = 0.0
        for block in blocks:
            pair = self._ffn_pair(block)
            if pair is not None:
                up, down = pair
                total += 2.0 * (int(up.in_features) * int(up.out_features) + int(down.in_features) * int(down.out_features))
            for name in ("attn1", "attn2"):
                attention = getattr(block, name, None)
                if attention is None:
                    continue
                projections = 0
                for projection_name in ("to_q", "to_k", "to_v", "to_out"):
                    projection = getattr(attention, projection_name, None)
                    if isinstance(projection, (tuple, list)):
                        projection = projection[0] if projection else None
                    if projection is not None and hasattr(projection, "in_features"):
                        projections += int(projection.in_features) * int(projection.out_features)
                total += 2.0 * projections
                heads = int(getattr(attention, "heads", 1))
                head_dim = int(getattr(attention, "inner_dim", heads) // max(heads, 1))
                total += 4.0 * heads * head_dim * 64.0 * 64.0
        token_policy = getattr(transformer, "_harness4h3_token_sparsity_policy", {})
        token_merge_policy = getattr(transformer, "_harness4h3_token_merge_policy", {})
        kv_policy = getattr(transformer, "_harness4h3_kv_token_selection_policy", {})
        sparse_policy = getattr(transformer, "_harness4h3_sparse_runtime_policy", {})
        window_policy = getattr(transformer, "_harness4h3_attention_window_policy", {})
        sampling_policy = getattr(transformer, "_harness4h3_sampling_step_policy", {})
        if token_policy:
            total *= float(token_policy.get("keep_ratio", 1.0))
        if token_merge_policy:
            total *= float(token_merge_policy.get("keep_ratio", 1.0)) ** 2
        if kv_policy:
            total *= float(kv_policy.get("keep_ratio", 1.0))
        if sparse_policy:
            total *= float(sparse_policy.get("keep_ratio", 1.0))
        if window_policy:
            total *= min(1.0, float(window_policy.get("window_size", 64)) / 64.0)
        if sampling_policy:
            total *= float(sampling_policy.get("target_steps", 16)) / 16.0
        return int(max(0.0, round(total)))

    def profile_sample(self, model: Any, seed: int = 7) -> Mapping[str, Any]:
        """Create a small valid Wan transformer probe without running T5/VAE."""

        torch, _ = _torch_modules()
        transformer = self._transformer(model)
        patch_embedding = getattr(transformer, "patch_embedding", None)
        parameter = getattr(patch_embedding, "weight", None)
        if parameter is None:
            parameter = next(transformer.parameters())
        config = transformer.config
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        def rand(shape: tuple[int, ...]) -> Any:
            return torch.randn(shape, generator=generator, dtype=parameter.dtype).to(parameter.device)
        return {
            "hidden_states": rand((1, int(config.in_channels), 1, 8, 8)),
            "timestep": torch.tensor([500], device=parameter.device, dtype=torch.long),
            "encoder_hidden_states": rand((1, 8, int(config.text_dim))),
        }

    @staticmethod
    def _probe_output(value: Any) -> Any:
        return getattr(value, "sample", value)

    def forward_profile(self, model: Any, sample: Mapping[str, Any]) -> Any:
        torch, _ = _torch_modules()
        with torch.no_grad():
            return self._probe_output(self._transformer(model)(**dict(sample)))

    def runtime_profile(self, model: Any, seed: int = 7) -> Any:
        """Run runtime policies on a small mixed-timestep Wan probe.

        The ordinary profile probe is single-sample and is sufficient for
        structural comparison.  Runtime policies need a sequence-shaped
        input and, for timestep-conditioned policies, two independent batch
        timesteps so the adapter cannot silently choose the first element.
        """

        torch, _ = _torch_modules()
        sample = self.profile_sample(model, seed)
        hidden_states = sample["hidden_states"]
        encoder_hidden_states = sample["encoder_hidden_states"]
        if int(hidden_states.shape[0]) == 1:
            hidden_states = hidden_states.repeat(2, *([1] * (hidden_states.ndim - 1)))
            encoder_hidden_states = encoder_hidden_states.repeat(2, *([1] * (encoder_hidden_states.ndim - 1)))
        timestep = sample["timestep"]
        timestep = timestep.new_tensor([100, 900]) if int(timestep.shape[0]) == 1 else timestep
        return self.forward_profile(
            model,
            {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
            },
        )

    def forward_timestep_profile(self, model: Any, bucket: str, sample: Mapping[str, Any]) -> Any:
        values = {"high_noise": 900, "middle": 500, "low_noise": 100}
        probe = dict(sample)
        probe["timestep"] = probe["timestep"].new_full(probe["timestep"].shape, values[bucket])
        return self.forward_profile(model, probe)

    def tica_calibration_inputs(self, model: Any, manifest: Any) -> tuple[Any, ...]:
        """Load only explicitly cached real Wan transformer inputs.

        TICA deliberately refuses to synthesize a random real-Wan probe. The
        full pipeline may be used once by an external preparation command to
        create the cached tensors referenced by the manifest.
        """

        from ...evolution.tica_calibration import TICACalibrationInput

        torch, _ = _torch_modules()
        transformer = self._transformer(model)
        parameter = next(transformer.parameters())

        def load_tensor(path: str | None, label: str) -> Any:
            if not path or not Path(path).is_file():
                raise ValueError("unsupported_tica_calibration_input: cached calibration tensor missing: %s" % label)
            try:
                value = torch.load(path, map_location=parameter.device, weights_only=True)
            except TypeError:
                value = torch.load(path, map_location=parameter.device)
            if not hasattr(value, "shape"):
                raise ValueError("unsupported_tica_calibration_input: cached %s is not a tensor" % label)
            return value.to(device=parameter.device, dtype=parameter.dtype)

        samples = []
        for spec in manifest.specs:
            hidden_states = load_tensor(spec.latent_path, "latent")
            encoder_hidden_states = load_tensor(spec.encoder_hidden_states_path, "encoder_hidden_states")
            if len(tuple(hidden_states.shape)) != 5 or len(tuple(encoder_hidden_states.shape)) != 3:
                raise ValueError("unsupported_tica_calibration_input: invalid Wan tensor rank")
            samples.append(
                TICACalibrationInput(
                    spec=spec,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    metadata={
                        "device": str(parameter.device),
                        "dtype": str(parameter.dtype),
                        "provider": "cached_real_wan",
                        "offline_simulation": False,
                    },
                )
            )
        return tuple(samples)

    def forward_denoising(self, model: Any, calibration_input: Any, timestep: float) -> Any:
        torch, _ = _torch_modules()
        hidden_states = calibration_input.hidden_states
        timestep_tensor = torch.full(
            (int(hidden_states.shape[0]),),
            int(round(float(timestep) * 1000.0)),
            device=hidden_states.device,
            dtype=torch.long,
        )
        return self.forward_profile(
            model,
            {
                "hidden_states": hidden_states,
                "timestep": timestep_tensor,
                "encoder_hidden_states": calibration_input.encoder_hidden_states,
            },
        )

    def apply_timestep_block_gating(self, model: Any, rules: Sequence[Mapping[str, Any]]) -> StructuralChange:
        torch, _ = _torch_modules()
        transformer = self._transformer(model)
        _path, blocks = self._blocks(model)
        block_ids = ["block.%d" % index for index in range(len(blocks))]
        validated = validate_timestep_gate_plan(
            {"rules": [dict(item) for item in rules], "policy": "skip_when_in_range"},
            block_ids,
        )
        by_block = {rule.block_id: rule for rule in validated}
        if callable(getattr(transformer, "register_to_config", None)):
            transformer.register_to_config(harness4h3_timestep_block_gating=rules_to_dict(validated))
        else:
            setattr(transformer.config, "harness4h3_timestep_block_gating", rules_to_dict(validated))
        transformer._harness4h3_gate_rules = by_block
        transformer._harness4h3_gate_stats = {
            block_id: {"executed": 0, "skipped": 0} for block_id in block_ids
        }
        _record_runtime_operation(transformer, "timestep_block_gate", {"rules": rules_to_dict(validated), "policy": "skip_when_in_range"})

        if not hasattr(transformer, "_harness4h3_gate_base_forward"):
            base_transformer_forward = transformer.forward

            def gated_transformer_forward(transformer_self: Any, hidden_states: Any, timestep: Any, *args: Any, **kwargs: Any) -> Any:
                transformer_self._harness4h3_current_timestep = timestep
                try:
                    return base_transformer_forward(hidden_states, timestep, *args, **kwargs)
                finally:
                    transformer_self._harness4h3_current_timestep = None

            transformer._harness4h3_gate_base_forward = base_transformer_forward
            transformer.forward = MethodType(gated_transformer_forward, transformer)

        for index, block in enumerate(blocks):
            block_id = "block.%d" % index
            if hasattr(block, "_harness4h3_gate_base_forward"):
                continue
            base_block_forward = block.forward

            def gated_block_forward(block_self: Any, hidden_states: Any, *args: Any, _block_id: str = block_id, _base: Any = base_block_forward, **kwargs: Any) -> Any:
                current_timestep = getattr(transformer, "_harness4h3_current_timestep", None)
                rule = getattr(transformer, "_harness4h3_gate_rules", {}).get(_block_id)
                batch_size = int(hidden_states.shape[0]) if getattr(hidden_states, "ndim", 0) >= 1 else 1
                timesteps = normalized_timesteps(current_timestep, batch_size) if current_timestep is not None else (0.5,) * batch_size
                skip = tuple(rule is not None and rule_matches(rule, value) for value in timesteps)
                if rule is not None and all(skip):
                    # Preserve the public block-call counters while the
                    # mixed-batch path below executes only active samples.
                    transformer._harness4h3_gate_stats[_block_id]["skipped"] += 1
                    return hidden_states
                active = tuple(index for index, should_skip in enumerate(skip) if not should_skip)
                if len(active) == batch_size:
                    transformer._harness4h3_gate_stats[_block_id]["executed"] += 1
                    return _base(hidden_states, *args, **kwargs)
                active_indices = torch.tensor(active, device=hidden_states.device, dtype=torch.long)

                def select_batch(value: Any) -> Any:
                    if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0 and int(value.shape[0]) == batch_size:
                        return value.index_select(0, active_indices.to(value.device))
                    if isinstance(value, tuple):
                        return tuple(select_batch(item) for item in value)
                    if isinstance(value, list):
                        return [select_batch(item) for item in value]
                    if isinstance(value, Mapping):
                        return {key: select_batch(item) for key, item in value.items()}
                    return value

                def select_timestep(value: Any) -> Any:
                    if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0 and int(value.shape[0]) == batch_size:
                        return value.index_select(0, active_indices.to(value.device))
                    if isinstance(value, (list, tuple)) and len(value) == batch_size:
                        selected = [value[int(index)] for index in active_indices.detach().cpu().tolist()]
                        return tuple(selected) if isinstance(value, tuple) else selected
                    return value

                active_args = tuple(select_batch(item) for item in args)
                active_kwargs = {key: select_batch(item) for key, item in kwargs.items()}
                previous_timestep = getattr(transformer, "_harness4h3_current_timestep", None)
                if previous_timestep is not None:
                    transformer._harness4h3_current_timestep = select_timestep(previous_timestep)
                try:
                    output = _base(hidden_states.index_select(0, active_indices), *active_args, **active_kwargs)
                finally:
                    transformer._harness4h3_current_timestep = previous_timestep
                restored = hidden_states.clone()
                restored.index_copy_(0, active_indices, output)
                transformer._harness4h3_gate_stats[_block_id]["executed"] += 1
                transformer._harness4h3_gate_stats[_block_id]["skipped"] += 1
                return restored

            block._harness4h3_gate_base_forward = base_block_forward
            block.forward = MethodType(gated_block_forward, block)
        return StructuralChange(
            "runtime_timestep_gating",
            {
                "runtime_operator": "timestep_block_gate",
                "rules": rules_to_dict(validated),
                "parameter_reduction_ratio": 0.0,
                "dynamic_compute": True,
            },
        )

    def apply_timestep_attention_policy(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        """Select a real self-attention K/V budget from the current timestep.

        The wrapper groups mixed-timestep batches before attention, so no
        sample silently inherits the first element's policy.  Matched groups
        call the same RoPE/QK-Norm/output path as the ordinary KV selector,
        but pass a smaller K/V tensor to SDPA; unmatched groups delegate to
        the original (or previously installed) attention implementation.
        """

        torch, _ = _torch_modules()
        transformer = self._transformer(model)
        _path, blocks = self._blocks(model)
        block_ids = ["block.%d" % index for index in range(len(blocks))]
        validated = validate_timestep_attention_plan(
            {"rules": [dict(item) for item in policy.get("rules", [])], "policy": policy.get("policy")},
            block_ids,
        )
        serialized_rules = [rule.to_dict() for rule in validated]
        _register_config(transformer, "harness4h3_timestep_attention_policy", {"rules": serialized_rules, "policy": "select_kv_tokens"})
        transformer._harness4h3_timestep_attention_policy = {"rules": serialized_rules, "policy": "select_kv_tokens"}
        transformer._harness4h3_timestep_attention_rules = validated
        transformer._harness4h3_timestep_attention_stats = {
            block_id: {"calls": 0, "full_kv_tokens": 0, "selected_kv_tokens": 0, "by_rule": {}}
            for block_id in block_ids
        }
        _record_runtime_operation(transformer, "timestep_attention_policy", transformer._harness4h3_timestep_attention_policy)

        if not hasattr(transformer, "_harness4h3_timestep_attention_base_forward"):
            base_transformer_forward = transformer.forward

            def timestep_transformer_forward(transformer_self: Any, hidden_states: Any, timestep: Any, *args: Any, _base: Any = base_transformer_forward, **kwargs: Any) -> Any:
                transformer_self._harness4h3_current_timestep = timestep
                try:
                    return _base(hidden_states, timestep, *args, **kwargs)
                finally:
                    transformer_self._harness4h3_current_timestep = None

            transformer._harness4h3_timestep_attention_base_forward = base_transformer_forward
            transformer.forward = MethodType(timestep_transformer_forward, transformer)

        for index, block in enumerate(blocks):
            block_id = "block.%d" % index
            attention = getattr(block, "attn1", None)
            if attention is None or hasattr(attention, "_harness4h3_timestep_attention_base_forward"):
                continue
            base = attention.forward

            def timestep_attention_forward(
                attention_self: Any,
                hidden_states: Any,
                encoder_hidden_states: Any = None,
                attention_mask: Any = None,
                rotary_emb: Any = None,
                *args: Any,
                _block_id: str = block_id,
                _base: Any = base,
                **kwargs: Any,
            ) -> Any:
                current_timestep = getattr(transformer, "_harness4h3_current_timestep", None)
                if current_timestep is None or encoder_hidden_states is not None or getattr(hidden_states, "ndim", 0) < 3 or int(hidden_states.shape[1]) <= 1:
                    return _base(hidden_states, encoder_hidden_states, attention_mask, rotary_emb, *args, **kwargs)
                batch_size = int(hidden_states.shape[0])
                timesteps = normalized_timesteps(current_timestep, batch_size)
                selected_rules = tuple(
                    next((rule for rule in getattr(transformer, "_harness4h3_timestep_attention_rules", ()) if rule.applies_to(_block_id) and rule_matches(rule, value)), None)
                    for value in timesteps
                )
                if not any(rule is not None for rule in selected_rules):
                    return _base(hidden_states, encoder_hidden_states, attention_mask, rotary_emb, *args, **kwargs)

                def select_batch(value: Any, indices: Any) -> Any:
                    if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0 and int(value.shape[0]) == batch_size:
                        return value.index_select(0, indices.to(value.device))
                    if isinstance(value, tuple):
                        return tuple(select_batch(item, indices) for item in value)
                    if isinstance(value, list):
                        return [select_batch(item, indices) for item in value]
                    if isinstance(value, Mapping):
                        return {key: select_batch(item, indices) for key, item in value.items()}
                    return value

                output = hidden_states.clone()
                groups: dict[str | None, list[int]] = {}
                rule_by_key: dict[str | None, TimestepAttentionRule | None] = {}
                for local_index, rule in enumerate(selected_rules):
                    key = None if rule is None else repr(rule.to_dict())
                    groups.setdefault(key, []).append(local_index)
                    rule_by_key[key] = rule
                for key, local_indices in groups.items():
                    indices = torch.tensor(local_indices, device=hidden_states.device, dtype=torch.long)
                    local_hidden = hidden_states.index_select(0, indices)
                    local_encoder = select_batch(encoder_hidden_states, indices)
                    local_mask = select_batch(attention_mask, indices)
                    local_rotary = select_batch(rotary_emb, indices)
                    local_args = tuple(select_batch(item, indices) for item in args)
                    local_kwargs = {name: select_batch(item, indices) for name, item in kwargs.items()}
                    rule = rule_by_key[key]
                    if rule is None:
                        local_output = _base(local_hidden, local_encoder, local_mask, local_rotary, *local_args, **local_kwargs)
                    else:
                        kv_indices = _select_kv_indices(local_hidden, rule.keep_ratio, rule.score, torch)
                        keys = local_hidden.index_select(1, kv_indices)
                        local_output = _wan_sparse_self_attention(
                            attention_self,
                            local_hidden,
                            keys,
                            kv_indices,
                            local_rotary,
                            torch,
                        )
                        stats = transformer._harness4h3_timestep_attention_stats[_block_id]
                        stats["calls"] += 1
                        stats["full_kv_tokens"] += int(local_hidden.shape[1]) * int(local_hidden.shape[0])
                        stats["selected_kv_tokens"] += int(kv_indices.numel()) * int(local_hidden.shape[0])
                        rule_key = "%s:%s" % (rule.min_timestep, rule.max_timestep)
                        stats["by_rule"][rule_key] = int(stats["by_rule"].get(rule_key, 0)) + int(local_hidden.shape[0])
                    output.index_copy_(0, indices, local_output)
                return output

            attention._harness4h3_timestep_attention_base_forward = base
            attention.forward = MethodType(timestep_attention_forward, attention)
        return StructuralChange(
            "runtime_timestep_attention_policy",
            {
                "runtime_operator": "timestep_attention_policy",
                "rules": serialized_rules,
                "policy": "select_kv_tokens",
                "parameter_reduction_ratio": 0.0,
                "dynamic_compute": True,
                "per_sample_timestep": True,
            },
        )

    def probe_block_removal(self, model: Any, block_id: str, sample: Any, forward_fn: Any) -> Any:
        """Probe one block in-place to avoid 30 full 1.3B deep copies."""

        path, blocks = self._blocks(model)
        _, nn = _torch_modules()
        index = int(str(block_id).split(".")[-1])
        original = blocks
        kept = [block for current, block in enumerate(blocks) if current != index]
        transformer = self._transformer(model)
        setattr(transformer, path, nn.ModuleList(kept))
        _update_config(transformer, num_layers=len(kept))
        try:
            return forward_fn(model, sample)
        finally:
            setattr(transformer, path, original)
            _update_config(transformer, num_layers=len(original))

    def save_model(self, model: Any, output_dir: Path) -> CheckpointArtifact:
        target = Path(output_dir)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shared_components = False
        saved = False
        transformer = self._transformer(model)
        source_raw = getattr(model, "_harness4h3_checkpoint_root", "")
        source = Path(str(source_raw)) if str(source_raw).strip() else None
        if source is not None and hasattr(transformer, "save_pretrained"):
            shared_components = _copy_shared_components(source, target)
            if shared_components:
                transformer.save_pretrained(str(target / "transformer"))
                saved = True
            else:
                # ``DiffusionPipeline.save_pretrained`` is not guaranteed to
                # materialize components that were loaded with the current
                # diffusers offload/device policy.  Copy the trusted source
                # pipeline explicitly, then replace only its Transformer.
                saved = _copy_full_checkpoint(source, target)
                if saved:
                    transformer.save_pretrained(str(target / "transformer"))
        if not saved and hasattr(model, "save_pretrained"):
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            model.save_pretrained(str(target))
            saved = True
        if not saved:
            raise ValueError("unsupported_model_layout: model cannot save_pretrained")
        value = _transformer_checkpoint_sha256(target)
        model._harness4h3_checkpoint_sha256 = value
        structure = self.inspect(model)
        return CheckpointArtifact(
            str(target),
            value,
            self.architecture_name,
            structure,
            {
                "offline_simulation": False,
                "checkpoint_hash_scope": "model_index+transformer",
                "shared_components": "hardlink-or-external-symlink" if shared_components else "copied",
            },
        )

    def reload_model(self, artifact: CheckpointArtifact) -> Any:
        actual = _transformer_checkpoint_sha256(artifact.path)
        if actual != artifact.sha256:
            raise ValueError("checkpoint_reload_failed: digest mismatch")
        model = self.load_model(artifact.path, device=self._device, dtype=self._dtype)
        if getattr(model, "_harness4h3_checkpoint_sha256", None) != artifact.sha256:
            raise ValueError("checkpoint_reload_failed: loaded digest mismatch")
        return model


def _update_config(module: Any, **values: Any) -> None:
    register = getattr(module, "register_to_config", None)
    if callable(register):
        register(**values)
        return
    config = getattr(module, "config", None)
    if config is not None:
        for key, value in values.items():
            try:
                setattr(config, key, value)
            except Exception:
                pass
