from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .contracts import BlockRef, CheckpointArtifact, ChannelPruningConfig, ModelStructure, StructuralChange


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


def _module_at(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part) if not part.isdigit() else value[int(part)]
    return value


class DiffusersWanAdapter:
    """Best-effort structural boundary for real Wan Diffusers checkpoints.

    Diffusers has changed Wan internals across releases.  This adapter probes
    known module-list names and refuses unknown layouts instead of silently
    applying a partial mutation.
    """

    architecture_name = "WanDiffusersDiT"
    _block_paths = ("transformer_blocks", "blocks", "single_blocks")

    def __init__(self, local_files_only: bool = True) -> None:
        self.local_files_only = bool(local_files_only)
        self._device = "cpu"
        self._dtype = "float32"

    def load_model(self, checkpoint: str, *, device: Optional[str] = None, dtype: Optional[str] = None) -> Any:
        torch, _ = _torch_modules()
        pipeline_cls = _import_pipeline()
        device = str(self._device if device is None else device)
        dtype = str(self._dtype if dtype is None else dtype)
        pipe = pipeline_cls.from_pretrained(checkpoint, local_files_only=self.local_files_only, torch_dtype=getattr(torch, dtype))
        if not hasattr(pipe, "transformer"):
            raise ValueError("unsupported_model_layout: Wan pipeline has no transformer")
        # Structural exploration only needs the DiT transformer.  Moving the
        # T5 encoder and VAE as well can exceed one GPU and also triggers
        # allocator/NVML issues on otherwise usable multi-GPU hosts.
        pipe.transformer.to(device)
        pipe.transformer.eval()
        self._device = str(device)
        self._dtype = str(dtype)
        pipe._harness4h3_checkpoint_root = str(checkpoint)
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
        return ModelStructure(self.architecture_name, len(blocks), hidden, max(widths), _param_count(transformer), digest, {"block_path": path, "ffn_widths": widths, "offline_simulation": False, "profile_device": self._device, "profile_dtype": self._dtype})

    def get_blocks(self, model: Any) -> tuple[BlockRef, ...]:
        path, blocks = self._blocks(model)
        prefix = "transformer." if hasattr(model, "transformer") else ""
        return tuple(BlockRef("block.%d" % index, index, _param_count(block), "%s%s.%d" % (prefix, path, index)) for index, block in enumerate(blocks))

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

    def forward_timestep_profile(self, model: Any, bucket: str, sample: Mapping[str, Any]) -> Any:
        values = {"high_noise": 900, "middle": 500, "low_noise": 100}
        probe = dict(sample)
        probe["timestep"] = probe["timestep"].new_full(probe["timestep"].shape, values[bucket])
        return self.forward_profile(model, probe)

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
        target.mkdir(parents=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(target))
        else:
            raise ValueError("unsupported_model_layout: model cannot save_pretrained")
        digest = hashlib.sha256()
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(target)).encode("utf-8"))
            digest.update(path.read_bytes())
        value = digest.hexdigest()
        model._harness4h3_checkpoint_sha256 = value
        structure = self.inspect(model)
        return CheckpointArtifact(str(target), value, self.architecture_name, structure, {"offline_simulation": False})

    def reload_model(self, artifact: CheckpointArtifact) -> Any:
        model = self.load_model(artifact.path, device=self._device, dtype=self._dtype)
        model._harness4h3_checkpoint_sha256 = artifact.sha256
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
