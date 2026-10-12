from __future__ import annotations

import hashlib
import io
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import (
    BlockRef,
    CheckpointArtifact,
    ChannelPruningConfig,
    ModelStructure,
    StructuralChange,
    stable_digest,
)


def _torch() -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("fake Wan adapter requires the optional torch dependency") from exc
    return torch, nn


class FakeWanBlock:
    """A tiny DiT-like residual block with explicit FFN projections."""

    def __init__(self, hidden_size: int, ffn_width: int) -> None:
        torch, nn = _torch()
        self._module = nn.Module()
        self._module.up_proj = nn.Linear(hidden_size, ffn_width)
        self._module.down_proj = nn.Linear(ffn_width, hidden_size)
        self._module.activation = nn.GELU()

    @property
    def module(self) -> Any:
        return self._module


def _copy_module(module: Any) -> Any:
    return deepcopy(module)


class FakeWanDiT:
    """Small Torch module kept separate so the adapter owns structure policy."""

    def __init__(self, block_count: int, hidden_size: int, ffn_width: int, *, block_ids: Sequence[str] | None = None, ffn_widths: Sequence[int] | None = None) -> None:
        _, nn = _torch()
        widths = list(ffn_widths or [ffn_width] * block_count)
        if len(widths) != block_count:
            raise ValueError("ffn_widths length must match block_count")
        self.hidden_size = int(hidden_size)
        self.block_ids = list(block_ids or ["block.%d" % index for index in range(block_count)])
        if len(self.block_ids) != block_count:
            raise ValueError("block_ids length must match block_count")
        self.ffn_widths = [int(width) for width in widths]
        self.blocks = nn.ModuleList()
        for width in self.ffn_widths:
            block = nn.Module()
            block.up_proj = nn.Linear(self.hidden_size, width)
            block.down_proj = nn.Linear(width, self.hidden_size)
            block.activation = nn.GELU()
            self.blocks.append(block)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        self._artifact_sha256 = "0" * 64

    def __call__(self, value: Any) -> Any:
        return self.forward(value)

    def forward(self, value: Any) -> Any:
        for block in self.blocks:
            value = value + block.down_proj(block.activation(block.up_proj(value)))
        return self.output(value)

    def parameters(self) -> Iterable[Any]:
        return self.blocks.parameters()

    def state_dict(self) -> Any:
        return self.blocks.state_dict() | {"output." + key: value for key, value in self.output.state_dict().items()}

    def load_state_dict(self, state: Any, strict: bool = True) -> Any:
        block_state = {key: value for key, value in state.items() if not key.startswith("output.")}
        output_state = {key[len("output."):]: value for key, value in state.items() if key.startswith("output.")}
        result = self.blocks.load_state_dict(block_state, strict=strict)
        self.output.load_state_dict(output_state, strict=strict)
        return result

    def to(self, *args: Any, **kwargs: Any) -> "FakeWanDiT":
        self.blocks.to(*args, **kwargs)
        self.output.to(*args, **kwargs)
        return self

    def eval(self) -> "FakeWanDiT":
        self.blocks.eval()
        self.output.eval()
        return self


def _state_bytes(model: FakeWanDiT) -> bytes:
    torch, _ = _torch()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _parameter_count(module: Any) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


class FakeWanDiTAdapter:
    architecture_name = "WanFakeDiT"

    def __init__(self, block_count: int = 4, hidden_size: int = 8, ffn_width: int = 16, seed: int = 7) -> None:
        if block_count < 2 or hidden_size <= 0 or ffn_width <= 0:
            raise ValueError("fake Wan dimensions are invalid")
        self.block_count = int(block_count)
        self.hidden_size = int(hidden_size)
        self.ffn_width = int(ffn_width)
        self.seed = int(seed)

    def load_model(self, checkpoint: str, *, device: str = "cpu", dtype: str = "float32") -> FakeWanDiT:
        torch, _ = _torch()
        if str(checkpoint).startswith("fake://"):
            torch.manual_seed(self.seed)
            model = FakeWanDiT(self.block_count, self.hidden_size, self.ffn_width)
        else:
            root = Path(checkpoint)
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            model = FakeWanDiT(
                int(config["block_count"]),
                int(config["hidden_size"]),
                int(config["ffn_width"]),
                block_ids=config["block_ids"],
                ffn_widths=config["ffn_widths"],
            )
            model.load_state_dict(torch.load(root / "model.pt", map_location="cpu", weights_only=True))
        model.to(device=device)
        if dtype != "float32":
            model.to(dtype=getattr(torch, str(dtype)))
        model.eval()
        return model

    def inspect(self, model: FakeWanDiT) -> ModelStructure:
        if not isinstance(model, FakeWanDiT):
            raise ValueError("unsupported_model_layout: expected FakeWanDiT")
        digest = str(getattr(model, "_artifact_sha256", "0" * 64))
        if digest == "0" * 64:
            digest = hashlib.sha256(_state_bytes(model)).hexdigest()
        return ModelStructure(
            architecture=self.architecture_name,
            block_count=len(model.blocks),
            hidden_size=model.hidden_size,
            ffn_width=max(model.ffn_widths),
            parameter_count=_parameter_count(model),
            checkpoint_sha256=digest,
            metadata={"block_ids": list(model.block_ids), "ffn_widths": list(model.ffn_widths), "offline_simulation": True},
        )

    def get_blocks(self, model: FakeWanDiT) -> tuple[BlockRef, ...]:
        if not isinstance(model, FakeWanDiT):
            raise ValueError("unsupported_model_layout: expected FakeWanDiT")
        return tuple(
            BlockRef(
                block_id=block_id,
                index=index,
                parameter_count=_parameter_count(model.blocks[index]),
                module_path="blocks.%d" % index,
            )
            for index, block_id in enumerate(model.block_ids)
        )

    def remove_blocks(self, model: FakeWanDiT, block_ids: Sequence[str]) -> StructuralChange:
        selected = tuple(str(item) for item in block_ids)
        known = set(model.block_ids)
        if not selected:
            raise ValueError("invalid_block_selection: selection is empty")
        if len(set(selected)) != len(selected):
            raise ValueError("invalid_block_selection: duplicate block id")
        unknown = sorted(set(selected) - known)
        if unknown:
            raise ValueError("invalid_block_selection: unknown block %s" % unknown[0])
        if len(selected) >= len(model.blocks):
            raise ValueError("invalid_block_selection: cannot remove every block")
        keep = [(block_id, block) for block_id, block in zip(model.block_ids, model.blocks) if block_id not in selected]
        _, nn = _torch()
        model.blocks = nn.ModuleList([_copy_module(block) for _, block in keep])
        model.block_ids = [block_id for block_id, _ in keep]
        model.ffn_widths = [int(block.up_proj.out_features) for block in model.blocks]
        return StructuralChange("block_pruning", {"removed_block_ids": list(selected), "original_blocks": len(selected) + len(keep), "retained_blocks": len(keep)})

    def prune_channels(self, model: FakeWanDiT, config: ChannelPruningConfig) -> StructuralChange:
        known = set(model.block_ids)
        unknown = sorted(set(config.block_ids) - known)
        if unknown:
            raise ValueError("invalid_channel_selection: unknown block %s" % unknown[0])
        max_width = max(model.ffn_widths)
        if any(index >= max_width for index in config.retained_indices):
            raise ValueError("invalid_channel_selection: channel index out of range")
        torch, nn = _torch()
        selected = set(config.block_ids)
        original_widths: dict[str, int] = {}
        for index, block_id in enumerate(model.block_ids):
            if block_id not in selected:
                continue
            block = model.blocks[index]
            width = int(block.up_proj.out_features)
            if any(channel >= width for channel in config.retained_indices):
                raise ValueError("invalid_channel_selection: channel index out of range for %s" % block_id)
            retained = tuple(config.retained_indices)
            if len(retained) >= width:
                raise ValueError("invalid_channel_selection: no channels removed")
            original_widths[block_id] = width
            up = nn.Linear(model.hidden_size, len(retained), bias=block.up_proj.bias is not None)
            down = nn.Linear(len(retained), model.hidden_size, bias=block.down_proj.bias is not None)
            with torch.no_grad():
                up.weight.copy_(block.up_proj.weight[torch.tensor(retained)])
                if up.bias is not None:
                    up.bias.copy_(block.up_proj.bias[torch.tensor(retained)])
                down.weight.copy_(block.down_proj.weight[:, torch.tensor(retained)])
                if down.bias is not None:
                    down.bias.copy_(block.down_proj.bias)
            block.up_proj = up
            block.down_proj = down
            model.ffn_widths[index] = len(retained)
        return StructuralChange("ffn_channel_pruning", {"block_ids": list(config.block_ids), "retained_indices": list(config.retained_indices), "original_widths": original_widths, "retained_width": len(config.retained_indices)})

    def save_model(self, model: FakeWanDiT, output_dir: Path) -> CheckpointArtifact:
        torch, _ = _torch()
        target = Path(output_dir)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=False)
        state_path = target / "model.pt"
        config_path = target / "config.json"
        torch.save(model.state_dict(), state_path)
        config = {
            "architecture": self.architecture_name,
            "block_count": len(model.blocks),
            "hidden_size": model.hidden_size,
            "ffn_width": max(model.ffn_widths),
            "ffn_widths": list(model.ffn_widths),
            "block_ids": list(model.block_ids),
        }
        config_path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        digest = hashlib.sha256(state_path.read_bytes() + config_path.read_bytes()).hexdigest()
        model._artifact_sha256 = digest
        structure = self.inspect(model)
        if structure.checkpoint_sha256 != digest:
            raise RuntimeError("checkpoint_digest_mismatch")
        return CheckpointArtifact(str(target), digest, self.architecture_name, structure, {"offline_simulation": True})

    def reload_model(self, artifact: CheckpointArtifact) -> FakeWanDiT:
        model = self.load_model(artifact.path)
        structure = self.inspect(model)
        if structure.checkpoint_sha256 != artifact.sha256:
            # The loader has no need to know the original digest before it has
            # read both files, so compare the canonical artifact files here.
            root = Path(artifact.path)
            actual = hashlib.sha256((root / "model.pt").read_bytes() + (root / "config.json").read_bytes()).hexdigest()
            if actual != artifact.sha256:
                raise ValueError("checkpoint_reload_failed: digest mismatch")
            model._artifact_sha256 = actual
        return model

