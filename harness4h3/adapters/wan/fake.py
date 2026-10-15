from __future__ import annotations

import hashlib
import io
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    BlockRef,
    CheckpointArtifact,
    ChannelPruningConfig,
    ModelStructure,
    StructuralChange,
    checkpoint_content_sha256,
)
from ...operators.wan.runtime_gating import TimestepGateRule, normalized_timesteps, rule_matches, validate_timestep_gate_plan
from ...operators.wan.runtime_policy import AttentionWindowPolicy, KVTokenSelectionPolicy, SamplingStepPolicy, SparseRuntimePolicy, TokenMergePolicy, TokenSparsityPolicy, select_sampling_steps
from .attention import prune_attention_module_heads


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


def _fake_attention_module(hidden_size: int, heads: int, head_dim: int | None = None) -> Any:
    """Return a deepcopy-safe Wan-shaped attention module."""

    torch, nn = _torch()
    effective_head_dim = int(head_dim or (hidden_size // heads))
    if effective_head_dim <= 0:
        raise ValueError("fake attention head dimension is invalid")

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = int(heads)
            self.inner_dim = int(effective_head_dim * heads)
            self.kv_inner_dim = int(effective_head_dim * heads)
            self.to_q = nn.Linear(hidden_size, self.inner_dim)
            self.to_k = nn.Linear(hidden_size, self.inner_dim)
            self.to_v = nn.Linear(hidden_size, self.inner_dim)
            self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, hidden_size), nn.Dropout(0.0)])

        def _attend(self, query: Any, context: Any) -> Any:
            heads_now = int(self.heads)
            dim = int(self.inner_dim) // heads_now
            query_tokens = int(query.shape[1])
            query = self.to_q(query).reshape(query.shape[0], query.shape[1], heads_now, dim).transpose(1, 2)
            key = self.to_k(context).reshape(context.shape[0], context.shape[1], heads_now, dim).transpose(1, 2)
            val = self.to_v(context).reshape(context.shape[0], context.shape[1], heads_now, dim).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) / max(dim, 1) ** 0.5
            weights = torch.softmax(scores, dim=-1)
            result = torch.matmul(weights, val).transpose(1, 2).reshape(query.shape[0], query_tokens, heads_now * dim)
            return self.to_out[0](result)

        def forward(self, value: Any, encoder_hidden_states: Any = None, **kwargs: Any) -> Any:
            squeeze = value.ndim == 2
            hidden = value.unsqueeze(1) if squeeze else value
            context = hidden if encoder_hidden_states is None else encoder_hidden_states
            merge_groups = None
            merge = getattr(self, "_harness4h3_token_merge_policy", None)
            block_id = str(getattr(self, "_harness4h3_block_id", ""))
            if merge and encoder_hidden_states is None and hidden.ndim >= 3 and hidden.shape[1] > 1 and (not merge.get("block_ids") or block_id in merge["block_ids"]):
                keep = max(1, min(int(hidden.shape[1]), int(round(float(hidden.shape[1]) * float(merge["keep_ratio"])))) )
                positions = torch.arange(int(hidden.shape[1]), device=hidden.device)
                merge_groups = torch.div(positions * keep, int(hidden.shape[1]), rounding_mode="floor")
                merged = hidden.new_zeros((hidden.shape[0], keep, hidden.shape[2]))
                merged.scatter_add_(1, merge_groups.view(1, -1, 1).expand(hidden.shape[0], -1, hidden.shape[2]), hidden)
                counts = torch.bincount(merge_groups, minlength=keep).clamp_min(1).to(hidden.dtype).view(1, keep, 1)
                hidden = merged / counts
                context = hidden
                anchors = torch.empty(keep, device=hidden.device, dtype=torch.long)
                for group in range(keep):
                    anchors[group] = positions[merge_groups == group][0]
                self._harness4h3_token_merge_stats = {
                    "input_tokens": int(positions.numel()),
                    "merged_tokens": int(keep),
                    "groups": int(keep),
                    "anchors": anchors.detach().cpu().tolist(),
                }
            policy = getattr(self, "_harness4h3_sparse_policy", None)
            kv_policy = getattr(self, "_harness4h3_kv_token_selection_policy", None)
            if kv_policy and encoder_hidden_states is None and hidden.ndim >= 3 and hidden.shape[1] > 1 and (not kv_policy.get("block_ids") or block_id in kv_policy["block_ids"]):
                keep = max(1, min(int(hidden.shape[1]), int(round(float(hidden.shape[1]) * float(kv_policy["keep_ratio"])))) )
                scores = context.detach().float().pow(2).mean(dim=-1) if kv_policy.get("score", "l2_norm") == "l2_norm" else context.detach().float().abs().mean(dim=-1)
                indices = scores.topk(keep, dim=1).indices.sort(dim=1).values
                context = context.gather(1, indices.unsqueeze(-1).expand(-1, -1, context.shape[-1]))
                self._harness4h3_kv_token_selection_stats = {
                    "query_tokens": int(hidden.shape[1]),
                    "full_kv_tokens": int(hidden.shape[1]),
                    "selected_kv_tokens": int(keep),
                    "indices": indices.detach().cpu().tolist(),
                }
            if policy and context.ndim >= 3 and context.shape[1] > 1 and encoder_hidden_states is None:
                keep = max(1, min(int(context.shape[1]), int(round(float(context.shape[1]) * float(policy["keep_ratio"])))) )
                scores = context.detach().float().pow(2).mean(dim=-1)
                indices = scores.topk(keep, dim=1).indices
                context = context.gather(1, indices.unsqueeze(-1).expand(-1, -1, context.shape[-1]))
                self._harness4h3_sparse_stats = {"full_tokens": int(hidden.shape[1]), "selected_tokens": int(keep)}
            window = getattr(self, "_harness4h3_attention_window_policy", None)
            if window and encoder_hidden_states is None and hidden.ndim >= 3 and hidden.shape[1] > 1:
                width = int(window["window_size"])
                pieces = []
                selected = 0
                for start in range(0, int(hidden.shape[1]), width):
                    stop = min(int(hidden.shape[1]), start + width)
                    pieces.append(self._attend(hidden[:, start:stop], hidden[:, start:stop]))
                    selected += stop - start
                result = torch.cat(pieces, dim=1)
                self._harness4h3_window_stats = {"query_tokens": int(hidden.shape[1]), "selected_kv_tokens": int(selected), "windows": len(pieces)}
            else:
                result = self._attend(hidden, context)
            if merge_groups is not None:
                result = result.index_select(1, merge_groups)
            return result.squeeze(1) if squeeze else result

    return Attention()


def _fake_block_module(hidden_size: int, width: int, heads: int, head_dim: int | None = None) -> Any:
    torch, nn = _torch()

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = nn.Linear(hidden_size, width)
            self.down_proj = nn.Linear(width, hidden_size)
            self.activation = nn.GELU()
            # Preserve the historical fake's FFN initialization stream. The
            # new attention modules are contract infrastructure and must not
            # perturb old TICA CPU evidence merely by consuming RNG draws.
            rng_state = torch.get_rng_state()
            self.attn1 = _fake_attention_module(hidden_size, heads, head_dim)
            self.attn2 = _fake_attention_module(hidden_size, heads, head_dim)
            torch.set_rng_state(rng_state)

        def forward(self, value: Any) -> Any:
            squeeze = value.ndim == 2
            current = value.unsqueeze(1) if squeeze else value
            # The CPU fake keeps the attention branch observationally
            # neutral so historical TICA thresholds remain comparable to the
            # pre-attention fake.  The Q/K/V modules still execute, expose
            # real head geometry, and are mutated by the head/sparse tests;
            # real Wan uses the unmodified attention residual path.
            attention_probe = self.attn1(current)
            attention_probe = self.attn2(current)
            current = self.down_proj(self.activation(self.up_proj(current)))
            current = current + 0.0 * attention_probe
            return current.squeeze(1) if squeeze else current

    return Block()


def _copy_module(module: Any) -> Any:
    return deepcopy(module)


class FakeWanDiT:
    """Small Torch module kept separate so the adapter owns structure policy."""

    def __init__(self, block_count: int, hidden_size: int, ffn_width: int, *, block_ids: Sequence[str] | None = None, ffn_widths: Sequence[int] | None = None, num_heads: int = 4, attention_head_dim: int | None = None, runtime_gate_rules: Sequence[Mapping[str, Any]] | None = None, token_sparsity_policy: Mapping[str, Any] | None = None, token_merge_policy: Mapping[str, Any] | None = None, kv_token_selection_policy: Mapping[str, Any] | None = None, sparse_runtime_policy: Mapping[str, Any] | None = None, attention_window_policy: Mapping[str, Any] | None = None, sampling_step_policy: Mapping[str, Any] | None = None, runtime_operations: Sequence[Mapping[str, Any]] | None = None) -> None:
        _, nn = _torch()
        widths = list(ffn_widths or [ffn_width] * block_count)
        if len(widths) != block_count:
            raise ValueError("ffn_widths length must match block_count")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.attention_head_dim = int(attention_head_dim or (self.hidden_size // self.num_heads))
        if self.num_heads < 1 or self.attention_head_dim <= 0:
            raise ValueError("fake attention head geometry is invalid")
        self.block_ids = list(block_ids or ["block.%d" % index for index in range(block_count)])
        if len(self.block_ids) != block_count:
            raise ValueError("block_ids length must match block_count")
        self.ffn_widths = [int(width) for width in widths]
        self.blocks = nn.ModuleList()
        for width in self.ffn_widths:
            self.blocks.append(_fake_block_module(self.hidden_size, width, self.num_heads, self.attention_head_dim))
        for block_id, block in zip(self.block_ids, self.blocks):
            block.attn1._harness4h3_block_id = str(block_id)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        self._artifact_sha256 = "0" * 64
        self.runtime_gate_rules = [dict(item) for item in (runtime_gate_rules or [])]
        self._harness4h3_token_sparsity_policy = dict(token_sparsity_policy or {})
        self._harness4h3_token_merge_policy = dict(token_merge_policy or {})
        self._harness4h3_kv_token_selection_policy = dict(kv_token_selection_policy or {})
        self._harness4h3_sparse_runtime_policy = dict(sparse_runtime_policy or {})
        self._harness4h3_attention_window_policy = dict(attention_window_policy or {})
        self._harness4h3_sampling_step_policy = dict(sampling_step_policy or {})
        self._harness4h3_runtime_operations = [dict(item) for item in (runtime_operations or [])]
        if self._harness4h3_sparse_runtime_policy:
            for block in self.blocks:
                block.attn1._harness4h3_sparse_policy = dict(self._harness4h3_sparse_runtime_policy)
        if self._harness4h3_token_merge_policy:
            for block in self.blocks:
                block.attn1._harness4h3_token_merge_policy = dict(self._harness4h3_token_merge_policy)
        if self._harness4h3_kv_token_selection_policy:
            for block in self.blocks:
                block.attn1._harness4h3_kv_token_selection_policy = dict(self._harness4h3_kv_token_selection_policy)
        if self._harness4h3_attention_window_policy:
            for block in self.blocks:
                block.attn1._harness4h3_attention_window_policy = dict(self._harness4h3_attention_window_policy)
        self.runtime_gate_stats = {
            str(block_id): {"executed": 0, "skipped": 0}
            for block_id in self.block_ids
        }
        self.runtime_token_stats = {str(block_id): {"selected": 0, "skipped": 0} for block_id in self.block_ids}

    def __call__(self, value: Any, timestep: Any = None) -> Any:
        return self.forward(value, timestep=timestep)

    def forward(self, value: Any, timestep: Any = None) -> Any:
        batch_size = int(value.shape[0]) if getattr(value, "ndim", 0) >= 1 else 1
        current_timesteps = (0.5,) * batch_size if timestep is None else normalized_timesteps(timestep, batch_size)
        rules = {str(item["block_id"]): TimestepGateRule.from_mapping(item) for item in self.runtime_gate_rules}
        token_policy = TokenSparsityPolicy.from_mapping(self._harness4h3_token_sparsity_policy) if self._harness4h3_token_sparsity_policy else None

        def run_block(inputs: Any, block: Any) -> Any:
            if token_policy is not None and (not token_policy.block_ids or block_id in token_policy.block_ids) and inputs.ndim >= 3 and inputs.shape[1] > 1:
                count = max(1, min(int(inputs.shape[1]), int(round(float(inputs.shape[1]) * token_policy.keep_ratio))))
                scores = inputs.detach().float().pow(2).mean(dim=-1)
                indices = scores.topk(count, dim=1).indices
                selected = inputs.gather(1, indices.unsqueeze(-1).expand(-1, -1, inputs.shape[-1]))
                updated = selected + block(selected)
                result = inputs.clone()
                result.scatter_(1, indices.unsqueeze(-1).expand_as(updated), updated)
                self.runtime_token_stats[str(block_id)]["selected"] += int(count * inputs.shape[0])
                self.runtime_token_stats[str(block_id)]["skipped"] += int((inputs.shape[1] - count) * inputs.shape[0])
                return result
            return inputs + block(inputs)

        for block_id, block in zip(self.block_ids, self.blocks):
            rule = rules.get(str(block_id))
            skip = tuple(rule is not None and rule_matches(rule, current) for current in current_timesteps)
            if rule is not None and all(skip):
                # Keep the historical counters as block-call counts; a mixed
                # batch still exposes the active subset execution below.
                self.runtime_gate_stats[str(block_id)]["skipped"] += 1
                continue
            active = tuple(index for index, should_skip in enumerate(skip) if not should_skip)
            if len(active) == batch_size:
                self.runtime_gate_stats[str(block_id)]["executed"] += 1
                value = run_block(value, block)
            else:
                torch, _ = _torch()
                active_indices = torch.tensor(active, device=value.device, dtype=torch.long)
                active_value = value.index_select(0, active_indices)
                active_value = run_block(active_value, block)
                value = value.clone()
                value.index_copy_(0, active_indices, active_value)
                self.runtime_gate_stats[str(block_id)]["executed"] += 1
                self.runtime_gate_stats[str(block_id)]["skipped"] += 1
        return self.output(value)

    def parameters(self) -> Iterable[Any]:
        for parameter in self.blocks.parameters():
            yield parameter
        for parameter in self.output.parameters():
            yield parameter

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
            model = FakeWanDiT(self.block_count, self.hidden_size, self.ffn_width, num_heads=4)
        else:
            root = Path(checkpoint)
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            model = FakeWanDiT(
                int(config["block_count"]),
                int(config["hidden_size"]),
                int(config["ffn_width"]),
                block_ids=config["block_ids"],
                ffn_widths=config["ffn_widths"],
                num_heads=int(config.get("num_heads", 4)),
                attention_head_dim=int(config.get("attention_head_dim", int(config["hidden_size"]) // int(config.get("num_heads", 4)))),
                runtime_gate_rules=config.get("runtime_gate_rules", []),
                token_sparsity_policy=config.get("token_sparsity_policy", {}),
                token_merge_policy=config.get("token_merge_policy", {}),
                kv_token_selection_policy=config.get("kv_token_selection_policy", {}),
                sparse_runtime_policy=config.get("sparse_runtime_policy", {}),
                attention_window_policy=config.get("attention_window_policy", {}),
                sampling_step_policy=config.get("sampling_step_policy", {}),
                runtime_operations=config.get("runtime_operations", []),
            )
            model.load_state_dict(torch.load(root / "model.pt", map_location="cpu", weights_only=True))
            model._artifact_sha256 = checkpoint_content_sha256(root)
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
            metadata={
                "block_ids": list(model.block_ids),
                "ffn_widths": list(model.ffn_widths),
                "offline_simulation": True,
                "runtime_operator": "timestep_block_gate" if model.runtime_gate_rules else None,
                "num_attention_heads": int(model.blocks[0].attn1.heads),
                "attention_head_dim": int(model.blocks[0].attn1.inner_dim // model.blocks[0].attn1.heads),
                "token_sparsity_policy": dict(model._harness4h3_token_sparsity_policy),
                "token_merge_policy": dict(model._harness4h3_token_merge_policy),
                "kv_token_selection_policy": dict(model._harness4h3_kv_token_selection_policy),
                "sparse_runtime_policy": dict(model._harness4h3_sparse_runtime_policy),
                "attention_window_policy": dict(model._harness4h3_attention_window_policy),
                "sampling_step_policy": dict(model._harness4h3_sampling_step_policy),
                "runtime_operations": [dict(item) for item in model._harness4h3_runtime_operations],
            },
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

    @staticmethod
    def ffn_components(block: Any) -> tuple[Any, Any, Any] | None:
        """Return the paired up/activation/down modules for profiling."""

        if all(hasattr(block, name) for name in ("up_proj", "activation", "down_proj")):
            return block.up_proj, block.activation, block.down_proj
        return None

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

    def prune_attention_heads(self, model: FakeWanDiT, removed_heads: Sequence[int], attention_names: Sequence[str] = ("attn1", "attn2")) -> StructuralChange:
        names = tuple(str(item) for item in attention_names)
        if not names:
            raise ValueError("invalid_attention_selection")
        available_names = tuple(name for name in ("attn1", "attn2") if hasattr(model.blocks[0], name))
        if set(names) != set(available_names):
            raise ValueError("global_attention_alignment_required")
        before = int(self.inspect(model).parameter_count)
        changes = []
        for block_id, block in zip(model.block_ids, model.blocks):
            for name in names:
                attention = getattr(block, name, None)
                if attention is None:
                    raise ValueError("unsupported_attention_layout: %s" % name)
                changes.append({"block_id": block_id, "attention": name, **prune_attention_module_heads(attention, removed_heads)})
        after = int(self.inspect(model).parameter_count)
        return StructuralChange("attention_head_pruning", {"attention_names": list(names), "head_indices": list(removed_heads), "changes": changes, "parameter_reduction_ratio": 1.0 - float(after) / max(float(before), 1.0), "remaining_heads": int(model.blocks[0].attn1.heads)})

    def apply_token_sparsity(self, model: FakeWanDiT, policy: Mapping[str, Any]) -> StructuralChange:
        parsed = TokenSparsityPolicy.from_mapping(policy)
        unknown = sorted(set(parsed.block_ids) - set(model.block_ids))
        if unknown:
            raise ValueError("invalid_token_selection: unknown block %s" % unknown[0])
        model._harness4h3_token_sparsity_policy = parsed.to_dict()
        model.runtime_token_stats = {str(block_id): {"selected": 0, "skipped": 0} for block_id in model.block_ids}
        model._harness4h3_runtime_operations.append({"operator": "prune_tokens", "args": parsed.to_dict()})
        return StructuralChange("runtime_token_sparsity", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0})

    def apply_token_merge(self, model: FakeWanDiT, policy: Mapping[str, Any]) -> StructuralChange:
        parsed = TokenMergePolicy.from_mapping(policy)
        unknown = sorted(set(parsed.block_ids) - set(model.block_ids))
        if unknown:
            raise ValueError("invalid_token_selection: unknown block %s" % unknown[0])
        model._harness4h3_token_merge_policy = parsed.to_dict()
        for block in model.blocks:
            block.attn1._harness4h3_token_merge_policy = parsed.to_dict()
        model._harness4h3_runtime_operations.append({"operator": "token_merge", "args": parsed.to_dict()})
        return StructuralChange("runtime_token_merge", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0, "unmerge": True})

    def apply_kv_token_selection(self, model: FakeWanDiT, policy: Mapping[str, Any]) -> StructuralChange:
        parsed = KVTokenSelectionPolicy.from_mapping(policy)
        unknown = sorted(set(parsed.block_ids) - set(model.block_ids))
        if unknown:
            raise ValueError("invalid_token_selection: unknown block %s" % unknown[0])
        if model._harness4h3_attention_window_policy:
            raise ValueError("incompatible_runtime_attention_composition: kv_token_select and attention_window")
        model._harness4h3_kv_token_selection_policy = parsed.to_dict()
        for block in model.blocks:
            block.attn1._harness4h3_kv_token_selection_policy = parsed.to_dict()
        model._harness4h3_runtime_operations.append({"operator": "kv_token_select", "args": parsed.to_dict()})
        return StructuralChange("runtime_kv_token_selection", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0, "full_query_length_preserved": True})

    def apply_sparse_runtime(self, model: FakeWanDiT, policy: Mapping[str, Any]) -> StructuralChange:
        parsed = SparseRuntimePolicy.from_mapping(policy)
        if model._harness4h3_attention_window_policy:
            raise ValueError("incompatible_runtime_attention_composition: attention_window and sparse_runtime")
        for block in model.blocks:
            if parsed.attention in {"attn1", "self", "all"}:
                block.attn1._harness4h3_sparse_policy = parsed.to_dict()
        model._harness4h3_sparse_runtime_policy = parsed.to_dict()
        model._harness4h3_runtime_operations.append({"operator": "sparse_runtime", "args": parsed.to_dict()})
        return StructuralChange("runtime_sparse_attention", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0})

    def apply_attention_window(self, model: FakeWanDiT, policy: Mapping[str, Any]) -> StructuralChange:
        parsed = AttentionWindowPolicy.from_mapping(policy)
        if model._harness4h3_sparse_runtime_policy:
            raise ValueError("incompatible_runtime_attention_composition: sparse_runtime and attention_window")
        for block in model.blocks:
            if parsed.attention in {"attn1", "self"}:
                block.attn1._harness4h3_attention_window_policy = parsed.to_dict()
        model._harness4h3_attention_window_policy = parsed.to_dict()
        model._harness4h3_runtime_operations.append({"operator": "attention_window", "args": parsed.to_dict()})
        return StructuralChange("runtime_attention_window", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0})

    def apply_sampling_step_policy(self, model: FakeWanDiT, policy: Mapping[str, Any]) -> StructuralChange:
        parsed = SamplingStepPolicy.from_mapping(policy)
        model._harness4h3_sampling_step_policy = parsed.to_dict()
        model._harness4h3_runtime_operations.append({"operator": "sampling_step_skip", "args": parsed.to_dict()})
        return StructuralChange("runtime_sampling_step_skip", {"policy": parsed.to_dict(), "dynamic_compute": True, "parameter_reduction_ratio": 0.0})

    def sampling_steps(self, model: FakeWanDiT, timesteps: Sequence[Any]) -> tuple[Any, ...]:
        policy = model._harness4h3_sampling_step_policy
        return select_sampling_steps(timesteps, policy) if policy else tuple(timesteps)

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
            "num_heads": int(model.blocks[0].attn1.heads),
            "attention_head_dim": int(model.blocks[0].attn1.inner_dim // model.blocks[0].attn1.heads),
            "runtime_gate_rules": list(model.runtime_gate_rules),
            "token_sparsity_policy": dict(model._harness4h3_token_sparsity_policy),
            "token_merge_policy": dict(model._harness4h3_token_merge_policy),
            "kv_token_selection_policy": dict(model._harness4h3_kv_token_selection_policy),
            "sparse_runtime_policy": dict(model._harness4h3_sparse_runtime_policy),
            "attention_window_policy": dict(model._harness4h3_attention_window_policy),
            "sampling_step_policy": dict(model._harness4h3_sampling_step_policy),
            "runtime_operations": [dict(item) for item in model._harness4h3_runtime_operations],
        }
        config_path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        digest = checkpoint_content_sha256(target)
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
            actual = checkpoint_content_sha256(root)
            if actual != artifact.sha256:
                raise ValueError("checkpoint_reload_failed: digest mismatch")
            model._artifact_sha256 = actual
        return model

    def profile_sample(self, model: FakeWanDiT, seed: int = 7) -> Any:
        """Create the fixed CPU probe shared by profiling and RSI evaluation."""

        torch, _ = _torch()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(2, int(model.hidden_size), generator=generator)

    def forward_profile(self, model: FakeWanDiT, sample: Any) -> Any:
        torch, _ = _torch()
        with torch.no_grad():
            return model(sample)

    def estimate_flops(self, model: FakeWanDiT) -> int:
        """Return a deterministic structural multiply-add estimate, not hardware latency."""

        hidden = int(model.hidden_size)
        output = 2 * hidden * hidden
        blocks = sum(4 * hidden * int(width) for width in model.ffn_widths)
        attention = sum(
            2 * int(block.attn1.inner_dim) * hidden + 2 * int(block.attn2.inner_dim) * hidden
            for block in model.blocks
        )
        estimate = float(output + blocks + attention)
        token_policy = getattr(model, "_harness4h3_token_sparsity_policy", {})
        token_merge_policy = getattr(model, "_harness4h3_token_merge_policy", {})
        kv_policy = getattr(model, "_harness4h3_kv_token_selection_policy", {})
        sparse_policy = getattr(model, "_harness4h3_sparse_runtime_policy", {})
        window_policy = getattr(model, "_harness4h3_attention_window_policy", {})
        sampling_policy = getattr(model, "_harness4h3_sampling_step_policy", {})
        if token_policy:
            estimate *= float(token_policy.get("keep_ratio", 1.0))
        if token_merge_policy:
            estimate *= float(token_merge_policy.get("keep_ratio", 1.0)) ** 2
        if kv_policy:
            estimate *= float(kv_policy.get("keep_ratio", 1.0))
        if sparse_policy:
            estimate *= float(sparse_policy.get("keep_ratio", 1.0))
        if window_policy:
            estimate *= min(1.0, float(window_policy.get("window_size", 64)) / 64.0)
        if sampling_policy:
            estimate *= float(sampling_policy.get("target_steps", 16)) / 16.0
        return int(round(estimate))

    def tica_calibration_inputs(self, model: FakeWanDiT, manifest: Any) -> tuple[Any, ...]:
        """Create deterministic CPU transformer inputs for TICA replay."""

        from ...evolution.tica_calibration import TICACalibrationInput

        torch, _ = _torch()
        samples = []
        for spec in manifest.specs:
            shape = tuple(int(value) for value in spec.latent_shape)
            if not shape or shape[-1] != int(model.hidden_size):
                raise ValueError("unsupported_tica_calibration_input: Fake hidden dimension mismatch")
            generator = torch.Generator(device="cpu").manual_seed(int(spec.seed))
            hidden_states = torch.randn(shape, generator=generator, device="cpu")
            encoder_hidden_states = torch.randn(
                (shape[0], int(model.hidden_size)), generator=generator, device="cpu"
            )
            samples.append(
                TICACalibrationInput(
                    spec=spec,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    metadata={
                        "device": "cpu",
                        "provider": "fake_wan_cpu",
                        "offline_simulation": True,
                    },
                )
            )
        return tuple(samples)

    def forward_denoising(self, model: FakeWanDiT, calibration_input: Any, timestep: float) -> Any:
        """Approximate one denoising transformer prediction at a normalized timestep."""

        torch, _ = _torch()
        hidden = calibration_input.hidden_states
        conditioning = calibration_input.encoder_hidden_states
        if tuple(hidden.shape) != tuple(conditioning.shape):
            raise ValueError("unsupported_tica_calibration_input: Fake tensor shapes differ")
        scale = 1.0 + 0.25 * float(timestep)
        with torch.no_grad():
            return model(hidden * scale + 0.05 * conditioning, timestep=float(timestep))

    def apply_timestep_block_gating(self, model: FakeWanDiT, rules: Sequence[Mapping[str, Any]]) -> StructuralChange:
        block_ids = [item.block_id for item in self.get_blocks(model)]
        validated = validate_timestep_gate_plan({"rules": [dict(item) for item in rules], "policy": "skip_when_in_range"}, block_ids)
        model.runtime_gate_rules = [rule.to_dict() for rule in validated]
        model.runtime_gate_stats = {str(block_id): {"executed": 0, "skipped": 0} for block_id in block_ids}
        model._harness4h3_runtime_operations.append({"operator": "timestep_block_gate", "args": {"rules": list(model.runtime_gate_rules), "policy": "skip_when_in_range"}})
        return StructuralChange(
            "runtime_timestep_gating",
            {
                "runtime_operator": "timestep_block_gate",
                "rules": [rule.to_dict() for rule in validated],
                "parameter_reduction_ratio": 0.0,
                "dynamic_compute": True,
            },
        )
