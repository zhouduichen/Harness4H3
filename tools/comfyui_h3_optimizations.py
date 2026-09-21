"""Opt-in MiniMax-H3 runtime nodes for the remote campaign.

This file is installed into ComfyUI/custom_nodes by the remote installer.  It
is deliberately opt-in: the normal H3 workflow and the normal Attention
implementation are unchanged until ``H3OptimizationConfig`` is connected.

The LPL node is a schedule reducer, not a weight-changing training operator.
It samples a verified base sigma schedule at fewer points and keeps the final
zero sigma required by ComfyUI's sampler contract.  TDTM is implemented as a
conservative inference-time merge of adjacent temporal video rows.  Text,
audio, references, conditioning rows, and temporal alignment are never
merged.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.samplers
from comfy_api.latest import ComfyExtension, io


H3_OPTIMIZATION_EXTENSION_VERSION = "0.1.0"


def _reduced_sigmas(model: Any, scheduler: str, steps: int, denoise: float, target_steps: int) -> torch.Tensor:
    """Return a shortened, endpoint-preserving sigma schedule."""

    steps = int(steps)
    target_steps = int(target_steps)
    if steps < 1 or target_steps < 1 or target_steps > steps:
        raise ValueError("target_steps must be in [1, steps]")
    total_steps = steps
    if float(denoise) < 1.0:
        if float(denoise) <= 0.0:
            return torch.zeros(0, dtype=torch.float32)
        total_steps = int(steps / float(denoise))
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, total_steps
    ).cpu()
    sigmas = sigmas[-(steps + 1) :]
    if target_steps == steps:
        return sigmas
    # The first sigma keeps the requested noise level, the last element is
    # ComfyUI's terminal zero.  Intermediate points are selected from the
    # original schedule rather than interpolated in sigma space.
    indices = torch.linspace(0, steps - 1, target_steps, dtype=torch.float64).round().long()
    indices = torch.unique_consecutive(indices).clamp(0, max(0, steps - 1))
    selected = sigmas.index_select(0, indices)
    return torch.cat((selected, sigmas[-1:].clone()), dim=0)


class H3LPLScheduler(io.ComfyNode):
    """Endpoint-safe LPL-style fewer-step scheduler for MiniMax-H3."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LPLScheduler",
            display_name="H3 LPL Scheduler",
            category="model/sampling/schedulers",
            description="Reduce H3 denoising steps using a verified base sigma schedule.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES),
                io.Int.Input("steps", default=32, min=1, max=10000),
                io.Int.Input("target_steps", default=16, min=1, max=10000),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, model, scheduler, steps, target_steps, denoise) -> io.NodeOutput:
        return io.NodeOutput(_reduced_sigmas(model, scheduler, steps, denoise, target_steps))


def _step_index(options: Mapping[str, Any], device: torch.device) -> int:
    sample_sigmas = options.get("sample_sigmas")
    sigma = options.get("sigmas")
    if not isinstance(sample_sigmas, torch.Tensor) or not isinstance(sigma, torch.Tensor):
        return -1
    if sample_sigmas.numel() == 0 or sigma.numel() == 0:
        return -1
    current = sigma.detach().reshape(-1)[0].to(device=device)
    return int((sample_sigmas.detach().to(device=device) - current).abs().argmin().item())


def _video_segment(layout: Any) -> Optional[Tuple[int, int, int, int]]:
    signature = getattr(layout, "signature", None)
    segments = getattr(layout, "segments", None)
    if not isinstance(signature, tuple) or len(signature) < 5 or not isinstance(segments, list):
        return None
    latent_t = int(signature[1])
    if latent_t < 2:
        return None
    for start, stop, kind in segments:
        if kind == "video":
            rows = int(stop) - int(start)
            if rows > 0 and rows % latent_t == 0:
                return int(start), int(stop), latent_t, rows // latent_t
    return None


def _merge_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    options: Mapping[str, Any],
    layout: Any,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]:
    """Merge similar adjacent temporal video rows and return inverse indices."""

    step_limit = int(options.get("harness4h3_tdtm_merge_steps", 0) or 0)
    threshold = float(options.get("harness4h3_tdtm_similarity_threshold", 0.985) or 0.985)
    if step_limit <= 0 or not 0.0 <= threshold <= 1.0:
        return None
    step = _step_index(options, x.device)
    if step < 0 or step >= step_limit:
        return None
    segment = _video_segment(layout)
    if segment is None:
        return None
    start, stop, latent_t, frame_rows = segment
    target_x = x[start:stop].reshape(latent_t, frame_rows, -1)
    pair_count = latent_t // 2
    if pair_count < 1:
        return None
    similarity = F.cosine_similarity(
        target_x[0 : 2 * pair_count : 2],
        target_x[1 : 2 * pair_count : 2],
        dim=-1,
    )
    merge = similarity >= threshold
    if not bool(merge.any().item()):
        return None

    target_count = stop - start
    drop = torch.zeros((latent_t, frame_rows), dtype=torch.bool, device=x.device)
    drop[1 : 2 * pair_count : 2] = merge
    keep_flat = ~drop.reshape(-1)
    group_grid = (torch.cumsum(keep_flat.to(torch.long), dim=0) - 1).reshape(latent_t, frame_rows)
    # A merged odd frame row maps to the corresponding spatial row in the
    # preceding even frame, not to the immediately preceding flattened token.
    # This distinction matters because H3 packs each frame in spatial-row
    # order, so a single temporal pair contains many independent groups.
    odd_groups = group_grid[1 : 2 * pair_count : 2]
    even_groups = group_grid[0 : 2 * pair_count : 2]
    group_grid[1 : 2 * pair_count : 2] = torch.where(merge, even_groups, odd_groups)
    group_ids = group_grid.reshape(-1)
    reduced_count = int(keep_flat.sum().item())
    target_q = q[start:stop]
    target_k = k[start:stop]
    target_v = v[start:stop]

    def reduce(value: torch.Tensor) -> torch.Tensor:
        result = value.new_zeros((reduced_count,) + tuple(value.shape[1:]))
        result.index_add_(0, group_ids, value)
        counts = value.new_zeros((reduced_count,))
        counts.index_add_(0, group_ids, torch.ones_like(group_ids, dtype=value.dtype))
        return result / counts.reshape((-1,) + (1,) * (value.ndim - 1))

    def join(value: torch.Tensor, reduced: torch.Tensor) -> torch.Tensor:
        return torch.cat((value[:start], reduced, value[stop:]), dim=0)

    inverse = group_ids
    return (
        join(q, reduce(target_q)),
        join(k, reduce(target_k)),
        join(v, reduce(target_v)),
        inverse,
        target_count,
        reduced_count,
    )


def _patched_attention_forward(self, x, rope_freqs=None, transformer_options=None):
    """H3 Attention.forward with an opt-in TDTM sequence reduction."""

    options = transformer_options if isinstance(transformer_options, dict) else {}
    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        import comfy.model_management

        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot
            )
        q, k = q[0], k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))

    layout = options.get("minimax_h3_layout")
    merged = None
    if rope_freqs is not None and layout is not None:
        merged = _merge_rows(q, k, v, x, options, layout)
    inverse = None
    original_target = reduced_target = 0
    if merged is not None:
        q, k, v, inverse, original_target, reduced_target = merged

    from comfy.ldm.minimax.model import AttentionTensorContainer, optimized_attention

    q_container = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k_container = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v_container = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(
        q_container,
        k_container,
        v_container,
        self.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=options,
    ).squeeze(0)
    if inverse is not None:
        start, stop, _, _ = _video_segment(layout)
        expanded = out.new_empty((original_target,) + tuple(out.shape[1:]))
        expanded.copy_(out[start : start + reduced_target].index_select(0, inverse))
        out = torch.cat((out[:start], expanded, out[start + reduced_target :]), dim=0)
        stats = options.setdefault("harness4h3_tdtm_stats", {})
        stats.update(
            {
                "step": _step_index(options, x.device),
                "original_tokens": int(s),
                "reduced_tokens": int(s - original_target + reduced_target),
                "merged_tokens": int(original_target - reduced_target),
                "merge_ratio": float((original_target - reduced_target) / max(1, s)),
            }
        )
    return self.out_proj(out)


def _install_tdtm_hook() -> bool:
    try:
        import comfy.ldm.minimax.model as minimax_model

        current = minimax_model.Attention.forward
        if getattr(current, "_harness4h3_tdtm", False):
            return True
        minimax_model.Attention.forward = _patched_attention_forward
        setattr(minimax_model.Attention.forward, "_harness4h3_tdtm", True)
        return True
    except Exception:
        # ComfyUI can import custom nodes before the MiniMax module exists.
        # The node itself installs the hook lazily when it is actually used.
        return False


class H3OptimizationConfig(io.ComfyNode):
    """Attach opt-in TDTM controls to a cloned H3 ModelPatcher."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3OptimizationConfig",
            display_name="H3 Optimization Config",
            category="model/optimization",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("merge_steps", default=0, min=0, max=10000),
                io.Float.Input("similarity_threshold", default=0.985, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[io.Model.Output(display_name="optimized_model")],
        )

    @classmethod
    def execute(cls, model, merge_steps, similarity_threshold) -> io.NodeOutput:
        if not 0.0 <= float(similarity_threshold) <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        _install_tdtm_hook()
        patched = model.clone()
        options = patched.model_options.setdefault("transformer_options", {})
        options["harness4h3_tdtm_merge_steps"] = int(merge_steps)
        options["harness4h3_tdtm_similarity_threshold"] = float(similarity_threshold)
        options["harness4h3_optimization_extension_version"] = H3_OPTIMIZATION_EXTENSION_VERSION
        return io.NodeOutput(patched)


class H3OptimizationExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [H3LPLScheduler, H3OptimizationConfig]


async def comfy_entrypoint() -> H3OptimizationExtension:
    _install_tdtm_hook()
    return H3OptimizationExtension()
