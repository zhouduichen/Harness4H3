from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..backends.comfyui import BackendError


def _node_by_class(workflow: Mapping[str, Any], class_type: str) -> Optional[Dict[str, Any]]:
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return node
    return None


def _policy_args(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    args = policy.get("args")
    return args if isinstance(args, Mapping) else {}


def apply_runtime_policy(workflow: Dict[str, Any], policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply one runtime policy to an already copied API-format workflow.

    Only inputs known to the workflow are touched. This keeps the operation
    compatible with different ComfyUI/custom-node versions and makes missing
    capabilities an explicit, recorded branch failure.
    """
    if not isinstance(policy, Mapping):
        raise BackendError("runtime policy must be a mapping", "runtime_policy_unsupported")
    kind = str(policy.get("kind", ""))
    args = _policy_args(policy)
    if kind == "runtime_offload":
        lora = _node_by_class(workflow, "MiniMaxH3TurboLoRA")
        attention = _node_by_class(workflow, "PathchSageAttentionKJ")
        if lora is None or not isinstance(lora.get("inputs"), dict) or "low_vram" not in lora["inputs"]:
            raise BackendError("workflow has no low_vram LoRA control", "runtime_policy_unsupported")
        if str(args.get("mode", "")) not in {"balanced", "aggressive"}:
            raise BackendError("runtime_offload.mode is unsupported", "runtime_policy_unsupported")
        lora["inputs"]["low_vram"] = True
        if attention is not None and isinstance(attention.get("inputs"), dict) and "allow_compile" in attention["inputs"]:
            attention["inputs"]["allow_compile"] = False
        return workflow
    if kind == "vae_tiling":
        decoder = _node_by_class(workflow, "VAEDecode")
        if decoder is None or not isinstance(decoder.get("inputs"), dict):
            raise BackendError("workflow has no VAEDecode node", "runtime_policy_unsupported")
        if not {"tile_size", "overlap"}.issubset(args):
            raise BackendError("vae_tiling requires tile_size and overlap", "runtime_policy_unsupported")
        decoder["class_type"] = "VAEDecodeTiled"
        decoder["inputs"]["tile_size"] = int(args["tile_size"])
        decoder["inputs"]["overlap"] = int(args["overlap"])
        # ComfyUI's tiled decoder exposes temporal controls as required
        # inputs for video VAEs. Keep them deterministic while allowing the
        # controller to choose the primary spatial intervention only.
        decoder["inputs"].setdefault("temporal_size", 64)
        decoder["inputs"].setdefault("temporal_overlap", 8)
        return workflow
    if kind == "inference_chunking":
        node = _node_by_class(workflow, "MiniMaxH3ImageToVideo")
        if node is None or not isinstance(node.get("inputs"), dict) or "chunk_size" not in node["inputs"]:
            raise BackendError("workflow has no H3 chunk_size input", "runtime_policy_unsupported")
        node["inputs"]["chunk_size"] = int(args.get("chunk_size", 0))
        return workflow
    if kind == "component_lifecycle_optimize":
        expected = {
            "unload_text_encoder_after_encode",
            "offload_vae_until_decode",
            "free_cache_before_decode",
        }
        if set(args) != expected or any(not isinstance(args[key], bool) for key in expected):
            raise BackendError("component_lifecycle_optimize controls are unsupported", "runtime_policy_unsupported")
        clip = _node_by_class(workflow, "CLIPLoaderGGUF")
        decoder = _node_by_class(workflow, "VAEDecode")
        if args["unload_text_encoder_after_encode"]:
            if clip is None or not isinstance(clip.get("inputs"), dict) or "unload_after_encode" not in clip["inputs"]:
                raise BackendError("workflow has no text-encoder lifecycle control", "runtime_policy_unsupported")
            clip["inputs"]["unload_after_encode"] = True
        if args["offload_vae_until_decode"]:
            if decoder is None or not isinstance(decoder.get("inputs"), dict):
                raise BackendError("workflow has no VAE lifecycle control", "runtime_policy_unsupported")
            key = next((name for name in ("offload_device", "device") if name in decoder["inputs"]), None)
            if key is None:
                raise BackendError("workflow has no VAE offload control", "runtime_policy_unsupported")
            decoder["inputs"][key] = "cpu"
        if args["free_cache_before_decode"]:
            if decoder is None or not isinstance(decoder.get("inputs"), dict) or "free_cache_before_decode" not in decoder["inputs"]:
                raise BackendError("workflow has no decode cache-release control", "runtime_policy_unsupported")
            decoder["inputs"]["free_cache_before_decode"] = True
        return workflow
    if kind == "vae_decode_offload":
        if str(args.get("mode", "")) not in {"balanced", "cpu"}:
            raise BackendError("vae_decode_offload.mode is unsupported", "runtime_policy_unsupported")
        decoder = _node_by_class(workflow, "VAEDecode")
        if decoder is None or not isinstance(decoder.get("inputs"), dict):
            raise BackendError("workflow has no VAE decode node", "runtime_policy_unsupported")
        key = next((name for name in ("offload_device", "device") if name in decoder["inputs"]), None)
        if key is None:
            raise BackendError("workflow has no VAE offload control", "runtime_policy_unsupported")
        decoder["inputs"][key] = "cpu" if args["mode"] == "cpu" else "auto"
        return workflow
    if kind == "cache_release":
        if str(args.get("stage", "")) not in {"before_decode", "between_stages", "always"}:
            raise BackendError("cache_release.stage is unsupported", "runtime_policy_unsupported")
        # The ComfyUI adapter performs the release at the nearest safe task
        # boundary. There is no workflow mutation when the backend exposes no
        # stage callback, so the capability is handled explicitly by H3Runner.
        return workflow
    raise BackendError("unsupported runtime policy: %s" % kind, "runtime_policy_unsupported")
