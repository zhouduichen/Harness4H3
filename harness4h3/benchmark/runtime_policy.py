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
        return workflow
    if kind == "inference_chunking":
        node = _node_by_class(workflow, "MiniMaxH3ImageToVideo")
        if node is None or not isinstance(node.get("inputs"), dict) or "chunk_size" not in node["inputs"]:
            raise BackendError("workflow has no H3 chunk_size input", "runtime_policy_unsupported")
        node["inputs"]["chunk_size"] = int(args.get("chunk_size", 0))
        return workflow
    raise BackendError("unsupported runtime policy: %s" % kind, "runtime_policy_unsupported")
