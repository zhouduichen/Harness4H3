"""Capability probing for optimizations around the real MiniMax-H3 runtime.

LPL, TDTM, and CI-DL are video-diffusion inference techniques, not generic
model-training operators.  This module keeps that distinction explicit and
fail-closed: a controller may use the manifest as evidence, but an
optimization is not executable merely because its name appears in the goal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return ""


def _load_workflow(path: Path) -> Mapping[str, Any]:
    import json

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, Mapping) else {}


def _workflow_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        return _load_workflow(Path(value))
    return {}


def _node_classes(workflow: Mapping[str, Any]) -> set[str]:
    return {
        str(node["class_type"])
        for node in workflow.values()
        if isinstance(node, Mapping) and node.get("class_type")
    }


def _workflow_hook_readiness(workflow: Mapping[str, Any]) -> Dict[str, bool]:
    """Check that the adapter has a real insertion point for each hook.

    An installed custom node is not enough: the benchmark adapter must be able
    to wire it into the exact API workflow that will be measured.  Keeping
    this check here makes the capability manifest fail closed before an LLM is
    allowed to put the recipe in a plan.
    """

    scheduler_ready = False
    model_loader_ready = False
    model_consumer_ready = False
    h3_generator_ready = False
    for node in workflow.values():
        if not isinstance(node, Mapping):
            continue
        class_type = str(node.get("class_type", "")).lower()
        inputs = node.get("inputs") if isinstance(node.get("inputs"), Mapping) else {}
        if class_type == "basicscheduler":
            scheduler_ready = scheduler_ready or {
                "model",
                "scheduler",
                "steps",
                "denoise",
            }.issubset(inputs)
        if class_type in {"unetloader", "unetloadergguf"}:
            model_loader_ready = True
        if class_type in {"cfgguider", "samplercustomadvanced"} and "model" in inputs:
            model_consumer_ready = True
        if class_type == "minimaxh3imagetovideo":
            h3_generator_ready = True
    return {
        "lpl": scheduler_ready,
        "tdtm": model_loader_ready and model_consumer_ready and h3_generator_ready,
    }


def probe_minimax_h3_capabilities(
    workflow: Any = None,
    comfyui_root: Optional[Any] = None,
    live_node_classes: Optional[Any] = None,
    runtime_evidence: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a compact, evidence-backed H3 optimization capability matrix.

    The probe does not load weights or start ComfyUI.  It checks the workflow
    and source-level hooks required for the three named techniques.  ``ci_dl``
    reports ComfyUI's existing dynamic block prefetch path as a baseline
    capability, not as a new checkpoint operator.
    """

    workflow_map = _workflow_mapping(workflow)
    classes = _node_classes(workflow_map)
    workflow_hooks = _workflow_hook_readiness(workflow_map)
    root = Path(comfyui_root).expanduser() if comfyui_root else None
    model_source = _read_text(root / "comfy/ldm/minimax/model.py") if root else ""
    model_base_source = _read_text(root / "comfy/model_base.py") if root else ""
    prefetch_source = _read_text(root / "comfy/model_prefetch.py") if root else ""
    patcher_source = _read_text(root / "comfy/model_patcher.py") if root else ""
    optimization_extension_source = _read_text(
        root / "custom_nodes/harness4h3_h3_optimizations.py"
    ) if root else ""

    has_lpl_node = any(
        name.lower() in {"lplscheduler", "linearproportionalleap", "linearleap scheduler"}
        for name in classes
    )
    has_lpl_extension = all(
        marker in optimization_extension_source
        for marker in ("H3LPLScheduler", "H3_OPTIMIZATION_EXTENSION_VERSION", "_reduced_sigmas")
    )
    live_classes = None
    if live_node_classes is not None:
        if isinstance(live_node_classes, Mapping):
            live_classes = {str(name) for name in live_node_classes}
        elif isinstance(live_node_classes, (list, tuple, set, frozenset)):
            live_classes = {str(name) for name in live_node_classes}
        else:
            live_classes = set()
    live_lpl = live_classes is None or "H3LPLScheduler" in live_classes
    has_lpl_input = any(
        isinstance(node, Mapping)
        and isinstance(node.get("inputs"), Mapping)
        and any("leap" in str(key).lower() or "sigma" in str(key).lower() for key in node["inputs"])
        for node in workflow_map.values()
    )
    has_tdtm_hook = any(
        marker in model_source.lower()
        for marker in ("tdtm", "temporal_token_merge", "temporal_token_merging", "token_merge")
    )
    has_tdtm_extension = all(
        marker in optimization_extension_source
        for marker in ("H3OptimizationConfig", "_patched_attention_forward", "_merge_rows")
    )
    live_tdtm = live_classes is None or "H3OptimizationConfig" in live_classes
    has_dynamic_block_path = all(
        marker in source
        for source, marker in (
            (model_source, "make_prefetch_queue"),
            (model_source, "prefetch_queue_pop"),
            (model_base_source, "current_patcher.is_dynamic"),
            (prefetch_source, "prefetch_dynamic_vbars"),
            (patcher_source, "def _load_list"),
        )
    )
    runtime_map = runtime_evidence if isinstance(runtime_evidence, Mapping) else {}
    runtime_confirmed = runtime_map.get("dynamic_vram_enabled") is True

    return {
        "schema_version": 1,
        "model_family": "MiniMax-H3-FL2VA",
        "probe": {
            "workflow_classes": sorted(classes),
            "workflow_loaded": bool(workflow_map),
            "comfyui_source_root_present": bool(root and root.is_dir()),
            "workflow_hooks": dict(workflow_hooks),
        },
        "lpl": {
            "status": "executable" if has_lpl_extension and live_lpl and workflow_hooks["lpl"] else "not_executable",
            "safe_to_plan": bool(has_lpl_extension and live_lpl and workflow_hooks["lpl"]),
            "kind": "inference_only",
            "execution_contract": {
                "operator": "step_distill",
                "operator_args": "lpl_target_steps",
                "workflow_hook": "H3LPLScheduler",
                "materialization": "ModelState.runtime_state.h3_optimizations.lpl",
                "creates_checkpoint": False,
            },
            "reason": (
                "workflow exposes the installed H3 LPL scheduler"
                if has_lpl_extension and live_lpl and workflow_hooks["lpl"]
                else "H3 LPL extension is installed but the live ComfyUI node is not registered"
                if has_lpl_extension and live_classes is not None and not live_lpl
                else "H3 LPL extension is installed but the benchmark workflow has no compatible BasicScheduler hook"
                if has_lpl_extension and live_lpl and not workflow_hooks["lpl"]
                else "no verified H3 LPL extension; integer BasicScheduler steps are not LPL"
            ),
            "extension_installed": has_lpl_extension,
            "live_node_registered": live_lpl if live_classes is not None else None,
            "workflow_hook_ready": workflow_hooks["lpl"],
        },
        "tdtm": {
            "status": "executable" if has_tdtm_extension and live_tdtm and workflow_hooks["tdtm"] else "not_executable",
            "safe_to_plan": bool(has_tdtm_extension and live_tdtm and workflow_hooks["tdtm"]),
            "kind": "inference_only",
            "execution_contract": {
                "operator": "step_distill",
                "operator_args": "tdtm_merge_steps,tdtm_similarity_threshold",
                "workflow_hook": "H3OptimizationConfig",
                "materialization": "ModelState.runtime_state.h3_optimizations.tdtm",
                "creates_checkpoint": False,
            },
            "reason": (
                "installed H3 extension exposes a verified temporal token merge hook"
                if has_tdtm_extension and live_tdtm and workflow_hooks["tdtm"]
                else "H3 TDTM extension is installed but the live ComfyUI node is not registered"
                if has_tdtm_extension and live_classes is not None and not live_tdtm
                else "H3 TDTM extension is installed but the benchmark workflow has no compatible H3 model hook"
                if has_tdtm_extension and live_tdtm and not workflow_hooks["tdtm"]
                else "H3 packed AV forward has no verified temporal-token merge hook; audio/video alignment must be preserved"
            ),
            "extension_installed": has_tdtm_extension,
            "live_node_registered": live_tdtm if live_classes is not None else None,
            "workflow_hook_ready": workflow_hooks["tdtm"],
        },
        "ci_dl": {
            "status": (
                "active"
                if has_dynamic_block_path and runtime_confirmed
                else "baseline"
                if has_dynamic_block_path
                else "not_verified"
            ),
            "safe_to_plan": False,
            "safe_to_measure": bool(has_dynamic_block_path),
            "planning_mode": "measure_only",
            "kind": "runtime_memory_baseline",
            "execution_contract": {
                "operator": None,
                "workflow_hook": "ComfyUI dynamic VBAR block prefetch",
                "measurement": "runtime_evidence.dynamic_vram_enabled",
                "evidence": [
                    "runtime_evidence.dynamic_vram_enabled",
                    "evaluation.hardware.peak_memory_gb",
                    "evaluation.quality_metrics.power_sampling",
                ],
                "creates_checkpoint": False,
                "planning": "measure_existing_path",
            },
            "implementation": "ComfyUI dynamic VBAR block prefetch" if has_dynamic_block_path else None,
            "already_active_when": "current_patcher.is_dynamic()" if has_dynamic_block_path else None,
            "runtime_confirmed": runtime_confirmed if runtime_evidence is not None else None,
            "reason": (
                "live ComfyUI startup evidence confirms dynamic VBAR block prefetch; it is automatic, so do not create a fake child checkpoint"
                if has_dynamic_block_path and runtime_confirmed
                else "H3 already iterates transformer blocks through ComfyUI dynamic prefetch; measure it, do not create a fake child checkpoint"
                if has_dynamic_block_path
                else "required dynamic block-loading hooks were not found"
            ),
        },
    }


__all__ = ["probe_minimax_h3_capabilities"]
