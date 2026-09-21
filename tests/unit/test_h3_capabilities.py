import json

from harness4h3.benchmark.capabilities import probe_minimax_h3_capabilities


def test_h3_capability_probe_distinguishes_baseline_and_unavailable_methods(tmp_path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "comfy/ldm/minimax").mkdir(parents=True)
    (comfy / "comfy").mkdir(exist_ok=True)
    (comfy / "comfy/ldm/minimax/model.py").write_text(
        "make_prefetch_queue prefetch_queue_pop blocks", encoding="utf-8"
    )
    (comfy / "comfy/model_base.py").write_text("current_patcher.is_dynamic()", encoding="utf-8")
    (comfy / "comfy/model_prefetch.py").write_text("prefetch_dynamic_vbars", encoding="utf-8")
    (comfy / "comfy/model_patcher.py").write_text("def _load_list", encoding="utf-8")
    workflow = json.loads(
        '{"1":{"class_type":"BasicScheduler","inputs":{"steps":32}},'
        '"2":{"class_type":"MiniMaxH3ImageToVideo","inputs":{}}}'
    )

    result = probe_minimax_h3_capabilities(workflow, comfy)

    assert result["lpl"]["status"] == "not_executable"
    assert result["tdtm"]["status"] == "not_executable"
    assert result["ci_dl"]["status"] == "baseline"
    assert result["ci_dl"]["safe_to_plan"] is False
    assert result["ci_dl"]["safe_to_measure"] is True
    assert result["ci_dl"]["planning_mode"] == "measure_only"
    assert "evaluation.hardware.peak_memory_gb" in result["ci_dl"]["execution_contract"]["evidence"]
    assert "dynamic VBAR" in result["ci_dl"]["implementation"]


def test_h3_capability_probe_recognizes_installed_runtime_extension(tmp_path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "custom_nodes").mkdir(parents=True)
    (comfy / "custom_nodes/harness4h3_h3_optimizations.py").write_text(
        "H3LPLScheduler H3_OPTIMIZATION_EXTENSION_VERSION _reduced_sigmas "
        "H3OptimizationConfig _patched_attention_forward _merge_rows",
        encoding="utf-8",
    )

    result = probe_minimax_h3_capabilities(
        {
            "132": {
                "class_type": "BasicScheduler",
                "inputs": {"model": ["135", 0], "scheduler": "simple", "steps": 32, "denoise": 1.0},
            },
            "135": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors"}},
            "138": {"class_type": "CFGGuider", "inputs": {"model": ["135", 0]}},
            "139": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {}},
        },
        comfy,
    )

    assert result["lpl"]["status"] == "executable"
    assert result["lpl"]["safe_to_plan"] is True
    assert result["tdtm"]["status"] == "executable"
    assert result["tdtm"]["safe_to_plan"] is True
    assert result["lpl"]["execution_contract"]["creates_checkpoint"] is False
    assert result["tdtm"]["execution_contract"]["workflow_hook"] == "H3OptimizationConfig"


def test_h3_capability_probe_requires_live_node_registration(tmp_path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "custom_nodes").mkdir(parents=True)
    (comfy / "custom_nodes/harness4h3_h3_optimizations.py").write_text(
        "H3LPLScheduler H3_OPTIMIZATION_EXTENSION_VERSION _reduced_sigmas "
        "H3OptimizationConfig _patched_attention_forward _merge_rows",
        encoding="utf-8",
    )

    result = probe_minimax_h3_capabilities({}, comfy, live_node_classes={"BasicScheduler": {}})

    assert result["lpl"]["extension_installed"] is True
    assert result["lpl"]["live_node_registered"] is False
    assert result["lpl"]["safe_to_plan"] is False
    assert result["tdtm"]["safe_to_plan"] is False


def test_h3_capability_probe_requires_workflow_insertion_points(tmp_path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "custom_nodes").mkdir(parents=True)
    (comfy / "custom_nodes/harness4h3_h3_optimizations.py").write_text(
        "H3LPLScheduler H3_OPTIMIZATION_EXTENSION_VERSION _reduced_sigmas "
        "H3OptimizationConfig _patched_attention_forward _merge_rows",
        encoding="utf-8",
    )

    result = probe_minimax_h3_capabilities(
        {"1": {"class_type": "BasicScheduler", "inputs": {"steps": 32}}},
        comfy,
        live_node_classes={"H3LPLScheduler": {}, "H3OptimizationConfig": {}},
    )

    assert result["probe"]["workflow_hooks"] == {"lpl": False, "tdtm": False}
    assert result["lpl"]["safe_to_plan"] is False
    assert result["tdtm"]["safe_to_plan"] is False
    assert "workflow" in result["lpl"]["reason"]


def test_h3_capability_probe_marks_ci_dl_active_only_with_runtime_evidence(tmp_path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "comfy/ldm/minimax").mkdir(parents=True)
    (comfy / "comfy").mkdir(exist_ok=True)
    (comfy / "comfy/ldm/minimax/model.py").write_text(
        "make_prefetch_queue prefetch_queue_pop", encoding="utf-8"
    )
    (comfy / "comfy/model_base.py").write_text("current_patcher.is_dynamic()", encoding="utf-8")
    (comfy / "comfy/model_prefetch.py").write_text("prefetch_dynamic_vbars", encoding="utf-8")
    (comfy / "comfy/model_patcher.py").write_text("def _load_list", encoding="utf-8")

    result = probe_minimax_h3_capabilities(
        {}, comfy, runtime_evidence={"dynamic_vram_enabled": True}
    )

    assert result["ci_dl"]["status"] == "active"
    assert result["ci_dl"]["runtime_confirmed"] is True
    assert result["ci_dl"]["safe_to_plan"] is False
    assert result["ci_dl"]["planning_mode"] == "measure_only"
