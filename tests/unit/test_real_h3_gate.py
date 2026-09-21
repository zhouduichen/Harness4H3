from __future__ import annotations

import json

from tools.run_real_h3_gate import preflight


def _config(tmp_path):
    parent = tmp_path / "parent.safetensors"
    parent.write_bytes(b"parent")
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    config = tmp_path / "worker.json"
    config.write_text(
        json.dumps(
            {
                "comfyui_root": str(comfy),
                "model_checkpoint": str(parent),
                "cache_dir": str(tmp_path / "cache"),
                "output_dir": str(tmp_path / "out"),
                "world_size": 4,
                "trainer_command": ["torchrun", "tools/h3_real_train_worker.py"],
            }
        ),
        encoding="utf-8",
    )
    return parent, config


def test_real_gate_accepts_only_real_trainer_contract(tmp_path):
    parent, config = _config(tmp_path)
    report = preflight(parent, config, require_gpu=False, check_endpoint=False)
    assert report["passed"] is True
    assert report["real_evidence_required"] is True
    assert report["offline_simulation"] is False


def test_real_gate_rejects_tiny_or_fake_worker(tmp_path):
    parent, config = _config(tmp_path)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["trainer_command"] = ["python", "tools/tiny_training_worker.py"]
    config.write_text(json.dumps(raw), encoding="utf-8")
    report = preflight(parent, config, require_gpu=False, check_endpoint=False)
    assert report["passed"] is False
    assert "real_h3_trainer" in report["errors"]


def test_real_gate_fails_closed_for_missing_gpu_prerequisite(tmp_path, monkeypatch):
    parent, config = _config(tmp_path)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = preflight(parent, config, require_gpu=True, check_endpoint=False)
    assert report["passed"] is False
    assert "cuda_available" in report["errors"]
    assert "cuda_world_size" in report["errors"]
