from pathlib import Path

import yaml

from tools.wan_loss_ablation import run


def test_loss_ablation_manifest_is_four_one_hot_rows(tmp_path):
    source = yaml.safe_load(Path("configs/wan-p1-loss-ablations.yaml").read_text(encoding="utf-8"))
    source["base"]["parent_checkpoint"] = "/tmp/frozen-r2.safetensors"
    config = tmp_path / "ablations.yaml"
    config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    result = run(config, tmp_path / "out", worker_python="python", worker_path=Path("worker.py"), dry_run=True)
    assert result["status"] == "dry_run"
    assert result["variant_count"] == 4
    assert [item["name"] for item in result["variants"]] == ["endpoint", "flow", "clean", "rollout"]
    assert all(item["status"] == "dry_run" for item in result["variants"])
