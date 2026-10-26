import yaml
import pytest

from tools.rsi_m0_preflight import preflight


CONFIG = "configs/rsi-harness/m0.yaml"


def test_offline_preflight_covers_bundle_components_failure_resume_and_costs(tmp_path):
    result = preflight(CONFIG, tmp_path / "bundles", tmp_path / "traces")
    assert result["passed"] is True
    assert result["bundle"]["file_count"] == 3
    assert all(item["passed"] for item in result["component_fixtures"].values())
    assert result["research_job_fixture"]["failure_charged"] is True
    assert result["research_job_fixture"]["duplicate_replayed"] is True
    assert result["research_job_fixture"]["identity_conflict_rejected"] is True
    assert result["budget"]["job_count_actual"] == 2
    assert result["trace"]["cost_summary"]["estimated_usd"]["status"] == "unavailable"
    assert result["execution_boundary"]["gpu"] == "not_used"


def test_preflight_rejects_duplicate_or_unallowlisted_editable_components(tmp_path):
    raw = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    raw["editable_components"]["retrieval"] = raw["editable_components"]["context"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="component_duplicate"):
        preflight(bad, tmp_path / "bundles", tmp_path / "traces")
