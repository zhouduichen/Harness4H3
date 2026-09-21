from __future__ import annotations

from harness4h3.campaign.capabilities import CapabilityRegistry
from harness4h3.operators.model_evolution import build_model_evolution_registry


def test_snapshot_exposes_only_available_registered_capabilities():
    registry = build_model_evolution_registry()
    snapshot = CapabilityRegistry.from_operator_registry(
        registry,
        {
            "distill": {"available": True, "backend": "tiny-reference", "evidence_level": "V2"},
            "dmd2": {"available": False, "reason": "backend_not_installed"},
        },
    )
    assert "distill" in snapshot.available_names()
    assert "dmd2" not in snapshot.available_names()
    assert snapshot.by_name("dmd2").reason == "backend_not_installed"


def test_missing_backend_status_is_fail_closed():
    registry = build_model_evolution_registry()
    snapshot = CapabilityRegistry.from_operator_registry(registry, {})
    assert snapshot.available_names() == ()
    assert all(item.available is False for item in snapshot.capabilities)


def test_snapshot_digest_changes_when_backend_evidence_changes():
    registry = build_model_evolution_registry()
    first = CapabilityRegistry.from_operator_registry(
        registry, {"distill": {"available": True, "backend": "a", "evidence_level": "V2"}}
    )
    second = CapabilityRegistry.from_operator_registry(
        registry, {"distill": {"available": True, "backend": "b", "evidence_level": "V2"}}
    )
    assert first.digest != second.digest
