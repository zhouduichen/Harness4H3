from __future__ import annotations

from harness4h3.campaign.base import ActorIdentity, CampaignBase


def make_base(**overrides):
    values = {
        "schema_version": 1,
        "campaign_id": "camp_test_0001",
        "target_profile": {"id": "mobile", "constraints": {"max_latency_s": 3.0}},
        "target_profile_hash": "sha256:target",
        "verifier_bank": {"version": "v1"},
        "verifier_bank_hash": "sha256:verifier",
        "dataset_manifest_hash": "sha256:dataset",
        "evaluation_recipe_hash": "sha256:evaluation",
        "controller_identity": ActorIdentity("controller", "model-a", "1"),
        "critic_identity": ActorIdentity("critic", "model-b", "1"),
        "evaluator_identity": ActorIdentity("evaluator", "fixed-v1", "1"),
        "prompt_version": "prompt-v1",
        "capability_snapshot": {"version": "cap-v1"},
    }
    values.update(overrides)
    return CampaignBase(**values)
