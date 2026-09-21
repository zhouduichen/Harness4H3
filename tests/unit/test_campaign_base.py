from __future__ import annotations

from dataclasses import replace

import pytest

from harness4h3.campaign.base import ActorIdentity, CampaignBaseError
from tests.unit.campaign_fixtures import make_base


def test_campaign_base_digest_is_stable_and_changes_when_target_changes():
    base = make_base()
    same = type(base).from_dict(base.to_dict())
    changed = replace(base, target_profile={"id": "mobile-v2"})
    assert same.digest == base.digest
    assert changed.digest != base.digest


def test_actor_identities_must_be_pairwise_distinct():
    with pytest.raises(CampaignBaseError, match="pairwise"):
        make_base(critic_identity=make_base().controller_identity)


def test_campaign_base_rejects_empty_hashes_and_nan_payloads():
    with pytest.raises(CampaignBaseError, match="target_profile_hash"):
        make_base(target_profile_hash="")
    with pytest.raises(CampaignBaseError, match="finite"):
        make_base(target_profile={"value": float("nan")})


def test_actor_identity_round_trips_as_value_data():
    actor = ActorIdentity("provider", "model", "v1")
    assert ActorIdentity.from_dict(actor.to_dict()) == actor
