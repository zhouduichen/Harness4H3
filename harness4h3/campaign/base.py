"""Immutable campaign identity and canonical digests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


class CampaignBaseError(ValueError):
    """Raised when a campaign verification base is malformed."""


def sha256_file(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file():
        raise CampaignBaseError("content identity path is not a file: %s" % path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or directory's bytes and relative names deterministically."""

    path = Path(path).resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise CampaignBaseError("content identity path does not exist: %s" % path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values deterministically and fail closed."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CampaignBaseError("value is not canonical JSON: %s" % exc) from exc


def canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _verify_canonical_content_hash(value: str, content: Any, name: str) -> None:
    """Reject tampering for new canonical hashes while reading legacy fixtures.

    Older campaign fixtures used shorthand values such as ``sha256:target``.
    They remain readable for compatibility, but every real campaign base emits
    a full 256-bit digest and those digests are always checked against content.
    """

    hash_content = content
    if name == "capability_snapshot_hash" and isinstance(content, Mapping) and "digest" in content:
        # CapabilitySnapshot.to_dict() carries its own digest.  Exclude that
        # derived field to avoid hashing a value that contains itself.
        hash_content = dict(content)
        hash_content.pop("digest", None)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) and value != canonical_digest(hash_content):
        raise CampaignBaseError("%s does not match its immutable content" % name)


def _copy_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignBaseError("%s must be a mapping" % name)
    try:
        copied = copy.deepcopy(dict(value))
        canonical_json(copied)
    except (TypeError, ValueError) as exc:
        raise CampaignBaseError("%s must contain finite JSON-compatible values" % name) from exc
    return copied


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignBaseError("%s must be a non-empty string" % name)
    return value.strip()


@dataclass(frozen=True)
class ActorIdentity:
    provider: str
    model: str
    version: str

    def __post_init__(self) -> None:
        _nonempty(self.provider, "actor.provider")
        _nonempty(self.model, "actor.model")
        _nonempty(self.version, "actor.version")

    def to_dict(self) -> Dict[str, str]:
        return {"provider": self.provider, "model": self.model, "version": self.version}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActorIdentity":
        if not isinstance(raw, Mapping):
            raise CampaignBaseError("actor identity must be a mapping")
        unknown = sorted(set(raw) - {"provider", "model", "version"})
        if unknown:
            raise CampaignBaseError("actor identity has unknown field(s): %s" % ", ".join(map(str, unknown)))
        missing = sorted({"provider", "model", "version"} - set(raw))
        if missing:
            raise CampaignBaseError("actor identity is missing field(s): %s" % ", ".join(missing))
        return cls(str(raw["provider"]), str(raw["model"]), str(raw["version"]))


@dataclass(frozen=True)
class CampaignBase:
    schema_version: int
    campaign_id: str
    target_profile: Mapping[str, Any]
    target_profile_hash: str
    verifier_bank: Mapping[str, Any]
    verifier_bank_hash: str
    dataset_manifest_hash: str
    evaluation_recipe_hash: str
    controller_identity: ActorIdentity
    critic_identity: ActorIdentity
    evaluator_identity: ActorIdentity
    prompt_version: str
    capability_snapshot: Mapping[str, Any]
    teacher_checkpoint_sha256: Optional[str] = None
    evaluation_manifest_digest: Optional[str] = None
    evaluator_version: str = ""
    clip_model_identity: str = ""
    clip_model_hash: Optional[str] = None
    capability_snapshot_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or int(self.schema_version) != 1:
            raise CampaignBaseError("schema_version must be 1")
        _nonempty(self.campaign_id, "campaign_id")
        for name in (
            "target_profile_hash",
            "verifier_bank_hash",
            "dataset_manifest_hash",
            "evaluation_recipe_hash",
            "prompt_version",
        ):
            _nonempty(getattr(self, name), name)
        for name in ("target_profile", "verifier_bank", "capability_snapshot"):
            object.__setattr__(self, name, _copy_mapping(getattr(self, name), name))
        identities = (self.controller_identity, self.critic_identity, self.evaluator_identity)
        if not all(isinstance(item, ActorIdentity) for item in identities):
            raise CampaignBaseError("all campaign actors must be ActorIdentity values")
        if len(set(identities)) != len(identities):
            raise CampaignBaseError("controller, critic, and evaluator identities must be pairwise distinct")
        for name in ("teacher_checkpoint_sha256", "evaluation_manifest_digest", "clip_model_hash", "capability_snapshot_hash"):
            value = getattr(self, name)
            if value is not None:
                _nonempty(value, name)
        for name in ("evaluator_version", "clip_model_identity"):
            value = getattr(self, name)
            if value:
                _nonempty(value, name)
        if self.capability_snapshot_hash is None:
            capability_content = dict(self.capability_snapshot)
            capability_content.pop("digest", None)
            object.__setattr__(self, "capability_snapshot_hash", canonical_digest(capability_content))
        _verify_canonical_content_hash(self.target_profile_hash, self.target_profile, "target_profile_hash")
        _verify_canonical_content_hash(self.verifier_bank_hash, self.verifier_bank, "verifier_bank_hash")
        _verify_canonical_content_hash(self.capability_snapshot_hash, self.capability_snapshot, "capability_snapshot_hash")
        try:
            self._payload()
        except (TypeError, ValueError) as exc:
            raise CampaignBaseError("campaign base contains non-finite or non-JSON values") from exc

    def _payload(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "campaign_id": self.campaign_id,
            "target_profile": copy.deepcopy(dict(self.target_profile)),
            "target_profile_hash": self.target_profile_hash,
            "verifier_bank": copy.deepcopy(dict(self.verifier_bank)),
            "verifier_bank_hash": self.verifier_bank_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "evaluation_recipe_hash": self.evaluation_recipe_hash,
            "controller_identity": self.controller_identity.to_dict(),
            "critic_identity": self.critic_identity.to_dict(),
            "evaluator_identity": self.evaluator_identity.to_dict(),
            "prompt_version": self.prompt_version,
            "capability_snapshot": copy.deepcopy(dict(self.capability_snapshot)),
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "evaluation_manifest_digest": self.evaluation_manifest_digest,
            "evaluator_version": self.evaluator_version,
            "clip_model_identity": self.clip_model_identity,
            "clip_model_hash": self.clip_model_hash,
            "capability_snapshot_hash": self.capability_snapshot_hash,
        }

    def payload(self) -> Mapping[str, Any]:
        return self._payload()

    @property
    def digest(self) -> str:
        return canonical_digest(self._payload())

    def to_dict(self) -> Dict[str, Any]:
        value = self._payload()
        value["base_digest"] = self.digest
        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CampaignBase":
        if not isinstance(raw, Mapping):
            raise CampaignBaseError("campaign base must be a mapping")
        allowed = {
            "schema_version",
            "campaign_id",
            "target_profile",
            "target_profile_hash",
            "verifier_bank",
            "verifier_bank_hash",
            "dataset_manifest_hash",
            "evaluation_recipe_hash",
            "controller_identity",
            "critic_identity",
            "evaluator_identity",
            "prompt_version",
            "capability_snapshot",
            "teacher_checkpoint_sha256",
            "evaluation_manifest_digest",
            "evaluator_version",
            "clip_model_identity",
            "clip_model_hash",
            "capability_snapshot_hash",
            "base_digest",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise CampaignBaseError("campaign base has unknown field(s): %s" % ", ".join(map(str, unknown)))
        required = sorted(allowed - {"base_digest"} - set(raw))
        if required:
            raise CampaignBaseError("campaign base is missing field(s): %s" % ", ".join(required))
        base = cls(
            schema_version=int(raw["schema_version"]),
            campaign_id=str(raw["campaign_id"]),
            target_profile=_copy_mapping(raw["target_profile"], "target_profile"),
            target_profile_hash=str(raw["target_profile_hash"]),
            verifier_bank=_copy_mapping(raw["verifier_bank"], "verifier_bank"),
            verifier_bank_hash=str(raw["verifier_bank_hash"]),
            dataset_manifest_hash=str(raw["dataset_manifest_hash"]),
            evaluation_recipe_hash=str(raw["evaluation_recipe_hash"]),
            controller_identity=ActorIdentity.from_dict(raw["controller_identity"]),
            critic_identity=ActorIdentity.from_dict(raw["critic_identity"]),
            evaluator_identity=ActorIdentity.from_dict(raw["evaluator_identity"]),
            prompt_version=str(raw["prompt_version"]),
            capability_snapshot=_copy_mapping(raw["capability_snapshot"], "capability_snapshot"),
            teacher_checkpoint_sha256=str(raw["teacher_checkpoint_sha256"]) if raw.get("teacher_checkpoint_sha256") is not None else None,
            evaluation_manifest_digest=str(raw["evaluation_manifest_digest"]) if raw.get("evaluation_manifest_digest") is not None else None,
            evaluator_version=str(raw.get("evaluator_version", "")),
            clip_model_identity=str(raw.get("clip_model_identity", "")),
            clip_model_hash=str(raw["clip_model_hash"]) if raw.get("clip_model_hash") is not None else None,
            capability_snapshot_hash=str(raw["capability_snapshot_hash"]) if raw.get("capability_snapshot_hash") is not None else None,
        )
        stored = raw.get("base_digest")
        if stored is not None and stored != base.digest:
            raise CampaignBaseError("base_digest mismatch")
        return base

    def assert_event_base(self, base_digest: str) -> None:
        if base_digest != self.digest:
            raise CampaignBaseError("event base digest does not match campaign base")
