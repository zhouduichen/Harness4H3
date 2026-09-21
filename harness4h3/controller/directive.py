"""Human-to-Controller optimization directives.

Directives are advisory, append-only observations.  They can influence the
next Controller plan, but they are deliberately not an execution interface:
the normal plan schema, campaign validation, evaluator, and trusted worker
remain the only paths that can change or run an experiment.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from ..memory.observation import ObservationRecord, ObservationStore, make_observation


DIRECTIVE_KIND = "human_directive"
DIRECTIVE_APPLY_AT = "next_controller_plan"
MAX_INSTRUCTION_CHARS = 4000
MAX_DIRECTIVE_ID_CHARS = 128
_DIRECTIVE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_instruction(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("directive instruction must be text")
    instruction = value.strip()
    if not instruction:
        raise ValueError("directive instruction must not be blank")
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        raise ValueError("directive instruction must be at most %d characters" % MAX_INSTRUCTION_CHARS)
    return instruction


def _normalize_directive_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("directive_id must be text")
    directive_id = value.strip()
    if not directive_id or len(directive_id) > MAX_DIRECTIVE_ID_CHARS or not _DIRECTIVE_ID_RE.fullmatch(directive_id):
        raise ValueError(
            "directive_id must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"
        )
    return directive_id


def _canonical_payload(directive_id: str, instruction: str, apply_at: str) -> bytes:
    return json.dumps(
        {
            "apply_at": apply_at,
            "directive_id": directive_id,
            "instruction": instruction,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True)
class HumanDirective:
    """A validated human objective that can be consumed at a plan boundary."""

    directive_id: str
    instruction: str
    apply_at: str
    created_at: str
    source_uri: str
    source_sha256: str

    def __post_init__(self) -> None:
        instruction = _normalize_instruction(self.instruction)
        directive_id = _normalize_directive_id(self.directive_id)
        if self.apply_at != DIRECTIVE_APPLY_AT:
            raise ValueError("directive apply_at must be %s" % DIRECTIVE_APPLY_AT)
        if not str(self.created_at).strip():
            raise ValueError("directive created_at must not be blank")
        expected_uri = "directive://%s" % directive_id
        if self.source_uri != expected_uri:
            raise ValueError("directive source_uri must be %s" % expected_uri)
        expected_sha256 = hashlib.sha256(_canonical_payload(directive_id, instruction, self.apply_at)).hexdigest()
        if self.source_sha256 != expected_sha256:
            raise ValueError("directive source_sha256 does not match its canonical payload")

    @classmethod
    def create(
        cls,
        instruction: str,
        *,
        directive_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> "HumanDirective":
        normalized_instruction = _normalize_instruction(instruction)
        if directive_id is None:
            digest = hashlib.sha256(normalized_instruction.encode("utf-8")).hexdigest()[:24]
            normalized_id = "dir-%s" % digest
        else:
            normalized_id = _normalize_directive_id(directive_id)
        apply_at = DIRECTIVE_APPLY_AT
        source_sha256 = hashlib.sha256(
            _canonical_payload(normalized_id, normalized_instruction, apply_at)
        ).hexdigest()
        return cls(
            directive_id=normalized_id,
            instruction=normalized_instruction,
            apply_at=apply_at,
            created_at=str(created_at or _now()),
            source_uri="directive://%s" % normalized_id,
            source_sha256=source_sha256,
        )

    def to_observation(self) -> ObservationRecord:
        return make_observation(
            "obs-%s" % self.directive_id,
            DIRECTIVE_KIND,
            self.source_uri,
            self.source_sha256,
            summary={
                "directive_id": self.directive_id,
                "instruction": self.instruction,
                "apply_at": self.apply_at,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))


def submit_directive(
    store: ObservationStore,
    instruction: str,
    *,
    directive_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Tuple[HumanDirective, bool]:
    """Append a directive observation and return ``(directive, newly_written)``."""

    directive = HumanDirective.create(
        instruction,
        directive_id=directive_id,
        created_at=created_at,
    )
    written = store.append(directive.to_observation())
    return directive, written


__all__ = [
    "DIRECTIVE_APPLY_AT",
    "DIRECTIVE_KIND",
    "HumanDirective",
    "MAX_DIRECTIVE_ID_CHARS",
    "MAX_INSTRUCTION_CHARS",
    "submit_directive",
]
