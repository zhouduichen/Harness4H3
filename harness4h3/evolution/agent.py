from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .candidate import AlgorithmCandidate


Provider = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _seed_source(block_id: str, algorithm_id: str) -> str:
    return (
        "def propose(profile, history):\n"
        "    # Seed algorithm %s selects the least output-sensitive block reported by the profiler.\n"
        "    selected = %r\n"
        "    return {'operator': 'prune_blocks', 'args': {'block_ids': [selected]}}\n" % (algorithm_id, block_id)
    )


@dataclass(frozen=True)
class AlgorithmAgent:
    provider: Optional[Provider] = None

    def generate(
        self,
        algorithm_id: str,
        *,
        parent_id: Optional[str],
        profile: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        skills: Sequence[Mapping[str, Any]] = (),
    ) -> AlgorithmCandidate:
        if self.provider is None:
            block_id = self._least_sensitive_block(profile)
            source = _seed_source(block_id, algorithm_id)
            return AlgorithmCandidate.from_source(
                algorithm_id,
                source,
                parent_id=parent_id,
                hypothesis="remove the lowest measured leave-one-block-out sensitivity block",
                metadata={"generator": "deterministic_seed", "operator": "prune_blocks"},
            )
        payload = {
            "profile": dict(profile),
            "history": [dict(item) for item in history],
            "skills": [dict(item) for item in skills],
            "parent_id": parent_id,
        }
        raw = self.provider(payload)
        if not isinstance(raw, Mapping):
            raise ValueError("algorithm_generation_invalid: provider response must be an object")
        unknown = sorted(set(raw) - {"source", "hypothesis", "parent_id"})
        if unknown:
            raise ValueError("algorithm_generation_invalid: unsupported field(s): %s" % ", ".join(unknown))
        response_parent = raw.get("parent_id", parent_id)
        if response_parent != parent_id:
            raise ValueError("algorithm_generation_invalid: provider cannot change parent_id")
        source = raw.get("source")
        if not isinstance(source, str):
            raise ValueError("algorithm_generation_invalid: source must be text")
        return AlgorithmCandidate.from_source(
            algorithm_id,
            source,
            parent_id=parent_id,
            hypothesis=str(raw.get("hypothesis", "")),
            metadata={"generator": "structured_provider"},
        )

    @staticmethod
    def _least_sensitive_block(profile: Mapping[str, Any]) -> str:
        records = profile.get("block_sensitivity", [])
        candidates = [item for item in records if isinstance(item, Mapping) and isinstance(item.get("block_id"), str) and isinstance(item.get("importance_score"), (int, float))]
        if candidates:
            return str(min(candidates, key=lambda item: float(item["importance_score"]))["block_id"])
        layers = profile.get("layers", [])
        for item in layers:
            if isinstance(item, Mapping) and isinstance(item.get("block_id"), str):
                return str(item["block_id"])
        return "block.0"
