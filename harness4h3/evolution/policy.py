from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SearchPolicy:
    exploit_ratio: float = 0.7
    seed: int = 7
    rounds: int = 0
    exploit_rounds: int = 0
    explore_rounds: int = 0
    duplicate_rounds: int = 0
    costly_failure_rounds: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.exploit_ratio) <= 1.0:
            raise ValueError("exploit_ratio must be in [0, 1]")

    def choose(self, *, has_best: bool) -> str:
        self.rounds += 1
        # Derive the draw from the persisted round number instead of keeping
        # an in-memory RNG state.  A resumed process therefore makes the same
        # next choice as an uninterrupted process.
        draw = random.Random(int(self.seed) + self.rounds).random()
        if not has_best or draw >= float(self.exploit_ratio):
            self.explore_rounds += 1
            return "explore"
        self.exploit_rounds += 1
        return "exploit"

    def update(self, *, feasible: bool, improved: bool, failure_type: str | None = None) -> None:
        """Adapt the next allocation from measured round feedback."""

        if feasible and improved:
            self.exploit_ratio = min(0.9, float(self.exploit_ratio) + 0.05)
        elif not feasible:
            self.exploit_ratio = max(0.3, float(self.exploit_ratio) - 0.10)
        if failure_type == "duplicate_algorithm":
            self.duplicate_rounds += 1
            self.exploit_ratio = max(0.2, float(self.exploit_ratio) - 0.10)
        if failure_type in {
            "evaluator_failed",
            "child_integrity_failed",
            "compression_failed",
            "sandbox_timeout",
            "no_substantial_efficiency_improvement",
            "model_size_improvement_below_threshold",
            "pareto_dominated",
        }:
            self.costly_failure_rounds += 1

    def to_dict(self) -> dict[str, int | float]:
        return {
            "exploit_ratio": float(self.exploit_ratio),
            "seed": int(self.seed),
            "rounds": int(self.rounds),
            "exploit_rounds": int(self.exploit_rounds),
            "explore_rounds": int(self.explore_rounds),
            "duplicate_rounds": int(self.duplicate_rounds),
            "costly_failure_rounds": int(self.costly_failure_rounds),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "SearchPolicy":
        return cls(
            exploit_ratio=float(raw.get("exploit_ratio", 0.7)),
            seed=int(raw.get("seed", 7)),
            rounds=int(raw.get("rounds", 0)),
            exploit_rounds=int(raw.get("exploit_rounds", 0)),
            explore_rounds=int(raw.get("explore_rounds", 0)),
            duplicate_rounds=int(raw.get("duplicate_rounds", 0)),
            costly_failure_rounds=int(raw.get("costly_failure_rounds", 0)),
        )
