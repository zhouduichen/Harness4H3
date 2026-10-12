from __future__ import annotations

import random
from dataclasses import asdict, dataclass


@dataclass
class SearchPolicy:
    exploit_ratio: float = 0.7
    seed: int = 7
    rounds: int = 0
    exploit_rounds: int = 0
    explore_rounds: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.exploit_ratio) <= 1.0:
            raise ValueError("exploit_ratio must be in [0, 1]")
        self._random = random.Random(int(self.seed))

    def choose(self, *, has_best: bool) -> str:
        self.rounds += 1
        if not has_best or self._random.random() >= float(self.exploit_ratio):
            self.explore_rounds += 1
            return "explore"
        self.exploit_rounds += 1
        return "exploit"

    def to_dict(self) -> dict[str, int | float]:
        return {
            "exploit_ratio": float(self.exploit_ratio),
            "seed": int(self.seed),
            "rounds": int(self.rounds),
            "exploit_rounds": int(self.exploit_rounds),
            "explore_rounds": int(self.explore_rounds),
        }

