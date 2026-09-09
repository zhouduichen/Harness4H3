from __future__ import annotations

from dataclasses import dataclass

from ..archive.model_candidate import ModelCandidate
from .state import ModelState


@dataclass(frozen=True)
class FakeH3Model:
    """Deterministic H3-shaped baseline used only by offline CI and demos."""

    state: ModelState

    @classmethod
    def baseline(cls) -> "FakeH3Model":
        return cls(ModelState.fake_baseline())

    def candidate(self) -> ModelCandidate:
        return ModelCandidate(
            id=self.state.model_id,
            parent_id=self.state.parent_model_id,
            generation=0,
            checkpoint_path=self.state.checkpoint_path,
            state=self.state,
            created_by_experiment_id=None,
            status="baseline",
        )
