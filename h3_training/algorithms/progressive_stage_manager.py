"""Explicit teacher promotion for binary progressive distillation experiments."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .progressive_distillation import DistillationStage, plan_binary_stages


@dataclass(frozen=True)
class StagePromotion:
    stage_index: int
    teacher_model_id: str
    teacher_checkpoint: Optional[str]
    teacher_nfe: int
    student_model_id: str
    student_checkpoint: str
    student_nfe: int


@dataclass
class ProgressiveStageManager:
    """Own the complete ``source_nfe -> target_nfe`` promotion state.

    The manager is deliberately independent of a Controller. A stage can only
    advance after an evaluator has accepted its child, at which point that
    child becomes the sole teacher for the next binary stage.
    """

    source_nfe: int
    target_nfe: int
    teacher_model_id: str
    teacher_checkpoint: Optional[str] = None
    promotions: List[StagePromotion] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._stages = plan_binary_stages(self.source_nfe, self.target_nfe)
        if not self.teacher_model_id:
            raise ValueError("teacher_model_id must not be empty")
        if len(self.promotions) > len(self._stages):
            raise ValueError("too many stage promotions")
        for expected_index, promotion in enumerate(self.promotions):
            expected = self._stages[expected_index]
            if promotion.stage_index != expected_index:
                raise ValueError("stage promotions must be ordered")
            if promotion.teacher_nfe != expected.teacher_nfe or promotion.student_nfe != expected.student_nfe:
                raise ValueError("stage promotion NFE does not match the binary plan")

    @property
    def stages(self):
        return self._stages

    @property
    def stage_index(self) -> int:
        return len(self.promotions)

    @property
    def complete(self) -> bool:
        return self.stage_index == len(self._stages)

    def next_stage(self) -> Optional[DistillationStage]:
        return None if self.complete else self._stages[self.stage_index]

    def next_stage_request(self) -> Optional[Mapping[str, Any]]:
        stage = self.next_stage()
        if stage is None:
            return None
        return {
            "stage_index": self.stage_index,
            "teacher_model_id": self.teacher_model_id,
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_nfe": stage.teacher_nfe,
            "student_nfe": stage.student_nfe,
        }

    def promote(
        self,
        student_model_id: str,
        student_checkpoint: Path,
        student_nfe: int,
        *,
        accepted: bool,
    ) -> StagePromotion:
        stage = self.next_stage()
        if stage is None:
            raise ValueError("progressive stage plan is already complete")
        if not accepted:
            raise ValueError("teacher promotion requires evaluator acceptance")
        if not student_model_id or student_model_id == self.teacher_model_id:
            raise ValueError("promoted student must have a distinct model id")
        if int(student_nfe) != stage.student_nfe:
            raise ValueError("promoted student NFE does not match the active stage")
        checkpoint = str(Path(student_checkpoint).resolve())
        if not checkpoint:
            raise ValueError("promoted student checkpoint must not be empty")
        promotion = StagePromotion(
            stage_index=self.stage_index,
            teacher_model_id=self.teacher_model_id,
            teacher_checkpoint=self.teacher_checkpoint,
            teacher_nfe=stage.teacher_nfe,
            student_model_id=student_model_id,
            student_checkpoint=checkpoint,
            student_nfe=stage.student_nfe,
        )
        self.promotions.append(promotion)
        self.teacher_model_id = student_model_id
        self.teacher_checkpoint = checkpoint
        return promotion

    def state_dict(self) -> Dict[str, Any]:
        first = self.promotions[0] if self.promotions else None
        return {
            "source_nfe": self.source_nfe,
            "target_nfe": self.target_nfe,
            "teacher_model_id": self.teacher_model_id,
            "teacher_checkpoint": self.teacher_checkpoint,
            "initial_teacher_model_id": first.teacher_model_id if first else self.teacher_model_id,
            "initial_teacher_checkpoint": first.teacher_checkpoint if first else self.teacher_checkpoint,
            "promotions": [asdict(item) for item in self.promotions],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ProgressiveStageManager":
        promotions = [StagePromotion(**dict(item)) for item in state.get("promotions", [])]
        manager = cls(
            source_nfe=int(state["source_nfe"]),
            target_nfe=int(state["target_nfe"]),
            teacher_model_id=str(state.get("initial_teacher_model_id", state["teacher_model_id"])),
            teacher_checkpoint=state.get("initial_teacher_checkpoint"),
        )
        for promotion in promotions:
            stage = manager.next_stage()
            if stage is None:
                raise ValueError("too many stage promotions")
            if promotion.teacher_model_id != manager.teacher_model_id:
                raise ValueError("promotion teacher does not match manager state")
            manager.promote(
                promotion.student_model_id,
                Path(promotion.student_checkpoint),
                promotion.student_nfe,
                accepted=True,
            )
        if manager.teacher_model_id != str(state["teacher_model_id"]):
            raise ValueError("manager teacher does not match serialized state")
        if manager.teacher_checkpoint != state.get("teacher_checkpoint"):
            raise ValueError("manager checkpoint does not match serialized state")
        return manager


__all__ = ["ProgressiveStageManager", "StagePromotion"]
