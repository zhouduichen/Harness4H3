"""Typed data contracts used by training methods and model adapters."""

from .schema import (
    Conditioning,
    MediaReferences,
    ModalInterval,
    ModalLatents,
    ModalPrediction,
    ModalSchedule,
    ModalTimesteps,
    PreparedBatch,
    TeacherSignals,
    TrainingSample,
)

__all__ = [
    "Conditioning",
    "MediaReferences",
    "ModalInterval",
    "ModalLatents",
    "ModalPrediction",
    "ModalSchedule",
    "ModalTimesteps",
    "PreparedBatch",
    "TeacherSignals",
    "TrainingSample",
]
