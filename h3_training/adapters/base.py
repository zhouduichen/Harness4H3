"""Abstract adapter that isolates algorithms from model parameterization."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping

from torch import Generator

from h3_training.data.schema import (
    Conditioning,
    ModalInterval,
    ModalLatents,
    ModalPrediction,
    ModalSchedule,
    ModalTimesteps,
    PreparedBatch,
)


class DenoisingModelAdapter(ABC):
    @abstractmethod
    def prepare_batch(self, raw: Any, generator: Generator) -> PreparedBatch:
        raise NotImplementedError

    @abstractmethod
    def add_noise(self, clean: ModalLatents, noise: ModalLatents, timestep: ModalTimesteps) -> ModalLatents:
        raise NotImplementedError

    @abstractmethod
    def predict(self, role: Any, noisy: ModalLatents, timestep: ModalTimesteps, conditioning: Conditioning) -> ModalPrediction:
        raise NotImplementedError

    @abstractmethod
    def prediction_to_clean(self, noisy: ModalLatents, prediction: ModalPrediction, timestep: ModalTimesteps) -> ModalLatents:
        raise NotImplementedError

    @abstractmethod
    def scheduler_step(self, role: Any, latent: ModalLatents, prediction: ModalPrediction, interval: ModalInterval) -> ModalLatents:
        raise NotImplementedError

    @abstractmethod
    def schedule(self, num_model_evaluations: int) -> ModalSchedule:
        raise NotImplementedError

    @abstractmethod
    def save_role(self, role: Any, path: Path) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reload_role(self, path: Path) -> Any:
        raise NotImplementedError

    def resolve_trainable_parameters(self, role: Any, policy: str) -> Iterable[str]:
        raise NotImplementedError("adapter does not implement trainable parameter policies")
