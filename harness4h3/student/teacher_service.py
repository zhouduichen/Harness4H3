"""Explicit online Teacher request/response boundary for Student training."""

from __future__ import annotations

import multiprocessing as mp
import queue
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
from torch import nn

from h3_training.algorithms.base import ModelRole
from h3_training.data.schema import Conditioning, ModalLatents, ModalPrediction, ModalTimesteps


class TeacherServiceError(RuntimeError):
    """Raised when an online Teacher rank cannot answer a request."""


class TeacherRoleProxy(nn.Module):
    """Parameter-free role model used by algorithms around a remote Teacher."""

    def forward(self, *args, **kwargs):  # pragma: no cover - service boundary never calls this
        raise TeacherServiceError("remote Teacher role cannot be called as a local module")


@dataclass(frozen=True)
class TeacherRequest:
    request_id: str
    noisy: ModalLatents
    timestep: ModalTimesteps
    conditioning: Conditioning


@dataclass(frozen=True)
class TeacherResponse:
    request_id: str
    prediction: ModalPrediction
    rank: int
    world_size: int


def _encode_modal(value: Optional[ModalLatents | ModalTimesteps | Conditioning]) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Conditioning):
        return {
            "text": value.text.detach().to(device="cpu"),
            "negative_text": value.negative_text.detach().to(device="cpu") if value.negative_text is not None else None,
        }
    return {
        name: tensor.detach().to(device="cpu") if tensor is not None else None
        for name, tensor in (("video", value.video), ("audio", value.audio))
    }


def _decode_latents(raw: dict[str, Any], device: torch.device, dtype: torch.dtype) -> ModalLatents:
    return ModalLatents(
        video=raw.get("video").to(device=device, dtype=dtype) if raw.get("video") is not None else None,
        audio=raw.get("audio").to(device=device, dtype=dtype) if raw.get("audio") is not None else None,
    )


def _decode_timesteps(raw: dict[str, Any], device: torch.device) -> ModalTimesteps:
    return ModalTimesteps(
        video=raw.get("video").to(device=device) if raw.get("video") is not None else None,
        audio=raw.get("audio").to(device=device) if raw.get("audio") is not None else None,
    )


def _rank_main(
    rank: int,
    world_size: int,
    device_name: str,
    checkpoint: str,
    comfyui_root: str,
    dtype: torch.dtype,
    request_queue: Any,
    response_queue: Any,
) -> None:
    """Load one real H3 Teacher rank and serve current-sample predictions."""

    try:
        from h3_training.adapters.real_h3 import RealMiniMaxH3Adapter

        device = torch.device(device_name)
        adapter = RealMiniMaxH3Adapter(Path(comfyui_root), device=device_name, dtype=dtype)
        role = adapter.load_role(Path(checkpoint), trainable=False)
        response_queue.put({"kind": "ready", "rank": rank, "world_size": world_size})
        while True:
            raw = request_queue.get()
            if raw is None:
                return
            request_id = str(raw.get("request_id", ""))
            try:
                noisy = _decode_latents(raw["noisy"], device, dtype)
                timesteps = _decode_timesteps(raw["timestep"], device)
                conditioning = Conditioning(
                    raw["conditioning"]["text"].to(device=device, dtype=dtype),
                    raw["conditioning"].get("negative_text").to(device=device, dtype=dtype)
                    if raw["conditioning"].get("negative_text") is not None
                    else None,
                )
                with torch.no_grad():
                    prediction = adapter.predict(role, noisy, timesteps, conditioning)
                response_queue.put(
                    {
                        "kind": "response",
                        "request_id": request_id,
                        "rank": rank,
                        "world_size": world_size,
                        "prediction": _encode_modal(prediction),
                    }
                )
            except Exception as exc:  # propagate the typed boundary failure to the parent
                response_queue.put(
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "rank": rank,
                        "message": "%s: %s" % (type(exc).__name__, exc),
                    }
                )
                return
    except Exception as exc:
        response_queue.put({"kind": "startup_error", "rank": rank, "message": "%s: %s" % (type(exc).__name__, exc)})


class TeacherServiceHandle:
    def __init__(
        self,
        predict_fn: Callable[[TeacherRequest], TeacherResponse],
        close_fn: Callable[[], None],
        *,
        world_size: int,
        devices: Sequence[str],
    ) -> None:
        self._predict_fn = predict_fn
        self._close_fn = close_fn
        self.world_size = int(world_size)
        self.devices = tuple(str(item) for item in devices)
        self.model = TeacherRoleProxy()
        self._closed = False
        self.ranks_used: set[int] = set()

    def predict(self, noisy: ModalLatents, timestep: ModalTimesteps, conditioning: Conditioning) -> ModalPrediction:
        if self._closed:
            raise TeacherServiceError("Teacher service is closed")
        response = self._predict_fn(TeacherRequest(uuid.uuid4().hex, noisy, timestep, conditioning))
        self.ranks_used.add(int(response.rank))
        return response.prediction

    def close(self) -> None:
        if not self._closed:
            self._close_fn()
            self._closed = True


class TeacherService:
    """Round-robin online Teacher service with one real H3 model per rank."""

    def __init__(
        self,
        checkpoint: Path,
        comfyui_root: Path,
        devices: Sequence[str],
        *,
        dtype: torch.dtype = torch.bfloat16,
        start_timeout_s: float = 180.0,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.comfyui_root = Path(comfyui_root).resolve()
        self.devices = tuple(str(item) for item in devices)
        self.dtype = dtype
        self.start_timeout_s = float(start_timeout_s)
        if not self.devices:
            raise ValueError("Teacher service requires at least one device")

    @classmethod
    def from_predictors(cls, predictors: Sequence[Callable[..., ModalPrediction]]) -> TeacherServiceHandle:
        """Build an in-process service for contract tests and CPU execution."""

        predictors = tuple(predictors)
        if not predictors:
            raise ValueError("at least one Teacher predictor is required")
        index = 0
        closed = False

        def predict(request: TeacherRequest) -> TeacherResponse:
            nonlocal index
            if closed:
                raise TeacherServiceError("Teacher service is closed")
            rank = index % len(predictors)
            index += 1
            predictor = predictors[rank]
            try:
                value = predictor(request.noisy, request.timestep, request.conditioning)
            except TypeError:
                value = predictor(request)
            if not isinstance(value, ModalPrediction):
                raise TeacherServiceError("fake Teacher predictor must return ModalPrediction")
            return TeacherResponse(request.request_id, value, rank, len(predictors))

        def close() -> None:
            nonlocal closed
            closed = True

        return TeacherServiceHandle(predict, close, world_size=len(predictors), devices=tuple("cpu:%d" % i for i in range(len(predictors))))

    def start(self) -> TeacherServiceHandle:
        if not self.checkpoint.is_file():
            raise TeacherServiceError("Teacher checkpoint does not exist: %s" % self.checkpoint)
        context = mp.get_context("spawn")
        request_queues = [context.Queue(maxsize=1) for _ in self.devices]
        response_queue = context.Queue()
        processes = []
        for rank, device in enumerate(self.devices):
            process = context.Process(
                target=_rank_main,
                args=(rank, len(self.devices), device, str(self.checkpoint), str(self.comfyui_root), self.dtype, request_queues[rank], response_queue),
                daemon=True,
            )
            process.start()
            processes.append(process)
        ready = set()
        deadline = __import__("time").monotonic() + self.start_timeout_s
        while len(ready) < len(processes):
            remaining = max(0.1, deadline - __import__("time").monotonic())
            if remaining <= 0:
                self._terminate(processes, request_queues)
                raise TeacherServiceError("Teacher service startup timed out")
            try:
                message = response_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if any(not process.is_alive() for process in processes):
                    self._terminate(processes, request_queues)
                    raise TeacherServiceError("Teacher service rank exited during startup")
                continue
            if message.get("kind") == "ready":
                ready.add(int(message["rank"]))
            elif message.get("kind") == "startup_error":
                self._terminate(processes, request_queues)
                raise TeacherServiceError(str(message.get("message", "Teacher rank startup failed")))

        next_rank = 0

        def predict(request: TeacherRequest) -> TeacherResponse:
            nonlocal next_rank
            rank = next_rank % len(request_queues)
            next_rank += 1
            request_queues[rank].put(
                {
                    "request_id": request.request_id,
                    "noisy": _encode_modal(request.noisy),
                    "timestep": _encode_modal(request.timestep),
                    "conditioning": _encode_modal(request.conditioning),
                }
            )
            deadline = __import__("time").monotonic() + self.start_timeout_s
            while True:
                remaining = deadline - __import__("time").monotonic()
                if remaining <= 0:
                    raise TeacherServiceError("Teacher prediction timed out")
                try:
                    message = response_queue.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    if not processes[rank].is_alive():
                        raise TeacherServiceError("Teacher rank %d exited during prediction" % rank)
                    continue
                if message.get("kind") == "response" and message.get("request_id") == request.request_id:
                    raw = message["prediction"]
                    return TeacherResponse(
                        request.request_id,
                        ModalPrediction(video=raw.get("video"), audio=raw.get("audio")),
                        int(message["rank"]),
                        int(message["world_size"]),
                    )
                if message.get("kind") == "error" and message.get("request_id") == request.request_id:
                    raise TeacherServiceError(str(message.get("message", "Teacher forward failed")))

        def close() -> None:
            self._terminate(processes, request_queues)

        return TeacherServiceHandle(predict, close, world_size=len(self.devices), devices=self.devices)

    @staticmethod
    def _terminate(processes: Sequence[Any], request_queues: Sequence[Any]) -> None:
        for request_queue in request_queues:
            try:
                request_queue.put_nowait(None)
            except (OSError, queue.Full):
                pass
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)


__all__ = [
    "TeacherRequest",
    "TeacherResponse",
    "TeacherRoleProxy",
    "TeacherService",
    "TeacherServiceError",
    "TeacherServiceHandle",
]
