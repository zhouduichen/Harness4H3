"""Collective, online MiniMax-H3 Teacher service for Student training."""

from __future__ import annotations

import multiprocessing as mp
import queue
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
from torch import nn

from h3_training.data.schema import Conditioning, ModalLatents, ModalPrediction, ModalTimesteps


class TeacherServiceError(RuntimeError):
    """Raised when the online Teacher cannot answer a request."""


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
    participating_ranks: tuple[int, ...] = ()
    forward_index: int = 0
    peak_memory_gb: float = 0.0


def _encode_modal(value: Optional[ModalLatents | ModalTimesteps | Conditioning]) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Conditioning):
        return {
            "text": value.text.detach().to(device="cpu"),
            "negative_text": value.negative_text.detach().to(device="cpu")
            if value.negative_text is not None
            else None,
        }
    return {
        name: tensor.detach().to(device="cpu") if tensor is not None else None
        for name, tensor in (("video", value.video), ("audio", value.audio))
    }


def _dtype_from_name(name: str) -> torch.dtype:
    value = getattr(torch, str(name).split(".")[-1], None)
    if not isinstance(value, torch.dtype):
        raise TeacherServiceError("unsupported request tensor dtype: %s" % name)
    return value


def _tensor_specs(raw: Mapping[str, Any]) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for group, fields in (
        ("noisy", ("video", "audio")),
        ("timestep", ("video", "audio")),
        ("conditioning", ("text", "negative_text")),
    ):
        source = raw.get(group) or {}
        for field in fields:
            tensor = source.get(field)
            key = "%s.%s" % (group, field)
            if tensor is None:
                specs[key] = None
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TeacherServiceError("request field %s must be a tensor" % key)
            specs[key] = {"shape": tuple(int(item) for item in tensor.shape), "dtype": str(tensor.dtype)}
    return specs


def _broadcast_tensor(
    value: Optional[torch.Tensor],
    spec: Optional[Mapping[str, Any]],
    *,
    rank: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if spec is None:
        return None
    dtype = _dtype_from_name(str(spec["dtype"]))
    shape = tuple(int(item) for item in spec["shape"])
    if rank == 0:
        if value is None:
            raise TeacherServiceError("rank 0 request tensor is missing")
        tensor = value.to(device=device, dtype=dtype, non_blocking=True).contiguous()
    else:
        tensor = torch.empty(shape, device=device, dtype=dtype)
    dist.broadcast(tensor, src=0)
    return tensor


def _broadcast_request(
    rank: int,
    request_queue: Any,
    device: torch.device,
) -> Optional[tuple[str, dict[str, Optional[torch.Tensor]]]]:
    """Read on rank 0, then broadcast metadata and every tensor over NCCL."""

    raw: Optional[Mapping[str, Any]] = request_queue.get() if rank == 0 else None
    if rank == 0:
        if raw is None:
            envelope: dict[str, Any] = {"kind": "stop"}
        else:
            envelope = {
                "kind": "request",
                "request_id": str(raw.get("request_id", "")),
                "specs": _tensor_specs(raw),
            }
    else:
        envelope = None
    objects = [envelope]
    dist.broadcast_object_list(objects, src=0, device=device)
    envelope = objects[0]
    if not isinstance(envelope, Mapping):
        raise TeacherServiceError("invalid distributed Teacher request envelope")
    if envelope.get("kind") == "stop":
        return None
    if envelope.get("kind") != "request":
        raise TeacherServiceError("invalid distributed Teacher request kind")
    specs = envelope.get("specs")
    if not isinstance(specs, Mapping):
        raise TeacherServiceError("distributed Teacher request has no tensor specs")
    raw = raw or {}
    decoded: dict[str, Optional[torch.Tensor]] = {}
    for group, fields in (
        ("noisy", ("video", "audio")),
        ("timestep", ("video", "audio")),
        ("conditioning", ("text", "negative_text")),
    ):
        source = raw.get(group) if isinstance(raw, Mapping) else {}
        source = source if isinstance(source, Mapping) else {}
        for field in fields:
            key = "%s.%s" % (group, field)
            decoded[key] = _broadcast_tensor(
                source.get(field) if rank == 0 else None,
                specs.get(key),
                rank=rank,
                device=device,
            )
    return str(envelope["request_id"]), decoded


def _construct_teacher_model(
    api: Mapping[str, Any],
    checkpoint: Path,
    device: torch.device,
    rank: int,
    *,
    load_checkpoint: bool,
) -> torch.nn.Module:
    """Use the validated H3 constructor with explicit checkpoint loading."""

    try:
        from tools.h3_teacher_target_worker import _construct_teacher_model as construct_teacher_model
    except ModuleNotFoundError:  # direct invocation from tools/ or an installed checkout
        from h3_teacher_target_worker import _construct_teacher_model as construct_teacher_model
    return construct_teacher_model(api, checkpoint, rank, device, load_checkpoint=load_checkpoint)


def _fsdp_rank_main(
    rank: int,
    world_size: int,
    device_name: str,
    checkpoint: str,
    comfyui_root: str,
    dtype: torch.dtype,
    init_method: str,
    distributed_timeout_s: float,
    request_queue: Any,
    response_queue: Any,
    load_checkpoint_all_ranks: bool,
) -> None:
    """Run one rank of one collective FSDP-sharded online H3 Teacher."""

    initialized = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("collective H3 Teacher requires CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=float(distributed_timeout_s)),
        )
        initialized = True
        try:
            from tools.h3_real_train_worker import _import_comfy, _wrap_fsdp
        except ModuleNotFoundError:  # direct invocation from tools/ or an installed checkout
            from h3_real_train_worker import _import_comfy, _wrap_fsdp

        api = _import_comfy(Path(comfyui_root).resolve())
        api["comfy_model_management"].in_training = True
        model = _construct_teacher_model(
            api,
            Path(checkpoint).resolve(),
            device,
            rank,
            load_checkpoint=bool(load_checkpoint_all_ranks or rank == 0),
        )
        fsdp_model = _wrap_fsdp(
            model,
            api,
            device,
            train_heads=False,
            sync_module_states=True,
        )
        fsdp_model.eval()
        dist.barrier()
        response_queue.put({"kind": "ready", "rank": rank, "world_size": world_size, "sharded": True})
        torch.cuda.reset_peak_memory_stats(device)
        forward_index = 0
        while True:
            request = _broadcast_request(rank, request_queue, device)
            if request is None:
                break
            request_id, tensors = request
            noisy_video = tensors["noisy.video"]
            noisy_audio = tensors["noisy.audio"]
            timestep_video = tensors["timestep.video"]
            timestep_audio = tensors["timestep.audio"]
            conditioning_text = tensors["conditioning.text"]
            conditioning_negative = tensors["conditioning.negative_text"]
            if any(value is None for value in (noisy_video, noisy_audio, timestep_video, conditioning_text)):
                raise TeacherServiceError("online H3 Teacher requires video, audio, timestep, and text conditioning")
            # Native H3 derives its audio branch from the shifted video
            # timestep.  Keep the audio timestep in the transport contract so
            # the service still receives the complete multimodal state.
            del timestep_audio, conditioning_negative
            layout = api["PackedLayout"](
                int(conditioning_text.shape[1]),
                int(noisy_video.shape[2]),
                int(noisy_video.shape[3]),
                int(noisy_video.shape[4]),
                int(noisy_audio.shape[3]),
            )
            local_error = torch.zeros(1, device=device, dtype=torch.int32)
            prediction: Optional[ModalPrediction] = None
            error_message = ""
            try:
                with torch.inference_mode():
                    raw_video, raw_audio = fsdp_model(
                        [noisy_video, noisy_audio],
                        timestep_video * 1000.0,
                        conditioning_text,
                        transformer_options={},
                        minimax_payload={"layout": layout, "audio_scale": 1.0},
                    )
                # ComfyUI H3 emits the opposite velocity convention from the
                # Student algorithms: Student consumes clean-minus-noise.
                prediction = ModalPrediction(video=-raw_video, audio=-raw_audio)
            except Exception as exc:  # all ranks report the collective failure
                local_error.fill_(1)
                error_message = "%s: %s" % (type(exc).__name__, exc)
            dist.all_reduce(local_error, op=dist.ReduceOp.MAX)
            peak = torch.tensor([float(torch.cuda.max_memory_allocated(device))], device=device, dtype=torch.float64)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if int(local_error.item()) != 0:
                if rank == 0:
                    response_queue.put(
                        {
                            "kind": "error",
                            "request_id": request_id,
                            "message": error_message or "a Teacher rank failed during collective forward",
                        }
                    )
                dist.barrier()
                continue
            if prediction is None:
                raise TeacherServiceError("rank produced no online H3 prediction")
            forward_index += 1
            if rank == 0:
                response_queue.put(
                    {
                        "kind": "response",
                        "request_id": request_id,
                        "rank": 0,
                        "world_size": world_size,
                        "participating_ranks": list(range(world_size)),
                        "forward_index": forward_index,
                        "peak_memory_gb": float(peak.item() / (1024**3)),
                        "prediction": _encode_modal(prediction),
                    }
                )
            # Keep all ranks at the same request boundary before rank 0 can
            # dequeue the next request.
            dist.barrier()
            torch.cuda.reset_peak_memory_stats(device)
    except Exception as exc:
        try:
            response_queue.put(
                {
                    "kind": "startup_error" if not initialized else "fatal_error",
                    "rank": rank,
                    "message": "%s: %s" % (type(exc).__name__, exc),
                }
            )
        except (BrokenPipeError, OSError):
            pass
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


class TeacherServiceHandle:
    def __init__(
        self,
        predict_fn: Callable[[TeacherRequest], TeacherResponse],
        close_fn: Callable[[], None],
        *,
        world_size: int,
        devices: Sequence[str],
        sharded: bool = False,
    ) -> None:
        self._predict_fn = predict_fn
        self._close_fn = close_fn
        self.world_size = int(world_size)
        self.devices = tuple(str(item) for item in devices)
        self.sharded = bool(sharded)
        self.model = TeacherRoleProxy()
        self._closed = False
        self.ranks_used: set[int] = set()
        self.rank_forward_counts: dict[int, int] = {}
        self.forward_count = 0
        self.teacher_peak_memory_gb = 0.0

    def predict(self, noisy: ModalLatents, timestep: ModalTimesteps, conditioning: Conditioning) -> ModalPrediction:
        if self._closed:
            raise TeacherServiceError("Teacher service is closed")
        response = self._predict_fn(TeacherRequest(uuid.uuid4().hex, noisy, timestep, conditioning))
        participants = response.participating_ranks or (int(response.rank),)
        self.ranks_used.update(int(rank) for rank in participants)
        self.forward_count += 1
        for rank in participants:
            self.rank_forward_counts[int(rank)] = self.rank_forward_counts.get(int(rank), 0) + 1
        self.teacher_peak_memory_gb = max(self.teacher_peak_memory_gb, float(response.peak_memory_gb))
        return response.prediction

    def close(self) -> None:
        if not self._closed:
            self._close_fn()
            self._closed = True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TeacherService:
    """One three-rank FSDP-sharded online Teacher with rank-0 ingress/egress."""

    REQUIRED_WORLD_SIZE = 3

    def __init__(
        self,
        checkpoint: Path,
        comfyui_root: Path,
        devices: Sequence[str],
        *,
        dtype: torch.dtype = torch.bfloat16,
        start_timeout_s: float = 180.0,
        distributed_timeout_s: float = 180.0,
        load_checkpoint_all_ranks: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.comfyui_root = Path(comfyui_root).resolve()
        self.devices = tuple(str(item) for item in devices)
        self.dtype = dtype
        self.start_timeout_s = float(start_timeout_s)
        self.distributed_timeout_s = float(distributed_timeout_s)
        self.load_checkpoint_all_ranks = bool(load_checkpoint_all_ranks)
        if len(self.devices) != self.REQUIRED_WORLD_SIZE:
            raise ValueError("online H3 Teacher requires exactly three devices/ranks")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("online H3 Teacher devices must be distinct")
        if any(not item.startswith("cuda:") for item in self.devices):
            raise ValueError("online H3 Teacher devices must be explicit cuda:N values")
        if self.start_timeout_s <= 0 or self.distributed_timeout_s <= 0:
            raise ValueError("Teacher service timeouts must be positive")

    @classmethod
    def from_predictors(cls, predictors: Sequence[Callable[..., ModalPrediction]]) -> TeacherServiceHandle:
        """Build an in-process non-FSDP service for CPU contract tests only."""

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
            return TeacherResponse(request.request_id, value, rank, len(predictors), (rank,))

        def close() -> None:
            nonlocal closed
            closed = True

        return TeacherServiceHandle(
            predict,
            close,
            world_size=len(predictors),
            devices=tuple("cpu:%d" % i for i in range(len(predictors))),
            sharded=False,
        )

    def start(self) -> TeacherServiceHandle:
        if not self.checkpoint.is_file():
            raise TeacherServiceError("Teacher checkpoint does not exist: %s" % self.checkpoint)
        if not self.comfyui_root.is_dir():
            raise TeacherServiceError("ComfyUI root does not exist: %s" % self.comfyui_root)
        context = mp.get_context("spawn")
        request_queue = context.Queue(maxsize=1)
        response_queue = context.Queue()
        processes = []
        init_method = "tcp://127.0.0.1:%d" % _free_port()
        for rank, device in enumerate(self.devices):
            process = context.Process(
                target=_fsdp_rank_main,
                args=(
                    rank,
                    self.REQUIRED_WORLD_SIZE,
                    device,
                    str(self.checkpoint),
                    str(self.comfyui_root),
                    self.dtype,
                    init_method,
                    self.distributed_timeout_s,
                    request_queue,
                    response_queue,
                    self.load_checkpoint_all_ranks,
                ),
                daemon=True,
            )
            process.start()
            processes.append(process)
        ready: set[int] = set()
        deadline = time.monotonic() + self.start_timeout_s
        try:
            while len(ready) < self.REQUIRED_WORLD_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TeacherServiceError("Teacher service startup timed out")
                try:
                    message = response_queue.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    if any(not process.is_alive() for process in processes):
                        raise TeacherServiceError("Teacher service rank exited during startup")
                    continue
                kind = message.get("kind")
                if kind == "ready":
                    ready.add(int(message["rank"]))
                elif kind in {"startup_error", "fatal_error"}:
                    raise TeacherServiceError(str(message.get("message", "Teacher rank startup failed")))
        except Exception:
            self._terminate(processes, request_queue)
            raise

        def predict(request: TeacherRequest) -> TeacherResponse:
            request_queue.put(
                {
                    "request_id": request.request_id,
                    "noisy": _encode_modal(request.noisy),
                    "timestep": _encode_modal(request.timestep),
                    "conditioning": _encode_modal(request.conditioning),
                }
            )
            deadline = time.monotonic() + self.start_timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TeacherServiceError("Teacher prediction timed out")
                try:
                    message = response_queue.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    if any(not process.is_alive() for process in processes):
                        raise TeacherServiceError("a collective Teacher rank exited during prediction")
                    continue
                kind = message.get("kind")
                if kind == "response" and message.get("request_id") == request.request_id:
                    raw = message["prediction"]
                    return TeacherResponse(
                        request.request_id,
                        ModalPrediction(video=raw.get("video"), audio=raw.get("audio")),
                        int(message.get("rank", 0)),
                        int(message.get("world_size", self.REQUIRED_WORLD_SIZE)),
                        tuple(int(rank) for rank in message.get("participating_ranks", ())),
                        int(message.get("forward_index", 0)),
                        float(message.get("peak_memory_gb", 0.0)),
                    )
                if kind == "error" and message.get("request_id") == request.request_id:
                    raise TeacherServiceError(str(message.get("message", "Teacher forward failed")))
                if kind in {"fatal_error", "startup_error"}:
                    raise TeacherServiceError(str(message.get("message", "collective Teacher rank failed")))

        def close() -> None:
            self._terminate(processes, request_queue)

        return TeacherServiceHandle(
            predict,
            close,
            world_size=self.REQUIRED_WORLD_SIZE,
            devices=self.devices,
            sharded=True,
        )

    @staticmethod
    def _terminate(processes: Sequence[Any], request_queue: Any) -> None:
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
    "_construct_teacher_model",
    "TeacherRequest",
    "TeacherResponse",
    "TeacherRoleProxy",
    "TeacherService",
    "TeacherServiceError",
    "TeacherServiceHandle",
]
