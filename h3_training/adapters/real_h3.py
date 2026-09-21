"""Source-backed MiniMax-H3 adapter for the ComfyUI model implementation.

The adapter intentionally has no TinyH3 or CPU fallback.  It is a thin bridge
between the generic training algorithms and the real ComfyUI MiniMax-H3 model;
the distributed L40 worker remains responsible for production-scale FSDP runs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from h3_training.adapters.base import DenoisingModelAdapter
from h3_training.algorithms.base import ModelRole
from h3_training.data.schema import (
    Conditioning,
    ModalInterval,
    ModalLatents,
    ModalPrediction,
    ModalSchedule,
    ModalTimesteps,
    PreparedBatch,
)
from h3_training.engine.state import TrainingFailure


TRAINABLE_TENSOR_NAMES = frozenset(
    {
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)


class RealMiniMaxH3Adapter(DenoisingModelAdapter):
    """Adapter for the real ComfyUI MiniMax-H3 FL2VA transformer.

    ``comfyui_root`` is explicit so importing this module never silently binds
    to an unrelated ComfyUI checkout.  The model config is read from the
    checkpoint metadata and the checkpoint is loaded with strict key checks;
    missing or unexpected tensors are training failures, not warnings.
    """

    def __init__(
        self,
        comfyui_root: Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
    ) -> None:
        self.comfyui_root = Path(comfyui_root).resolve()
        self.device = torch.device(device)
        self.dtype = dtype
        self.video_shift = float(video_shift)
        self.audio_shift = float(audio_shift)
        if self.video_shift <= 0 or self.audio_shift <= 0:
            raise ValueError("H3 modality shifts must be positive")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingFailure("device_unavailable", "CUDA is not available for real MiniMax-H3 training")
        self._api: Optional[Mapping[str, Any]] = None

    def _load_api(self) -> Mapping[str, Any]:
        if self._api is not None:
            return self._api
        if not self.comfyui_root.is_dir():
            raise TrainingFailure("h3_adapter_unavailable", "ComfyUI root does not exist: %s" % self.comfyui_root)
        root = str(self.comfyui_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import comfy.ops as ops
            from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout, time_shift_sigma
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise TrainingFailure("h3_adapter_unavailable", "ComfyUI MiniMax-H3 symbols are unavailable: %s" % exc) from exc
        self._api = {
            "ops": ops,
            "MiniMaxH3Model": MiniMaxH3Model,
            "PackedLayout": PackedLayout,
            "time_shift_sigma": time_shift_sigma,
        }
        return self._api

    @staticmethod
    def _metadata(path: Path) -> Dict[str, str]:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
        except (OSError, ValueError, RuntimeError) as exc:
            raise TrainingFailure("checkpoint_corrupt", "unable to read H3 checkpoint metadata: %s" % exc) from exc
        if not metadata.get("config"):
            raise TrainingFailure("checkpoint_corrupt", "H3 checkpoint metadata has no config")
        return metadata

    def _construct_model(self, path: Path) -> tuple[torch.nn.Module, Dict[str, str], Mapping[str, Any]]:
        metadata = self._metadata(path)
        try:
            raw_config = json.loads(metadata["config"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrainingFailure("checkpoint_corrupt", "H3 checkpoint config is not valid JSON") from exc
        if not isinstance(raw_config, Mapping) or not isinstance(raw_config.get("transformer"), Mapping):
            raise TrainingFailure("checkpoint_corrupt", "H3 checkpoint config has no transformer section")
        api = self._load_api()
        try:
            # Construct on meta and load safetensors directly on the teacher
            # GPU. Constructing on CPU and then calling model.to(device) causes
            # a full-model copy peak; the real H3 checkpoint leaves only a few
            # hundred MiB free on a 48GB L40, so that transient copy OOMs.
            with torch.device("meta"):
                model = api["MiniMaxH3Model"](
                    **dict(raw_config["transformer"]),
                    dtype=self.dtype,
                    device=torch.device("meta"),
                    operations=api["ops"].disable_weight_init,
                )
            state = load_file(str(path), device=str(self.device))
            expected = set(model.state_dict().keys())
            slots = {}
            for module_name, module in model.named_modules():
                prefix = (module_name + ".") if module_name else ""
                slots.update({prefix + name: (module, "parameter", name) for name in module._parameters})
                slots.update({prefix + name: (module, "buffer", name) for name in module._buffers})
            missing = sorted(expected - set(state))
            unexpected = sorted(set(state) - expected)
            for name, value in state.items():
                slot = slots.get(name)
                if slot is None:
                    continue
                module, kind, local_name = slot
                if kind == "parameter":
                    module._parameters[local_name] = torch.nn.Parameter(value, requires_grad=False)
                else:
                    module._buffers[local_name] = value
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise TrainingFailure("checkpoint_incompatible", "unable to load MiniMax-H3 weights: %s" % exc) from exc
        finally:
            if "state" in locals():
                del state
        if missing or unexpected:
            raise TrainingFailure(
                "checkpoint_incompatible",
                "H3 model key mismatch; missing=%s unexpected=%s" % (list(missing)[:3], list(unexpected)[:3]),
            )
        return model, metadata, raw_config

    def load_role(self, path: Path, trainable: bool = False) -> ModelRole:
        path = Path(path).resolve()
        if not path.is_file():
            raise TrainingFailure("checkpoint_corrupt", "H3 checkpoint does not exist: %s" % path)
        model, metadata, config = self._construct_model(path)
        model.train(bool(trainable))
        return ModelRole(
            "student",
            model,
            self,
            bool(trainable),
            {
                "checkpoint_path": str(path),
                "checkpoint_metadata": metadata,
                "transformer_config": dict(config["transformer"]),
            },
        )

    @staticmethod
    def _unpatch_video(
        rows: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        latent_frames: int = 5,
    ) -> torch.Tensor:
        if rows.ndim != 2 or rows.shape[1] != 96:
            raise TrainingFailure("invalid_training_config", "H3 video cache must have shape [rows, 96]")
        if isinstance(latent_frames, bool) or not isinstance(latent_frames, int) or latent_frames <= 0:
            raise TrainingFailure("invalid_training_config", "H3 latent_frames must be a positive integer")
        if rows.shape[0] % latent_frames:
            raise TrainingFailure("invalid_training_config", "H3 video rows are not divisible by latent_frames")
        rows_per_frame = rows.shape[0] // latent_frames
        patch_side = math.isqrt(rows_per_frame)
        if patch_side * patch_side != rows_per_frame:
            raise TrainingFailure(
                "invalid_training_config",
                "H3 video rows per frame must form a square patch grid",
            )
        frames = latent_frames
        return (
            rows.reshape(frames, patch_side, patch_side, 24, 1, 2, 2)
            .permute(3, 0, 4, 1, 5, 2, 6)
            .reshape(1, 24, frames, patch_side * 2, patch_side * 2)
            .to(device=device, dtype=dtype)
        )

    @staticmethod
    def _unpack_audio(rows: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if rows.ndim != 2 or rows.shape[1] != 32 or rows.shape[0] % 2:
            raise TrainingFailure("invalid_training_config", "H3 audio cache must have shape [frames*2, 32]")
        frames = rows.shape[0] // 2
        return rows.reshape(2, frames, 32).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    @staticmethod
    def _generator_device(generator: torch.Generator) -> torch.device:
        value = getattr(generator, "device", torch.device("cpu"))
        return value if isinstance(value, torch.device) else torch.device(value)

    def _shift_audio_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        try:
            shifted = self._load_api()["time_shift_sigma"](
                sigma, self.video_shift, self.audio_shift
            )
        except (TypeError, RuntimeError, ValueError) as exc:
            raise TrainingFailure("h3_adapter_unavailable", "ComfyUI H3 sigma transform failed: %s" % exc) from exc
        return shifted.to(device=sigma.device, dtype=sigma.dtype)

    def prepare_batch(self, raw: Any, generator: torch.Generator) -> PreparedBatch:
        if isinstance(raw, PreparedBatch):
            return raw
        if not isinstance(raw, Mapping):
            raise TrainingFailure("invalid_training_config", "real H3 batches must be cache mappings")
        required = ("video", "audio", "prompt")
        missing = [name for name in required if name not in raw]
        if missing:
            raise TrainingFailure("invalid_training_config", "H3 cache is missing: %s" % ", ".join(missing))
        source_device = self._generator_device(generator)
        latent_frames = raw.get("latent_frames", 5)
        video = self._unpatch_video(
            torch.as_tensor(raw["video"]).float(),
            self.device,
            self.dtype,
            int(latent_frames) if isinstance(latent_frames, int) and not isinstance(latent_frames, bool) else latent_frames,
        )
        audio = self._unpack_audio(torch.as_tensor(raw["audio"]).float(), self.device, self.dtype)
        prompt = torch.as_tensor(raw["prompt"]).to(device=self.device, dtype=self.dtype)
        if prompt.ndim != 2 or prompt.shape[1] != 5120:
            raise TrainingFailure("invalid_training_config", "H3 prompt embeddings must have shape [tokens, 5120]")
        context = prompt.unsqueeze(0)
        noise_video = torch.randn(video.shape, generator=generator, device=source_device, dtype=torch.float32).to(
            device=self.device, dtype=self.dtype
        )
        noise_audio = torch.randn(audio.shape, generator=generator, device=source_device, dtype=torch.float32).to(
            device=self.device, dtype=self.dtype
        )
        sigma = torch.rand((1,), generator=generator, device=source_device, dtype=torch.float32).to(
            device=self.device, dtype=self.dtype
        )
        return PreparedBatch(
            conditioning=Conditioning(context),
            latents=ModalLatents(video=video, audio=audio),
            noise=ModalLatents(video=noise_video, audio=noise_audio),
            timesteps=ModalTimesteps(video=sigma, audio=self._shift_audio_sigma(sigma)),
            sample_ids=(str(raw.get("id", "h3-cache")),),
            metadata={"real_h3": True, "offline_simulation": False},
        )

    @staticmethod
    def _mix(clean: Optional[torch.Tensor], noise: Optional[torch.Tensor], sigma: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if clean is None or noise is None or sigma is None:
            return None
        value = sigma.to(device=clean.device, dtype=clean.dtype)
        while value.ndim < clean.ndim:
            value = value.unsqueeze(-1)
        return (1.0 - value) * clean + value * noise

    def add_noise(self, clean: ModalLatents, noise: ModalLatents, timestep: ModalTimesteps) -> ModalLatents:
        return ModalLatents(
            video=self._mix(clean.video, noise.video, timestep.video),
            audio=self._mix(clean.audio, noise.audio, timestep.audio),
        )

    def predict(
        self,
        role: ModelRole,
        noisy: ModalLatents,
        timestep: ModalTimesteps,
        conditioning: Conditioning,
    ) -> ModalPrediction:
        if noisy.video is None or noisy.audio is None or timestep.video is None:
            raise TrainingFailure("invalid_training_config", "real H3 forward requires video, audio, and video timestep")
        api = self._load_api()
        layout = api["PackedLayout"](
            int(conditioning.text.shape[1]),
            int(noisy.video.shape[2]),
            int(noisy.video.shape[3]),
            int(noisy.video.shape[4]),
            int(noisy.audio.shape[3]),
        )
        try:
            raw_video, raw_audio = role.model(
                [noisy.video, noisy.audio],
                timestep.video.to(dtype=torch.float32) * 1000.0,
                conditioning.text,
                transformer_options={},
                minimax_payload={"layout": layout, "audio_scale": 1.0},
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise TrainingFailure("forward_failed", "MiniMax-H3 forward failed: %s" % exc) from exc
        return ModalPrediction(video=-raw_video, audio=-raw_audio)

    def prediction_to_clean(self, noisy: ModalLatents, prediction: ModalPrediction, timestep: ModalTimesteps) -> ModalLatents:
        return ModalLatents(
            video=self._add_velocity(noisy.video, prediction.video, timestep.video),
            audio=self._add_velocity(noisy.audio, prediction.audio, timestep.audio),
        )

    @staticmethod
    def _add_velocity(
        noisy: Optional[torch.Tensor], prediction: Optional[torch.Tensor], sigma: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if noisy is None or prediction is None or sigma is None:
            return None
        value = sigma.to(device=noisy.device, dtype=noisy.dtype)
        while value.ndim < noisy.ndim:
            value = value.unsqueeze(-1)
        return noisy + value * prediction

    def scheduler_step(self, role: ModelRole, latent: ModalLatents, prediction: ModalPrediction, interval: ModalInterval) -> ModalLatents:
        def step(value, velocity, bounds):
            if value is None or velocity is None or bounds is None:
                return None
            return value + (bounds[0] - bounds[1]) * velocity

        return ModalLatents(
            video=step(latent.video, prediction.video, interval.video),
            audio=step(latent.audio, prediction.audio, interval.audio),
        )

    @staticmethod
    def _shifted_sigma(value: float, shift: float) -> float:
        return shift * value / (1.0 + (shift - 1.0) * value)

    def schedule(self, num_model_evaluations: int) -> ModalSchedule:
        if num_model_evaluations <= 0:
            raise ValueError("number of model evaluations must be positive")
        base = [1.0 - index / float(num_model_evaluations) for index in range(num_model_evaluations + 1)]
        video = tuple(self._shifted_sigma(value, self.video_shift) for value in base)
        audio = tuple(self._shifted_sigma(value, self.audio_shift) for value in base)
        return ModalSchedule(video_sigmas=video, audio_sigmas=audio)

    def resolve_trainable_parameters(self, role: ModelRole, policy: str) -> Iterable[str]:
        names = [name for name, _ in role.model.named_parameters()]
        available = set(names)
        if policy == "all":
            selected = names
        elif policy == "heads":
            selected = [name for name in names if name in TRAINABLE_TENSOR_NAMES]
        else:
            prefixes = tuple(item.strip() for item in policy.split(",") if item.strip())
            selected = [name for name in names if name.startswith(prefixes)]
        if not selected:
            raise TrainingFailure("no_trainable_parameters", policy)
        if policy == "heads" and set(selected) != set(TRAINABLE_TENSOR_NAMES):
            raise TrainingFailure("no_trainable_parameters", "H3 output-head scope is incomplete")
        return selected

    def save_role(self, role: ModelRole, path: Path) -> Mapping[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in role.model.state_dict().items()
        }
        metadata = {
            str(key): str(value)
            for key, value in dict(role.scheduler_state.get("checkpoint_metadata") or {}).items()
        }
        if "config" not in metadata:
            raise TrainingFailure("checkpoint_corrupt", "cannot save H3 child without transformer config metadata")
        try:
            save_file(state, str(path), metadata=metadata)
        except (OSError, RuntimeError, ValueError) as exc:
            raise TrainingFailure("checkpoint_save_failed", str(exc)) from exc
        return {
            "architecture_name": "MiniMax-H3-FL2VA",
            "dtype": str(self.dtype).replace("torch.", ""),
            "offline_simulation": False,
            "checkpoint_path": str(path),
        }

    def reload_role(self, path: Path) -> ModelRole:
        return self.load_role(path, trainable=False)


__all__ = ["RealMiniMaxH3Adapter", "TRAINABLE_TENSOR_NAMES"]
