"""Wan-native bridge for the shared H3 training algorithms.

The adapter owns Wan's latent geometry and Flow-Match convention while the
training methods remain the existing Progressive Distillation and DMD2
implementations.  H3 targets enter through RGB and are encoded with the Wan
VAE; H3 cache rows are never treated as Wan latent tensors.
"""

from __future__ import annotations

import copy
import fnmatch
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
from torch import Generator, nn

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
from harness4h3.student.wan_schedule import WanSigmaSchedule, training_uniform_wan_schedule


class WanEndpointTeacher(nn.Module):
    """Parameter-free role marker for the H3 RGB-boundary endpoint teacher."""

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return hidden_states


class WanTrainingAdapter(DenoisingModelAdapter):
    """Adapt the existing algorithms to a Wan Transformer role."""

    generation_start_timestep = 1.0

    def __init__(
        self,
        *,
        transformer: nn.Module,
        scheduler: Any,
        encode_frames: Callable[[Any], torch.Tensor],
        encode_prompt: Callable[[str], torch.Tensor],
        save_checkpoint: Optional[Callable[[nn.Module, Path, Mapping[str, Any]], str]] = None,
        reload_checkpoint: Optional[Callable[[Path], ModelRole]] = None,
        schedule_provider: Optional[Callable[[int], WanSigmaSchedule]] = None,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        guidance_scale: float = 1.0,
    ) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.encode_frames = encode_frames
        self.encode_prompt = encode_prompt
        self.save_checkpoint = save_checkpoint
        self.reload_checkpoint = reload_checkpoint
        self.model_id = str(model_id)
        self.guidance_scale = float(guidance_scale)
        if not torch.isfinite(torch.tensor(self.guidance_scale)) or self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be finite and positive")
        self.schedule_provider = schedule_provider or (
            lambda nfe: training_uniform_wan_schedule(int(nfe), int(self._train_timesteps()))
        )
        # The shared algorithm transports sigma values, while Wan's native
        # Transformer consumes scheduler timesteps.  Keep the exact sigma ->
        # timestep tables from every schedule requested by the algorithm so
        # native integer timestep quantization is not lost at the adapter
        # boundary.
        self._model_timestep_tables: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        self._active_clean: Optional[torch.Tensor] = None
        # UniPC and similar Wan schedulers carry multistep history.  The
        # progressive-distillation algorithm creates independent teacher and
        # student trajectories, so each role needs a fresh scheduler context.
        self._trajectory_schedulers: dict[int, tuple[Any, int]] = {}
        # A multistep scheduler is history-dependent.  The training method
        # must replay the prefix of the real trajectory instead of treating
        # every sigma interval as an independent Euler-like transition.
        self.native_trajectory_training = callable(getattr(scheduler, "set_timesteps", None)) and callable(
            getattr(scheduler, "step", None)
        )

    @staticmethod
    def _sigma(timestep: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Sigma is a scheduler coordinate, not a model activation.  Keeping
        # it in FP32 is required for the Flow-Match equations and for exact
        # native-timestep lookup; casting it to the BF16 latent dtype first
        # perturbs the low-noise endpoint used by 16-step inference.
        value = timestep.to(device=reference.device, dtype=torch.float32)
        while value.ndim < reference.ndim:
            value = value.unsqueeze(-1)
        return value

    def _train_timesteps(self) -> float:
        config = getattr(self.scheduler, "config", None)
        return float(getattr(config, "num_train_timesteps", 1000.0))

    def _model_timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map algorithm sigma values to the exact native Wan timesteps."""

        fallback = sigma.float() * self._train_timesteps()
        if not self._model_timestep_tables:
            return fallback
        values = sigma.float().reshape(-1)
        best_distance = torch.full_like(values, float("inf"))
        best_timestep = fallback.reshape(-1)
        for sigmas, timesteps in self._model_timestep_tables:
            schedule_sigmas = torch.as_tensor(sigmas, device=values.device, dtype=torch.float32)
            schedule_timesteps = torch.as_tensor(timesteps, device=values.device, dtype=torch.float32)
            distance = (values[:, None] - schedule_sigmas[None, :]).abs()
            local_distance, local_index = distance.min(dim=1)
            local_timestep = schedule_timesteps[local_index]
            selected = local_distance < best_distance
            best_distance = torch.where(selected, local_distance, best_distance)
            best_timestep = torch.where(selected, local_timestep, best_timestep)
        return best_timestep.reshape_as(sigma)

    @staticmethod
    def _prediction_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            if not output:
                raise TrainingFailure("wan_adapter_forward_failed", "Wan Transformer returned an empty tuple")
            output = output[0]
        if hasattr(output, "sample"):
            output = output.sample
        if not isinstance(output, torch.Tensor):
            raise TrainingFailure("wan_adapter_forward_failed", "Wan Transformer did not return a tensor")
        return output

    def prepare_batch(self, raw: Any, generator: Generator) -> PreparedBatch:
        if isinstance(raw, PreparedBatch):
            return raw
        if not isinstance(raw, Mapping):
            raise TrainingFailure("invalid_training_config", "Wan batches must be mappings")
        if "frames" not in raw or "prompt" not in raw:
            raise TrainingFailure("invalid_training_config", "Wan batch requires frames and prompt")
        clean = raw.get("wan_latent")
        if clean is None:
            clean = self.encode_frames(raw["frames"])
        else:
            clean = torch.as_tensor(clean)
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5:
            raise TrainingFailure("wan_video_shape_invalid", "Wan frame encoder must return [B,C,T,H,W]")
        if not torch.isfinite(clean).all():
            raise TrainingFailure("wan_video_nonfinite", "Wan clean latent contains non-finite values")
        text = raw.get("prompt_embeds")
        negative_text = raw.get("negative_prompt_embeds")
        if text is None:
            encoded = self.encode_prompt(str(raw["prompt"]))
            if isinstance(encoded, (tuple, list)):
                if len(encoded) != 2:
                    raise TrainingFailure("wan_conditioning_invalid", "Wan prompt encoder must return text and negative text")
                text, negative_text = encoded
            else:
                text = encoded
        else:
            text = torch.as_tensor(text)
            if negative_text is not None:
                negative_text = torch.as_tensor(negative_text)
        if not isinstance(text, torch.Tensor) or text.ndim < 2:
            raise TrainingFailure("wan_conditioning_invalid", "Wan prompt encoder must return [B,S,D]")
        if text.shape[0] != clean.shape[0]:
            raise TrainingFailure("wan_conditioning_invalid", "prompt and latent batch sizes differ")
        if negative_text is not None:
            if not isinstance(negative_text, torch.Tensor) or negative_text.ndim < 2:
                raise TrainingFailure("wan_conditioning_invalid", "Wan negative prompt encoder must return [B,S,D]")
            if negative_text.shape != text.shape:
                raise TrainingFailure("wan_conditioning_invalid", "positive and negative prompt shapes differ")
        # TrainerEngine's CPU generator is intentional: sampling on CPU and
        # moving once gives deterministic batches on both CPU and CUDA hosts.
        noise = torch.randn(clean.shape, generator=generator, device="cpu", dtype=torch.float32)
        noise = noise.to(device=clean.device, dtype=clean.dtype)
        self._active_clean = clean.detach()
        sample_id = str(raw.get("sample_id", raw.get("id", "wan-sample")))
        return PreparedBatch(
            conditioning=Conditioning(
                text=text.detach(),
                negative_text=negative_text.detach() if negative_text is not None else None,
                metadata={"model_id": self.model_id, "guidance_scale": self.guidance_scale},
            ),
            latents=ModalLatents(video=clean),
            noise=ModalLatents(video=noise),
            sample_ids=(sample_id,),
            metadata={"prompt": str(raw["prompt"])},
        )

    def add_noise(self, clean: ModalLatents, noise: ModalLatents, timestep: ModalTimesteps) -> ModalLatents:
        if clean.video is None or noise.video is None or timestep.video is None:
            raise TrainingFailure("invalid_training_config", "Wan adapter requires video clean/noise/timestep")
        sigma = self._sigma(timestep.video, clean.video).float()
        value = (1.0 - sigma) * clean.video.float() + sigma * noise.video.float()
        return ModalLatents(video=value.to(dtype=clean.video.dtype))

    def predict(
        self,
        role: ModelRole,
        noisy: ModalLatents,
        timestep: ModalTimesteps,
        conditioning: Conditioning,
    ) -> ModalPrediction:
        if noisy.video is None or timestep.video is None:
            raise TrainingFailure("invalid_training_config", "Wan adapter requires video prediction inputs")
        # Preserve the scheduler coordinate in FP32.  Only hidden states and
        # conditioning are converted to the Transformer dtype below.
        sigma = timestep.video.to(device=noisy.video.device, dtype=torch.float32)
        if isinstance(role.model, WanEndpointTeacher):
            if self._active_clean is None:
                raise TrainingFailure("wan_teacher_target_missing", "H3 RGB-boundary target was not prepared")
            clean = self._active_clean.to(device=noisy.video.device, dtype=noisy.video.dtype)
            denominator = sigma.float().clamp_min(torch.finfo(torch.float32).eps)
            velocity = (noisy.video.float() - clean.float()) / denominator.view(-1, 1, 1, 1, 1)
            return ModalPrediction(video=velocity.to(dtype=noisy.video.dtype).detach())
        model_timestep = self._model_timestep(sigma)
        model_device = noisy.video.device
        patch_embedding = getattr(role.model, "patch_embedding", None)
        if patch_embedding is not None:
            model_dtype = next(patch_embedding.parameters()).dtype
        else:
            model_dtype = next(role.model.parameters(), noisy.video).dtype
        hidden_states = noisy.video.to(device=model_device, dtype=model_dtype)
        timestep = model_timestep.to(device=model_device, dtype=torch.float32)

        def forward(text: torch.Tensor) -> torch.Tensor:
            output = role.model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=text.to(device=model_device, dtype=model_dtype),
                return_dict=False,
            )
            return self._prediction_tensor(output)

        prediction = forward(conditioning.text)
        negative_text = conditioning.negative_text
        if negative_text is not None and self.guidance_scale != 1.0:
            unconditional = forward(negative_text)
            prediction = unconditional + self.guidance_scale * (prediction - unconditional)
        if prediction.shape != noisy.video.shape:
            raise TrainingFailure(
                "wan_adapter_forward_failed",
                "Wan prediction shape %s does not match latent shape %s" % (tuple(prediction.shape), tuple(noisy.video.shape)),
            )
        return ModalPrediction(video=prediction)

    def prediction_to_clean(
        self,
        noisy: ModalLatents,
        prediction: ModalPrediction,
        timestep: ModalTimesteps,
    ) -> ModalLatents:
        if noisy.video is None or prediction.video is None or timestep.video is None:
            raise TrainingFailure("invalid_training_config", "Wan adapter requires video clean reconstruction inputs")
        sigma = self._sigma(timestep.video, noisy.video).float()
        value = noisy.video.float() - sigma * prediction.video.float()
        return ModalLatents(video=value.to(dtype=noisy.video.dtype))

    def _clone_native_scheduler(self) -> Any:
        """Clone the configured Diffusers scheduler without sharing history."""

        try:
            return copy.deepcopy(self.scheduler)
        except Exception as deepcopy_error:
            constructor = getattr(type(self.scheduler), "from_config", None)
            config = getattr(self.scheduler, "config", None)
            if callable(constructor) and config is not None:
                try:
                    return constructor(config)
                except Exception as config_error:
                    raise TrainingFailure(
                        "wan_scheduler_init_failed",
                        "unable to clone native scheduler: %s; from_config: %s"
                        % (deepcopy_error, config_error),
                    ) from config_error
            raise TrainingFailure(
                "wan_scheduler_init_failed",
                "unable to clone native scheduler: %s" % deepcopy_error,
            ) from deepcopy_error

    @staticmethod
    def _scheduler_sample(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            if not output:
                raise TrainingFailure("wan_scheduler_step_failed", "native scheduler returned an empty tuple")
            output = output[0]
        if hasattr(output, "prev_sample"):
            output = output.prev_sample
        if not isinstance(output, torch.Tensor):
            raise TrainingFailure("wan_scheduler_step_failed", "native scheduler did not return a tensor")
        return output

    def begin_trajectory(self, role: ModelRole, schedule: ModalSchedule, start_index: int, device: Any) -> None:
        """Create a fresh native scheduler state for one role trajectory."""

        key = id(role)
        self._trajectory_schedulers.pop(key, None)
        set_timesteps = getattr(self.scheduler, "set_timesteps", None)
        step = getattr(self.scheduler, "step", None)
        sigmas = schedule.video_sigmas
        if not callable(set_timesteps) or not callable(step) or sigmas is None:
            return
        nfe = len(sigmas) - 1
        start = int(start_index)
        if nfe <= 0 or start < 0 or start >= nfe:
            raise TrainingFailure("wan_scheduler_trajectory_invalid", "trajectory index is outside the schedule")
        native = self._clone_native_scheduler()
        try:
            native.set_timesteps(nfe, device=device)
            native_sigmas = getattr(native, "sigmas", None)
            if native_sigmas is not None:
                expected = torch.as_tensor(sigmas[:-1], dtype=torch.float32)
                actual = torch.as_tensor(native_sigmas[:nfe], dtype=torch.float32, device="cpu")
                if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=5e-5, rtol=0.0):
                    raise TrainingFailure(
                        "wan_scheduler_schedule_mismatch",
                        "native scheduler sigmas do not match the training schedule",
                    )
            set_begin_index = getattr(native, "set_begin_index", None)
            if callable(set_begin_index):
                set_begin_index(start)
            elif start != 0:
                raise TrainingFailure(
                    "wan_scheduler_trajectory_invalid",
                    "native scheduler cannot start at a nonzero trajectory index",
                )
        except TrainingFailure:
            raise
        except Exception as exc:
            raise TrainingFailure("wan_scheduler_init_failed", "native scheduler setup failed: %s" % exc) from exc
        self._trajectory_schedulers[key] = (native, start)

    def scheduler_step(
        self,
        role: ModelRole,
        latent: ModalLatents,
        prediction: ModalPrediction,
        interval: ModalInterval,
    ) -> ModalLatents:
        if latent.video is None or prediction.video is None or interval.video is None:
            raise TrainingFailure("invalid_training_config", "Wan adapter requires video scheduler inputs")
        current_sigma, next_sigma = interval.video
        context = self._trajectory_schedulers.get(id(role))
        if context is not None:
            native, index = context
            native_sigmas = getattr(native, "sigmas", None)
            native_timesteps = getattr(native, "timesteps", None)
            if native_sigmas is None or native_timesteps is None or index >= len(native_timesteps):
                raise TrainingFailure("wan_scheduler_step_failed", "native scheduler state is incomplete")
            expected_current = float(current_sigma)
            expected_next = float(next_sigma)
            actual_current = float(native_sigmas[index])
            actual_next = float(native_sigmas[index + 1])
            if abs(actual_current - expected_current) > 5e-5 or abs(actual_next - expected_next) > 5e-5:
                raise TrainingFailure(
                    "wan_scheduler_schedule_mismatch",
                    "native scheduler interval does not match the training interval",
                )
            try:
                output = native.step(
                    prediction.video,
                    native_timesteps[index],
                    latent.video,
                    return_dict=False,
                )
            except Exception as exc:
                raise TrainingFailure("wan_scheduler_step_failed", "native scheduler step failed: %s" % exc) from exc
            value = self._scheduler_sample(output)
            if value.shape != latent.video.shape:
                raise TrainingFailure("wan_scheduler_step_failed", "native scheduler changed latent shape")
            self._trajectory_schedulers[id(role)] = (native, index + 1)
            return ModalLatents(video=value.to(dtype=latent.video.dtype))
        delta = float(next_sigma) - float(current_sigma)
        value = latent.video.float() + delta * prediction.video.float()
        return ModalLatents(video=value.to(dtype=latent.video.dtype))

    def schedule(self, num_model_evaluations: int) -> ModalSchedule:
        if isinstance(num_model_evaluations, bool) or int(num_model_evaluations) <= 0:
            raise ValueError("number of model evaluations must be positive")
        schedule = self.schedule_provider(int(num_model_evaluations))
        table = (tuple(schedule.sigmas[:-1]), tuple(schedule.timesteps))
        if table not in self._model_timestep_tables:
            self._model_timestep_tables.append(table)
        return ModalSchedule(video_sigmas=schedule.sigmas)

    def save_role(self, role: ModelRole, path: Path) -> Mapping[str, Any]:
        metadata = dict(role.scheduler_state)
        metadata.setdefault("model_id", self.model_id)
        metadata.setdefault("wan_scheduler_class", type(self.scheduler).__name__)
        metadata.setdefault(
            "wan_scheduler_integrator",
            "native_multistep" if callable(getattr(self.scheduler, "step", None)) else "euler_fallback",
        )
        if self.save_checkpoint is None:
            from harness4h3.student.wan_training import save_wan_checkpoint

            digest = save_wan_checkpoint(role.model, Path(path), metadata)
        else:
            digest = self.save_checkpoint(role.model, Path(path), metadata)
        return {"checkpoint": str(Path(path)), "checkpoint_sha256": str(digest), **metadata}

    def reload_role(self, path: Path) -> ModelRole:
        if self.reload_checkpoint is None:
            raise TrainingFailure("wan_adapter_reload_unavailable", "Wan adapter has no trusted checkpoint loader")
        return self.reload_checkpoint(Path(path))

    def resolve_trainable_parameters(self, role: ModelRole, policy: str) -> Iterable[str]:
        names = tuple(name for name, _ in role.model.named_parameters())
        selected = str(policy or "all").strip()
        if selected == "all":
            return names
        if selected == "heads":
            return tuple(name for name in names if any(token in name.lower() for token in ("head", "proj_out", "output")))
        patterns = tuple(item.strip() for item in selected.split(",") if item.strip())
        result = tuple(name for name in names if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns))
        if not result:
            raise TrainingFailure("no_trainable_parameters", selected)
        return result


__all__ = ["WanEndpointTeacher", "WanTrainingAdapter"]
