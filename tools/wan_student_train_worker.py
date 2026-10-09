#!/usr/bin/env python3
"""Run one real Wan2.1-T2V-1.3B 5-second H3-distillation experiment."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness4h3.student.inference import decode_video_latent, load_h3_cache_item
from harness4h3.student.evaluation_manifest import CacheContract, validate_h3_cache_item
from harness4h3.student.proposal import StudentTarget
from harness4h3.student.schedule import h3_video_sigmas
from harness4h3.student.teacher_service import TeacherService
from harness4h3.student.wan import WanStudentError, load_wan_spec
from harness4h3.student.wan_long_video import load_wan_transformer_checkpoint
from harness4h3.student.wan_schedule import WanSigmaSchedule, coarsen_wan_schedule, native_wan_schedule
from harness4h3.student.wan_training import (
    WanTrainingSpec,
    encode_wan_video,
    flow_match_batch,
    load_wan_trajectory_teacher,
    load_wan_training_components,
    resample_video_frames,
    save_wan_checkpoint,
    sha256_path,
)
from h3_training.adapters.wan import WanEndpointTeacher, WanTrainingAdapter
from h3_training.algorithms.dmd2 import DMD2, DMD2Config
from h3_training.algorithms.progressive_distillation import (
    ProgressiveDistillation,
    ProgressiveDistillationConfig,
    plan_binary_stages,
)
from h3_training.engine.trainer import TrainerConfig, TrainerEngine
from harness4h3.student.wan_campaign import WanTrainingPolicy
from h3_training.data.schema import Conditioning, ModalLatents, ModalTimesteps


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _stage_checkpoint_path(base: Path, stage_index: int) -> Path:
    return base.with_name("%s-stage-%02d%s" % (base.stem, int(stage_index), base.suffix))


def _load_mapping(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("wan_training"), Mapping):
        raise ValueError("config requires a wan_training mapping")
    return raw["wan_training"]


def _training_spec(raw: Mapping[str, Any], wan_config: Path) -> WanTrainingSpec:
    wan = load_wan_spec(wan_config)
    values = {
        name: getattr(wan, name)
        for name in (
            "model_id",
            "pretrained_path",
            "lineage_root",
            "width",
            "height",
            "num_frames",
            "native_fps",
            "dtype",
            "guidance_scale",
        )
    }
    for name in (
        "max_sequence_length",
        "learning_rate",
        "standard_loss_weight",
        "distill_loss_weight",
        "train_steps",
        "h3_sampling_steps",
        "h3_latent_frames",
        "h3_latent_height",
        "h3_latent_width",
        "h3_latent_channels",
        "vae_name",
    ):
        if name in raw:
            values[name] = raw[name]
    return WanTrainingSpec(**values)


def _wan_schedule_provider(scheduler: Any, *, source_steps: int):
    """Bind every progressive stage to the native Wan source grid."""

    source = native_wan_schedule(scheduler, int(source_steps))

    def provide(nfe: int) -> WanSigmaSchedule:
        requested = int(nfe)
        if requested <= 0:
            raise ValueError("Wan distillation schedule nfe must be positive")
        if requested == source.nfe:
            return source
        if source.nfe % requested:
            raise ValueError("Wan distillation schedule requires a divisor of source_steps")
        return coarsen_wan_schedule(source, source.nfe // requested)

    return provide


def _parse_devices(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = tuple(item.strip() for item in raw.split(",") if item.strip())
    elif isinstance(raw, (list, tuple)):
        values = tuple(str(item).strip() for item in raw if str(item).strip())
    else:
        values = ()
    if len(values) != TeacherService.REQUIRED_WORLD_SIZE:
        raise ValueError("teacher_devices must contain exactly three cuda:N devices")
    if len(set(values)) != len(values) or any(not item.startswith("cuda:") for item in values):
        raise ValueError("teacher_devices must contain three distinct explicit cuda:N devices")
    return values


def _cache_contract(raw: Mapping[str, Any]) -> CacheContract | None:
    selected = str(raw.get("cache_contract", raw.get("continuation_contract", ""))).strip()
    if not selected:
        return None
    if selected == "native_5s":
        return CacheContract.native_5s()
    raise ValueError("unsupported H3 cache contract: %s" % selected)


def _training_cache_paths(raw: Mapping[str, Any], *, limit: int) -> list[Path]:
    """Select one real H3 cache item per caption for prompt-diverse training."""

    cache_dir = Path(str(raw["h3_cache_dir"])).resolve()
    paths = sorted(cache_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError("no H3 cache items under %s" % cache_dir)
    selected: list[Path] = []
    captions: set[str] = set()
    contract = _cache_contract(raw)
    for path in paths:
        item = validate_h3_cache_item(path, contract) if contract is not None else torch.load(
            path, map_location="cpu", weights_only=False
        )
        caption = str(item.get("caption", "")).strip() if isinstance(item, Mapping) else ""
        del item
        key = caption or path.name
        if key in captions:
            continue
        captions.add(key)
        selected.append(path)
        if len(selected) >= int(limit):
            break
    if not selected:
        raise ValueError("H3 cache contains no usable captions")
    return selected


def _release_h3_vae() -> None:
    try:
        from harness4h3.student import inference

        inference._VAE_CACHE.clear()
    except Exception:
        pass
    try:
        import comfy.model_management as model_management

        if hasattr(model_management, "unload_all_models"):
            model_management.unload_all_models()
        if hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _encode_training_prompt(
    pipeline: Any, prompt: str, spec: WanTrainingSpec, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode frozen text without creating inference-mode tensors.

    The returned embeddings are consumed by the trainable Wan Transformer.
    ``inference_mode`` tensors cannot be saved for backward by autograd even
    when they are detached, so use ``no_grad`` and clone into normal tensors.
    """

    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=float(spec.guidance_scale) != 1.0,
            num_videos_per_prompt=1,
            max_sequence_length=spec.max_sequence_length,
            device=device,
            dtype=dtype,
        )
    if float(spec.guidance_scale) != 1.0 and negative_prompt_embeds is None:
        raise WanStudentError("wan_cfg_conditioning_missing", "Wan CFG requires negative prompt embeddings")
    return (
        prompt_embeds.detach().clone(),
        negative_prompt_embeds.detach().clone()
        if negative_prompt_embeds is not None
        else torch.empty_like(prompt_embeds),
    )


def _configure_trainable_scope(transformer: Any, scope: str | None) -> str:
    """Freeze the Wan backbone when a semantic adapter scope is requested."""

    selected = str(scope or "all").strip()
    if not selected or selected == "all":
        for parameter in transformer.parameters():
            parameter.requires_grad_(True)
        return "all"
    patterns = tuple(item.strip() for item in selected.split(",") if item.strip())
    if not patterns:
        raise ValueError("trainable_scope must be 'all' or contain at least one pattern")
    trainable = 0
    for name, parameter in transformer.named_parameters():
        enabled = any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        parameter.requires_grad_(enabled)
        trainable += int(enabled)
    if trainable <= 0:
        raise ValueError("trainable_scope selected no Wan Transformer parameters: %s" % selected)
    return ",".join(patterns)


def _trainable_parameter_probe(transformer: Any) -> tuple[str, torch.Tensor]:
    """Capture one actual trainable parameter for the post-step change gate."""

    for name, parameter in transformer.named_parameters():
        if parameter.requires_grad:
            return name, parameter.detach().float().clone()
    raise ValueError("transformer has no trainable parameters")


def _apply_initialization_mode(transformer: Any, mode: str | None) -> str:
    selected = str(mode or "inherit").strip()
    if selected == "inherit":
        return selected
    if selected not in {"generation_head_reset", "dit_zero"}:
        raise ValueError("unsupported Wan initialization mode: %s" % selected)
    candidates = []
    for name, module in transformer.named_modules():
        lowered = name.lower()
        if any(token in lowered for token in ("proj_out", "output_proj", "generation_head", "output_head")):
            if hasattr(module, "weight") and hasattr(module, "bias"):
                candidates.append((name, module))
    if not candidates:
        raise WanStudentError("initialization_unsupported", "Wan Transformer has no resettable generation output projection")
    with torch.no_grad():
        for _, module in candidates:
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()
    return selected


def _teacher_video(
    raw: Mapping[str, Any],
    spec: WanTrainingSpec,
    teacher_device: torch.device,
    decode_device: torch.device,
    *,
    cache_path: Optional[Path] = None,
) -> tuple[torch.Tensor, int, str, Mapping[str, Any]]:
    teacher_checkpoint = Path(str(raw["teacher_checkpoint"])).resolve()
    cache_dir = Path(str(raw["h3_cache_dir"])).resolve()
    comfyui_root = Path(str(raw["comfyui_root"])).resolve()
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError("H3 Teacher checkpoint does not exist: %s" % teacher_checkpoint)
    if not cache_dir.is_dir():
        raise FileNotFoundError("H3 cache directory does not exist: %s" % cache_dir)
    target = StudentTarget(
        latent_channels=spec.h3_latent_channels,
        latent_frames=spec.h3_latent_frames,
        latent_height=spec.h3_latent_height,
        latent_width=spec.h3_latent_width,
        condition_dim=5120,
    )
    target_mode = str(raw.get("teacher_target_mode", "online_h3")).strip()
    if target_mode not in {"online_h3", "cache_rgb"}:
        raise ValueError("unsupported teacher_target_mode: %s" % target_mode)
    cache_contract = _cache_contract(raw)
    cache_metadata: Mapping[str, Any] = {}
    if target_mode == "cache_rgb":
        if cache_contract is None or cache_path is None:
            raise ValueError("teacher_target_mode=cache_rgb requires the native H3 cache contract and cache item")
        cache_metadata = validate_h3_cache_item(cache_path, cache_contract)
    cache_item = load_h3_cache_item(
        cache_dir,
        target,
        teacher_device,
        torch.bfloat16,
        cache_path=cache_path,
        cache_contract=cache_contract,
    )
    if target_mode == "cache_rgb":
        # The native cache already contains an authenticated real H3 Teacher
        # video latent.  Decode that fixed target directly; do not silently
        # replace the requested single-Teacher video with a fresh random H3
        # sample.  This also makes cache RGB finite-ness a training invariant.
        latent = cache_item["latent"].to(device=decode_device, dtype=torch.bfloat16)
        if not torch.isfinite(latent).all():
            raise WanStudentError(
                "wan_cache_latent_nonfinite",
                "native H3 cache latent is non-finite for cache item %s" % cache_path,
            )
        frames = decode_video_latent(comfyui_root, spec.vae_name, latent)
        _release_h3_vae()
        if not torch.isfinite(frames).all():
            raise WanStudentError(
                "wan_cache_video_nonfinite",
                "native H3 cache decoded non-finite RGB frames for cache item %s" % cache_path,
            )
        frames = resample_video_frames(frames, num_frames=spec.num_frames, height=spec.height, width=spec.width)
        if not torch.isfinite(frames).all():
            raise WanStudentError(
                "wan_cache_video_nonfinite",
                "native H3 cache RGB resampling produced non-finite frames for cache item %s" % cache_path,
            )
        teacher_digest = sha256_path(teacher_checkpoint)
        cached_teacher_digest = str(cache_metadata.get("teacher_checkpoint_sha256", "")).strip()
        if cached_teacher_digest and cached_teacher_digest != teacher_digest:
            raise WanStudentError(
                "wan_cache_teacher_mismatch",
                "native H3 cache Teacher digest does not match configured Teacher for cache item %s" % cache_path,
            )
        metadata = {
            "teacher_sampling_steps": int(cache_metadata.get("teacher_sampling_steps", 0)),
            "teacher_forward_count": 0,
            "teacher_source": "native_h3_cache_rgb",
            "teacher_cache_generated": bool(cache_metadata.get("teacher_generated", False)),
            "teacher_wall_time_s": 0.0,
            "teacher_checkpoint_sha256": teacher_digest,
            "caption": str(cache_item.get("caption", "")).strip(),
            "cache_item": str(cache_item["path"]),
        }
        return frames, 0, teacher_digest, metadata
    prompt = cache_item["prompt"]
    h3_latent = cache_item["latent"]
    h3_audio = cache_item.get("audio_latent")
    if h3_audio is None:
        raise ValueError("H3 cache item lacks audio_latent required by the real Teacher")
    service = TeacherService(
        teacher_checkpoint,
        comfyui_root,
        _parse_devices(raw["teacher_devices"]),
        dtype=torch.bfloat16,
    ).start()
    started = time.perf_counter()
    try:
        generator = torch.Generator(device=str(teacher_device)).manual_seed(int(raw.get("seed", 20261008)))
        video = torch.randn(h3_latent.shape, generator=generator, device=teacher_device, dtype=torch.bfloat16)
        audio = torch.randn(h3_audio.shape, generator=generator, device=teacher_device, dtype=torch.bfloat16)
        video_sigmas = h3_video_sigmas(int(spec.h3_sampling_steps))
        audio_sigmas = tuple(
            3.0 * (1.0 - index / float(spec.h3_sampling_steps))
            / (1.0 + 2.0 * (1.0 - index / float(spec.h3_sampling_steps)))
            for index in range(int(spec.h3_sampling_steps) + 1)
        )
        # Keep the authenticated MiniMax-H3 token-routing metadata from the
        # cache.  The baseline worker already sends these tags; omitting them
        # here makes the native z=24 short-clip path numerically unstable even
        # though the same Teacher/cache pair decodes to a valid 81-frame video.
        conditioning_metadata = (
            {"text_token_tags": cache_item["text_token_tags"]}
            if cache_item.get("text_token_tags") is not None
            else {}
        )
        conditioning = Conditioning(
            prompt.to(device=teacher_device, dtype=torch.bfloat16),
            metadata=conditioning_metadata,
        )
        with torch.inference_mode():
            for index in range(int(spec.h3_sampling_steps)):
                timestep = ModalTimesteps(
                    video=torch.full((1,), float(video_sigmas[index]), device=teacher_device, dtype=torch.float32),
                    audio=torch.full((1,), float(audio_sigmas[index]), device=teacher_device, dtype=torch.float32),
                )
                prediction = service.predict(ModalLatents(video=video, audio=audio), timestep, conditioning)
                if prediction.video is None or prediction.audio is None:
                    raise RuntimeError("real H3 Teacher returned incomplete prediction")
                video = video + (video_sigmas[index] - video_sigmas[index + 1]) * prediction.video.to(
                    device=teacher_device, dtype=torch.bfloat16
                )
                audio = audio + (audio_sigmas[index] - audio_sigmas[index + 1]) * prediction.audio.to(
                    device=teacher_device, dtype=torch.bfloat16
                )
                if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
                    raise WanStudentError(
                        "wan_teacher_latent_nonfinite",
                        "H3 Teacher produced non-finite latent values for cache item %s at step %d"
                        % (cache_path, index),
                    )
        forward_count = int(service.forward_count)
        if forward_count <= 0:
            raise RuntimeError("real H3 Teacher completed without a forward")
        teacher_digest = sha256_path(teacher_checkpoint)
    finally:
        forward_count = int(service.forward_count)
        service.close()
    frames = decode_video_latent(comfyui_root, spec.vae_name, video.to(device=decode_device, dtype=torch.bfloat16))
    _release_h3_vae()
    if not torch.isfinite(frames).all():
        raise WanStudentError(
            "wan_teacher_video_nonfinite",
            "H3 VAE decoded non-finite RGB frames for cache item %s" % cache_path,
        )
    frames = resample_video_frames(frames, num_frames=spec.num_frames, height=spec.height, width=spec.width)
    if not torch.isfinite(frames).all():
        raise WanStudentError(
            "wan_teacher_video_nonfinite",
            "H3 RGB resampling produced non-finite frames for cache item %s" % cache_path,
        )
    metadata = {
        "teacher_sampling_steps": int(spec.h3_sampling_steps),
        "teacher_forward_count": int(forward_count),
        "teacher_wall_time_s": float(time.perf_counter() - started),
        "teacher_checkpoint_sha256": teacher_digest,
        "caption": str(cache_item.get("caption", "")).strip(),
        "cache_item": str(cache_item["path"]),
    }
    return frames, forward_count, teacher_digest, metadata


def run(
    raw: Mapping[str, Any],
    *,
    output: Path,
    result_path: Path,
    steps_override: int | None = None,
    reuse_override: int | None = None,
    parent_override: Path | None = None,
    learning_rate_override: float | None = None,
    trainable_scope_override: str | None = None,
    method_override: str | None = None,
    source_steps_override: int | None = None,
    target_steps_override: int | None = None,
    rollout_steps_override: int | None = None,
    rollout_every_override: int | None = None,
    rollout_endpoint_weight_override: float | None = None,
    rollout_clean_weight_override: float | None = None,
    endpoint_weight_override: float | None = None,
    velocity_weight_override: float | None = None,
    clean_anchor_weight_override: float | None = None,
    fp32_optimizer_state_override: bool | None = None,
    initialization_mode_override: str | None = None,
    training_checkpoint_override: Path | None = None,
    checkpoint_every_steps_override: int | None = None,
    dmd2_memory_admitted_override: bool | None = None,
    wan_dmd2_adapter_ready_override: bool | None = None,
) -> Mapping[str, Any]:
    wan_config = Path(str(raw["wan_config"])).resolve()
    spec = _training_spec(raw, wan_config)
    steps = int(steps_override if steps_override is not None else raw.get("train_steps", spec.train_steps))
    if steps <= 0:
        raise ValueError("train_steps must be positive")
    teacher_reuse_steps = int(reuse_override if reuse_override is not None else raw.get("teacher_reuse_steps", 1))
    if teacher_reuse_steps <= 0:
        raise ValueError("teacher_reuse_steps must be positive")
    learning_rate = float(
        learning_rate_override if learning_rate_override is not None else raw.get("learning_rate", spec.learning_rate)
    )
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    method = str(method_override or raw.get("method", "progressive_distillation"))
    source_steps = int(source_steps_override if source_steps_override is not None else raw.get("source_steps", 32))
    target_steps = int(target_steps_override if target_steps_override is not None else raw.get("target_steps", 8))
    rollout_steps = int(rollout_steps_override if rollout_steps_override is not None else raw.get("rollout_steps", 0))
    rollout_every = int(rollout_every_override if rollout_every_override is not None else raw.get("rollout_every", teacher_reuse_steps))
    rollout_endpoint_weight = float(
        rollout_endpoint_weight_override
        if rollout_endpoint_weight_override is not None
        else raw.get("rollout_endpoint_weight", 0.0)
    )
    rollout_clean_weight = float(
        rollout_clean_weight_override
        if rollout_clean_weight_override is not None
        else raw.get("rollout_clean_weight", 0.0)
    )
    endpoint_weight = float(
        endpoint_weight_override if endpoint_weight_override is not None else raw.get("endpoint_weight", 1.0)
    )
    velocity_weight = float(
        velocity_weight_override if velocity_weight_override is not None else raw.get("velocity_weight", 1.0)
    )
    clean_anchor_weight = float(
        clean_anchor_weight_override
        if clean_anchor_weight_override is not None
        else raw.get("clean_anchor_weight", 0.5)
    )
    fp32_optimizer_state = bool(
        fp32_optimizer_state_override
        if fp32_optimizer_state_override is not None
        else raw.get("fp32_optimizer_state", False)
    )
    checkpoint_every_steps = int(
        checkpoint_every_steps_override
        if checkpoint_every_steps_override is not None
        else raw.get("checkpoint_every_steps", 8)
    )
    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must not be negative")
    dmd2_memory_admitted = bool(
        dmd2_memory_admitted_override
        if dmd2_memory_admitted_override is not None
        else raw.get("dmd2_memory_admitted", False)
    )
    wan_dmd2_adapter_ready = bool(
        wan_dmd2_adapter_ready_override
        if wan_dmd2_adapter_ready_override is not None
        else raw.get("wan_dmd2_adapter_ready", False)
    )
    policy = WanTrainingPolicy.from_mapping(
        {
            "method": method,
            "source_steps": source_steps,
            "target_steps": target_steps,
            "train_steps": steps,
            "learning_rate": learning_rate,
            "critic_learning_rate": float(raw.get("critic_learning_rate", learning_rate)),
            "endpoint_weight": endpoint_weight,
            "velocity_weight": velocity_weight,
            "clean_anchor_weight": clean_anchor_weight,
            "rollout_steps": rollout_steps,
            "rollout_every": rollout_every,
            "rollout_endpoint_weight": rollout_endpoint_weight,
            "rollout_clean_weight": rollout_clean_weight,
            "trainable_scope": trainable_scope_override if trainable_scope_override is not None else raw.get("trainable_scope", "all"),
            "fp32_optimizer_state": fp32_optimizer_state,
        }
    )
    training_checkpoint = (
        Path(training_checkpoint_override).resolve()
        if training_checkpoint_override is not None
        else Path(output).resolve() / "training-checkpoint.pt"
    )
    checkpoint_evidence_path: Optional[Path] = None
    policy_digest = _digest({"policy": asdict(policy), "initialization_mode": initialization_mode_override or raw.get("initialization_mode", "inherit")})
    resumed_from_step = 0
    if policy.method == "dmd2" and (
        not dmd2_memory_admitted
        or not wan_dmd2_adapter_ready
    ):
        raise WanStudentError(
            "capability_unavailable",
            "Wan DMD2 requires critic memory admission and Wan adapter evidence",
        )
    requested_teacher_groups = int(math.ceil(float(steps) / float(max(1, teacher_reuse_steps))))
    teacher_groups = max(1, min(requested_teacher_groups, int(raw.get("teacher_target_groups", 4))))
    teacher_target_mode = str(raw.get("teacher_target_mode", "online_h3")).strip()
    teacher_devices = _parse_devices(raw["teacher_devices"])
    student_device = torch.device(str(raw.get("student_device", "cuda:3")))
    teacher_device = torch.device(teacher_devices[0])
    if student_device in {torch.device(item) for item in teacher_devices}:
        raise ValueError("Student device overlaps a Teacher device")
    if student_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real Wan training requires a CUDA Student device")
    wan_teacher_mode = str(raw.get("wan_teacher_mode", "h3_endpoint")).strip()
    if wan_teacher_mode not in {"h3_endpoint", "wan_base_trajectory"}:
        raise ValueError("unsupported wan_teacher_mode: %s" % wan_teacher_mode)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_paths = _training_cache_paths(raw, limit=max(1, teacher_groups))
    started = time.perf_counter()
    teacher_forward_count = 0
    teacher_digest: Optional[str] = None
    teacher_records: list[dict[str, Any]] = []
    teacher_samples: list[tuple[torch.Tensor, dict[str, Any]]] = []
    # Materialize the real H3 RGB targets before loading the Wan pipeline on
    # the Student GPU.  ComfyUI's H3 VAE and Wan's VAE otherwise share the
    # same allocator and can return non-finite RGB frames without an OOM.
    for group_index in range(teacher_groups):
        frames, forward_count, step_teacher_digest, step_teacher_metadata = _teacher_video(
            raw,
            spec,
            teacher_device,
            student_device,
            cache_path=cache_paths[group_index % len(cache_paths)],
        )
        teacher_forward_count += int(forward_count)
        if teacher_digest is None:
            teacher_digest = step_teacher_digest
        elif step_teacher_digest != teacher_digest:
            raise RuntimeError("Teacher checkpoint digest changed during training")
        metadata = dict(step_teacher_metadata)
        teacher_records.append(metadata)
        teacher_samples.append((frames, metadata))

    components = load_wan_training_components(spec, device=str(student_device))
    parent_raw = parent_override if parent_override is not None else raw.get("parent_checkpoint")
    parent_checkpoint: Optional[Path] = None
    parent_checkpoint_digest: Optional[str] = None
    if parent_raw:
        parent_checkpoint = Path(str(parent_raw)).resolve()
        parent_checkpoint_digest = load_wan_transformer_checkpoint(components.pipeline, parent_checkpoint)
    wan_teacher_digest: Optional[str] = None
    if wan_teacher_mode == "wan_base_trajectory":
        # Use the official pretrained Wan Transformer as the frozen 32-step
        # Teacher.  The H3 cache still supplies the clean RGB/Wan-VAE target,
        # but it must not stand in for the Wan denoising trajectory.
        wan_teacher_model = load_wan_trajectory_teacher(spec, device=str(student_device))
        wan_teacher_source = Path(str(spec.pretrained_path or spec.model_id))
        wan_teacher_digest = (
            sha256_path(wan_teacher_source)
            if wan_teacher_source.exists()
            else "model-id:%s" % spec.model_id
        )
    else:
        wan_teacher_model = WanEndpointTeacher()
    initialization_mode = _apply_initialization_mode(
        components.transformer,
        initialization_mode_override if initialization_mode_override is not None else raw.get("initialization_mode", "inherit"),
    )
    trainable_scope = _configure_trainable_scope(
        components.transformer,
        trainable_scope_override if trainable_scope_override is not None else raw.get("trainable_scope"),
    )
    probe_name, initial_probe = _trainable_parameter_probe(components.transformer)
    generator = torch.Generator(device="cpu").manual_seed(int(raw.get("seed", 20261008)) + 1)
    training_batches: list[dict[str, Any]] = []
    for group_index, (frames, metadata) in enumerate(teacher_samples):
        clean_latent = encode_wan_video(components.vae, frames, spec, components.device)
        prompt = str(metadata.get("caption", "")).strip() or "A natural cinematic video with coherent motion."
        prompt_embeds, negative_prompt_embeds = _encode_training_prompt(
            components.pipeline,
            prompt,
            spec,
            components.device,
            components.dtype,
        )
        training_batches.append(
            {
                "sample_id": "wan-%04d" % group_index,
                "frames": frames,
                "prompt": prompt,
                "wan_latent": clean_latent,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
            }
        )
        del clean_latent, prompt_embeds, negative_prompt_embeds
    teacher_samples.clear()
    adapter = WanTrainingAdapter(
        transformer=components.transformer,
        scheduler=components.scheduler,
        encode_frames=lambda value: encode_wan_video(components.vae, value, spec, components.device),
        encode_prompt=lambda value: _encode_training_prompt(
            components.pipeline, value, spec, components.device, components.dtype
        ),
        schedule_provider=_wan_schedule_provider(components.scheduler, source_steps=policy.source_steps),
        model_id=spec.model_id,
        guidance_scale=spec.guidance_scale,
    )
    source_schedule = adapter.schedule_provider(policy.source_steps)
    source_schedule_digest = _digest(source_schedule.to_dict())
    endpoint_teacher = wan_teacher_model
    stage_lineage: list[dict[str, Any]] = []
    total_optimizer_steps = 0
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    maximum_gradient_norm = 0.0
    resume_config_digest: Optional[str] = None
    algorithm_path = "h3_training.algorithms.%s.%s" % (
        "progressive_distillation" if policy.method == "progressive_distillation" else "dmd2",
        "ProgressiveDistillation" if policy.method == "progressive_distillation" else "DMD2",
    )
    if policy.method == "progressive_distillation":
        stages = plan_binary_stages(policy.source_steps, policy.target_steps)
        stage_budgets = [max(1, steps // len(stages)) for _ in stages]
        stage_budgets[-1] += max(0, steps - sum(stage_budgets))
        for stage_index, stage in enumerate(stages):
            stage_method = ProgressiveDistillation(
                components.transformer,
                endpoint_teacher,
                adapter,
                ProgressiveDistillationConfig(
                    stage=stage,
                    learning_rate=policy.learning_rate,
                    trainable_scope=policy.trainable_scope,
                    audio_weight=0.0,
                    endpoint_weight=policy.endpoint_weight,
                    velocity_weight=policy.velocity_weight,
                    clean_anchor_weight=policy.clean_anchor_weight,
                    rollout_steps=policy.rollout_steps,
                    rollout_every=policy.rollout_every,
                    rollout_endpoint_weight=policy.rollout_endpoint_weight,
                    rollout_clean_weight=policy.rollout_clean_weight,
                    fp32_optimizer_state=policy.fp32_optimizer_state,
                ),
            )
            stage_checkpoint = _stage_checkpoint_path(training_checkpoint, stage_index)
            trainer = TrainerEngine(
                TrainerConfig(
                    seed=20261008 + stage_index,
                    parent_sha256=parent_checkpoint_digest or "",
                    device=str(components.device),
                    policy_digest=_digest({"policy": policy_digest, "stage": stage_index, "stage_spec": stage.__dict__}),
                )
            )
            resume_config_digest = trainer.config.digest
            resume_from = stage_checkpoint if stage_checkpoint.is_file() else None
            if resume_from is not None:
                resumed_from_step += int(torch.load(resume_from, map_location="cpu", weights_only=True)["loop_state"]["global_step"])
            result = trainer.run(
                stage_method,
                training_batches,
                max_steps=stage_budgets[stage_index],
                resume_from=resume_from,
                checkpoint_path=stage_checkpoint,
                checkpoint_every_steps=checkpoint_every_steps,
            )
            if checkpoint_every_steps > 0:
                trainer.save_checkpoint(stage_method, result.loop_state, stage_checkpoint)
                checkpoint_evidence_path = stage_checkpoint
            stage_path = output / ("student-stage-%02d.safetensors" % (stage_index + 1))
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            stage_schedule = adapter.schedule_provider(stage.student_nfe)
            stage_schedule_digest = _digest(stage_schedule.to_dict())
            stage_digest = save_wan_checkpoint(
                components.transformer,
                stage_path,
                {
                    "model_id": spec.model_id,
                    "guidance_scale": spec.guidance_scale,
                    "algorithm_name": policy.method,
                    "teacher_nfe": stage.teacher_nfe,
                    "student_nfe": stage.student_nfe,
                    "stage_index": stage_index,
                    "teacher_checkpoint_sha256": teacher_digest or "",
                    "wan_schedule_mode": stage_schedule.mode,
                    "wan_schedule_shift": stage_schedule.shift,
                    "wan_schedule_source_nfe": stage_schedule.source_nfe or stage_schedule.nfe,
                    "wan_schedule_digest": stage_schedule_digest,
                    "offline_simulation": "false",
                },
            )
            optimizer_steps = sum(int(value) for value in result.optimizer_steps.values())
            total_optimizer_steps += optimizer_steps
            initial_loss = result.initial_loss if initial_loss is None else initial_loss
            final_loss = result.final_loss
            maximum_gradient_norm = max(maximum_gradient_norm, result.max_gradient_norm)
            stage_lineage.append(
                {
                    "stage_index": stage_index,
                    "teacher_nfe": stage.teacher_nfe,
                    "student_nfe": stage.student_nfe,
                    "stage_train_steps": stage_budgets[stage_index],
                    "stage_optimizer_steps": optimizer_steps,
                    "stage_child": str(stage_path),
                    "stage_child_sha256": stage_digest,
                    "teacher_checkpoint_sha256": teacher_digest,
                    "teacher_kind": (
                        "wan_base_trajectory"
                        if wan_teacher_mode == "wan_base_trajectory"
                        else ("native_h3_cache_rgb_endpoint" if teacher_target_mode == "cache_rgb" else "h3_rgb_boundary_endpoint")
                    ),
                    "wan_teacher_mode": wan_teacher_mode,
                    "wan_teacher_checkpoint_sha256": wan_teacher_digest or "",
                    "wan_schedule_mode": stage_schedule.mode,
                    "wan_schedule_shift": stage_schedule.shift,
                    "wan_schedule_source_nfe": stage_schedule.source_nfe or stage_schedule.nfe,
                    "wan_schedule_digest": stage_schedule_digest,
                    "loss_trace": [dict(item) for item in result.loss_trace],
                    "loss_weights": {
                        "endpoint": policy.endpoint_weight,
                        "flow": policy.velocity_weight,
                        "clean": policy.clean_anchor_weight,
                        "rollout_endpoint": policy.rollout_endpoint_weight,
                        "rollout_clean": policy.rollout_clean_weight,
                    },
                }
            )
        if stages and checkpoint_every_steps > 0:
            checkpoint_evidence_path = _stage_checkpoint_path(training_checkpoint, len(stages) - 1)
    else:
        try:
            critic = copy.deepcopy(components.transformer)
        except (RuntimeError, MemoryError) as exc:
            raise WanStudentError("capability_unavailable", "Wan DMD2 critic allocation failed: %s" % exc) from exc
        dmd2_method = DMD2(
            components.transformer,
            endpoint_teacher,
            critic,
            adapter,
            DMD2Config(
                student_learning_rate=policy.learning_rate,
                critic_learning_rate=policy.critic_learning_rate,
                data_mode="real_latent",
                regression_weight=policy.clean_anchor_weight,
                trainable_scope=policy.trainable_scope,
                video_weight=1.0,
                audio_weight=0.0,
                generator_update_interval=2,
            ),
        )
        trainer = TrainerEngine(
            TrainerConfig(
                seed=20261008,
                parent_sha256=parent_checkpoint_digest or "",
                device=str(components.device),
                policy_digest=policy_digest,
            )
        )
        resume_config_digest = trainer.config.digest
        resume_from = training_checkpoint if training_checkpoint.is_file() else None
        if resume_from is not None:
            resumed_from_step = int(torch.load(resume_from, map_location="cpu", weights_only=True)["loop_state"]["global_step"])
        result = trainer.run(
            dmd2_method,
            training_batches,
            max_steps=steps,
            resume_from=resume_from,
            checkpoint_path=training_checkpoint,
            checkpoint_every_steps=checkpoint_every_steps,
        )
        if checkpoint_every_steps > 0:
            trainer.save_checkpoint(dmd2_method, result.loop_state, training_checkpoint)
            checkpoint_evidence_path = training_checkpoint
        total_optimizer_steps = sum(int(value) for value in result.optimizer_steps.values())
        initial_loss = result.initial_loss
        final_loss = result.final_loss
        maximum_gradient_norm = result.max_gradient_norm
        stage_lineage.append(
            {
                "algorithm": "dmd2",
                "critic_parameter_count": sum(int(parameter.numel()) for parameter in critic.parameters()),
                "critic_updates": dmd2_method.critic_updates,
                "student_updates": dmd2_method.student_updates,
                "ema_updates": dmd2_method.ema.num_updates,
                "sampler_samples_drawn": dmd2_method.sampler.samples_drawn,
            }
        )
    final_probe = next(
        parameter for name, parameter in components.transformer.named_parameters() if name == probe_name
    ).detach().float()
    changed = not torch.equal(initial_probe.cpu(), final_probe.cpu())
    if not changed:
        raise RuntimeError("Wan Transformer parameters did not change")
    if teacher_digest is None:
        raise RuntimeError("training completed without a real Teacher target")
    parent_source = Path(str(spec.pretrained_path or spec.model_id))
    lineage_root_digest = sha256_path(parent_source) if parent_source.exists() else "model-id:%s" % spec.model_id
    parent_digest = parent_checkpoint_digest or lineage_root_digest
    checkpoint_path = output / "student.safetensors"
    checkpoint_digest = save_wan_checkpoint(
        components.transformer,
        checkpoint_path,
        {
            "model_id": spec.model_id,
            "lineage_root": spec.lineage_root,
            "pretrained": "true",
            "parent_model_digest": parent_digest,
            "lineage_root_digest": lineage_root_digest,
            "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else "",
            "parent_checkpoint_sha256": parent_checkpoint_digest or "",
            "teacher_checkpoint_sha256": teacher_digest,
            "wan_teacher_mode": wan_teacher_mode,
            "wan_teacher_checkpoint_sha256": wan_teacher_digest or "",
            "teacher_forward_count": teacher_forward_count,
            "teacher_target_mode": teacher_target_mode,
            "offline_simulation": "false",
            "frame_contract": "81@16fps@256x256",
            "train_steps": steps,
            "teacher_reuse_steps": teacher_reuse_steps,
            "learning_rate": learning_rate,
            "trainable_scope": trainable_scope,
            "algorithm_name": policy.method,
            "algorithm_path": algorithm_path,
            "source_steps": policy.source_steps,
            "target_steps": policy.target_steps,
            "rollout_steps": policy.rollout_steps,
            "rollout_every": policy.rollout_every,
            "rollout_endpoint_weight": policy.rollout_endpoint_weight,
            "rollout_clean_weight": policy.rollout_clean_weight,
            "velocity_weight": policy.velocity_weight,
            "clean_anchor_weight": policy.clean_anchor_weight,
            "fp32_optimizer_state": policy.fp32_optimizer_state,
            "loss_reduction": "fp32",
            "initialization_mode": initialization_mode,
            "stage_lineage": json.dumps(stage_lineage, sort_keys=True),
            "training_checkpoint": str(training_checkpoint),
            "training_checkpoint_evidence": str(checkpoint_evidence_path) if checkpoint_evidence_path else "",
            "checkpoint_every_steps": checkpoint_every_steps,
            "policy_digest": policy_digest,
            "resumed_from_step": resumed_from_step,
            "resume_config_digest": resume_config_digest or "",
            "dmd2_memory_admitted": dmd2_memory_admitted,
            "wan_dmd2_adapter_ready": wan_dmd2_adapter_ready,
            "wan_schedule_mode": source_schedule.mode,
            "wan_schedule_shift": source_schedule.shift,
            "wan_schedule_source_nfe": source_schedule.nfe,
            "wan_schedule_digest": source_schedule_digest,
        },
    )
    result = {
        "status": "success",
        "model_id": spec.model_id,
        "guidance_scale": spec.guidance_scale,
        "pretrained": True,
        "lineage_root": spec.lineage_root,
        "parent_model_digest": parent_digest,
        "lineage_root_digest": lineage_root_digest,
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "parent_checkpoint_sha256": parent_checkpoint_digest,
        "teacher_checkpoint_sha256": teacher_digest,
        "wan_teacher_mode": wan_teacher_mode,
        "wan_teacher_checkpoint_sha256": wan_teacher_digest or "",
        "student_checkpoint_sha256": checkpoint_digest,
        "student_checkpoint": str(checkpoint_path),
        "optimizer_steps": total_optimizer_steps,
        "algorithm_name": policy.method,
        "algorithm_path": algorithm_path,
        "teacher_reuse_steps": teacher_reuse_steps,
        "teacher_target_groups": teacher_groups,
        "learning_rate": learning_rate,
        "trainable_scope": trainable_scope,
        "rollout_policy": {
            "rollout_steps": policy.rollout_steps,
            "rollout_every": policy.rollout_every,
            "rollout_endpoint_weight": policy.rollout_endpoint_weight,
            "rollout_clean_weight": policy.rollout_clean_weight,
        },
        "loss_weights": {
            "endpoint": policy.endpoint_weight,
            "flow": policy.velocity_weight,
            "clean": policy.clean_anchor_weight,
            "rollout_endpoint": policy.rollout_endpoint_weight,
            "rollout_clean": policy.rollout_clean_weight,
        },
        "loss_reduction": "fp32",
        "fp32_optimizer_state": policy.fp32_optimizer_state,
        "initialization_mode": initialization_mode,
        "stage_lineage": stage_lineage,
        "teacher_forward_count": teacher_forward_count,
        "teacher_target_mode": teacher_target_mode,
        "offline_simulation": False,
        "frame_contract": {
            "frames": spec.num_frames,
            "fps": spec.native_fps,
            "duration_s": spec.duration_s,
            "width": spec.width,
            "height": spec.height,
        },
        "loss": {"total_final": final_loss, "initial": initial_loss},
        "teacher": (
            teacher_records[0]
            if len(teacher_records) == 1
            else {
                "teacher_checkpoint_sha256": teacher_digest,
                "teacher_forward_count": teacher_forward_count,
                "cache_items": teacher_records,
            }
        ),
        "training_cache_items": [str(path) for path in cache_paths],
        "wall_time_s": float(time.perf_counter() - started),
        "training_checkpoint": str(training_checkpoint),
        "training_checkpoint_evidence": str(checkpoint_evidence_path) if checkpoint_evidence_path else "",
        "training_checkpoint_sha256": sha256_path(checkpoint_evidence_path) if checkpoint_evidence_path else "",
        "checkpoint_every_steps": checkpoint_every_steps,
        "policy_digest": policy_digest,
        "resumed_from_step": resumed_from_step,
        "resume_config_digest": resume_config_digest or "",
        "dmd2_memory_admitted": dmd2_memory_admitted,
        "wan_dmd2_adapter_ready": wan_dmd2_adapter_ready,
        "wan_schedule": source_schedule.to_dict(),
        "wan_schedule_digest": source_schedule_digest,
        "loss_trace": (
            list(stage_lineage[-1].get("loss_trace", []))
            if stage_lineage and policy.method == "progressive_distillation"
            else []
        ),
    }
    _write(result_path, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--teacher-reuse-steps", type=int, default=None)
    parser.add_argument("--parent-checkpoint", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--trainable-scope", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--source-steps", type=int, default=None)
    parser.add_argument("--target-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--rollout-every", type=int, default=None)
    parser.add_argument("--rollout-endpoint-weight", type=float, default=None)
    parser.add_argument("--rollout-clean-weight", type=float, default=None)
    parser.add_argument("--endpoint-weight", type=float, default=None)
    parser.add_argument("--velocity-weight", type=float, default=None)
    parser.add_argument("--clean-anchor-weight", type=float, default=None)
    parser.add_argument("--fp32-optimizer-state", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--initialization-mode", default=None)
    parser.add_argument("--training-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--dmd2-memory-admitted", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wan-dmd2-adapter-ready", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = _load_mapping(args.config)
        if args.dry_run:
            spec = _training_spec(raw, Path(str(raw["wan_config"])))
            print(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        result = run(
            raw,
            output=args.output,
            result_path=args.result,
            steps_override=args.steps,
            reuse_override=args.teacher_reuse_steps,
            parent_override=args.parent_checkpoint,
            learning_rate_override=args.learning_rate,
            trainable_scope_override=args.trainable_scope,
            method_override=args.method,
            source_steps_override=args.source_steps,
            target_steps_override=args.target_steps,
            rollout_steps_override=args.rollout_steps,
            rollout_every_override=args.rollout_every,
            rollout_endpoint_weight_override=args.rollout_endpoint_weight,
            rollout_clean_weight_override=args.rollout_clean_weight,
            endpoint_weight_override=args.endpoint_weight,
            velocity_weight_override=args.velocity_weight,
            clean_anchor_weight_override=args.clean_anchor_weight,
            fp32_optimizer_state_override=args.fp32_optimizer_state,
            initialization_mode_override=args.initialization_mode,
            training_checkpoint_override=args.training_checkpoint,
            checkpoint_every_steps_override=args.checkpoint_every_steps,
            dmd2_memory_admitted_override=args.dmd2_memory_admitted,
            wan_dmd2_adapter_ready_override=args.wan_dmd2_adapter_ready,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, WanStudentError) as exc:
        failure = {
            "status": "failed",
            "offline_simulation": False,
            "error_code": getattr(exc, "code", "wan_training_failed"),
            "message": str(exc),
        }
        try:
            _write(args.result, failure)
        except OSError:
            pass
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
