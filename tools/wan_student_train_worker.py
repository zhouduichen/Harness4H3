#!/usr/bin/env python3
"""Run one real Wan2.1-T2V-1.3B 5-second H3-distillation experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness4h3.student.inference import decode_video_latent, load_h3_cache_item
from harness4h3.student.proposal import StudentTarget
from harness4h3.student.schedule import h3_video_sigmas
from harness4h3.student.teacher_service import TeacherService
from harness4h3.student.wan import WanStudentError, load_wan_spec
from harness4h3.student.wan_training import (
    WanTrainingSpec,
    encode_wan_video,
    flow_match_batch,
    load_wan_training_components,
    resample_video_frames,
    save_wan_checkpoint,
    sha256_path,
    wan_velocity_loss,
)
from h3_training.data.schema import Conditioning, ModalLatents, ModalTimesteps


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def _training_cache_paths(raw: Mapping[str, Any], *, limit: int) -> list[Path]:
    """Select one real H3 cache item per caption for prompt-diverse training."""

    cache_dir = Path(str(raw["h3_cache_dir"])).resolve()
    paths = sorted(cache_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError("no H3 cache items under %s" % cache_dir)
    selected: list[Path] = []
    captions: set[str] = set()
    for path in paths:
        item = torch.load(path, map_location="cpu", weights_only=False)
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


def _encode_training_prompt(pipeline: Any, prompt: str, spec: WanTrainingSpec, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode frozen text without creating inference-mode tensors.

    The returned embeddings are consumed by the trainable Wan Transformer.
    ``inference_mode`` tensors cannot be saved for backward by autograd even
    when they are detached, so use ``no_grad`` and clone into normal tensors.
    """

    with torch.no_grad():
        prompt_embeds, _ = pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=spec.max_sequence_length,
            device=device,
            dtype=dtype,
        )
    return prompt_embeds.detach().clone()


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
    cache_item = load_h3_cache_item(cache_dir, target, teacher_device, torch.bfloat16, cache_path=cache_path)
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
        conditioning = Conditioning(prompt.to(device=teacher_device, dtype=torch.bfloat16))
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
        forward_count = int(service.forward_count)
        if forward_count <= 0:
            raise RuntimeError("real H3 Teacher completed without a forward")
        teacher_digest = sha256_path(teacher_checkpoint)
    finally:
        forward_count = int(service.forward_count)
        service.close()
    frames = decode_video_latent(comfyui_root, spec.vae_name, video.to(device=decode_device, dtype=torch.bfloat16))
    _release_h3_vae()
    frames = resample_video_frames(frames, num_frames=spec.num_frames, height=spec.height, width=spec.width)
    metadata = {
        "teacher_sampling_steps": int(spec.h3_sampling_steps),
        "teacher_forward_count": int(forward_count),
        "teacher_wall_time_s": float(time.perf_counter() - started),
        "teacher_checkpoint_sha256": teacher_digest,
        "caption": str(cache_item.get("caption", "")).strip(),
        "cache_item": str(cache_item["path"]),
    }
    return frames, forward_count, teacher_digest, metadata


def run(raw: Mapping[str, Any], *, output: Path, result_path: Path, steps_override: int | None = None) -> Mapping[str, Any]:
    wan_config = Path(str(raw["wan_config"])).resolve()
    spec = _training_spec(raw, wan_config)
    steps = int(steps_override if steps_override is not None else raw.get("train_steps", spec.train_steps))
    if steps <= 0:
        raise ValueError("train_steps must be positive")
    teacher_devices = _parse_devices(raw["teacher_devices"])
    student_device = torch.device(str(raw.get("student_device", "cuda:3")))
    teacher_device = torch.device(teacher_devices[0])
    if student_device in {torch.device(item) for item in teacher_devices}:
        raise ValueError("Student device overlaps a Teacher device")
    if student_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real Wan training requires a CUDA Student device")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_paths = _training_cache_paths(raw, limit=max(1, steps))
    frames, teacher_forward_count, teacher_digest, teacher_metadata = _teacher_video(
        raw, spec, teacher_device, student_device, cache_path=cache_paths[0]
    )
    teacher_records = [dict(teacher_metadata)]
    components = load_wan_training_components(spec, device=str(student_device))
    optimizer = torch.optim.AdamW(components.transformer.parameters(), lr=float(spec.learning_rate))
    initial_probe = next(components.transformer.parameters()).detach().float().clone()
    generator = torch.Generator(device=str(components.device)).manual_seed(int(raw.get("seed", 20261008)) + 1)
    losses = []
    standard_losses = []
    distill_losses = []
    started = time.perf_counter()
    for step in range(steps):
        if step > 0:
            frames, forward_count, step_teacher_digest, step_teacher_metadata = _teacher_video(
                raw,
                spec,
                teacher_device,
                student_device,
                cache_path=cache_paths[step % len(cache_paths)],
            )
            teacher_forward_count += int(forward_count)
            if step_teacher_digest != teacher_digest:
                raise RuntimeError("Teacher checkpoint digest changed during training")
            teacher_records.append(dict(step_teacher_metadata))
        clean_latent = encode_wan_video(components.vae, frames, spec, components.device)
        prompt = str(teacher_records[-1].get("caption", "")).strip() or "A natural cinematic video with coherent motion."
        prompt_embeds = _encode_training_prompt(
            components.pipeline,
            prompt,
            spec,
            components.device,
            components.dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        noisy, timestep, target, sigma = flow_match_batch(
            clean_latent,
            scheduler=components.scheduler,
            generator=generator,
            device=components.device,
        )
        prediction = components.transformer(
            hidden_states=noisy,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        total, standard, distill = wan_velocity_loss(
            prediction,
            target,
            noisy,
            clean_latent,
            sigma,
            standard_weight=spec.standard_loss_weight,
            distill_weight=spec.distill_loss_weight,
        )
        if not torch.isfinite(total):
            raise RuntimeError("Wan training loss is non-finite")
        total.backward()
        torch.nn.utils.clip_grad_norm_(components.transformer.parameters(), 1.0)
        optimizer.step()
        losses.append(float(total.detach().cpu()))
        standard_losses.append(float(standard.cpu()))
        distill_losses.append(float(distill.cpu()))
        del clean_latent, prompt_embeds, frames
    final_probe = next(components.transformer.parameters()).detach().float()
    changed = not torch.equal(initial_probe.cpu(), final_probe.cpu())
    if not changed:
        raise RuntimeError("Wan Transformer parameters did not change")
    parent_source = Path(str(spec.pretrained_path or spec.model_id))
    parent_digest = sha256_path(parent_source) if parent_source.exists() else "model-id:%s" % spec.model_id
    checkpoint_path = output / "student.safetensors"
    checkpoint_digest = save_wan_checkpoint(
        components.transformer,
        checkpoint_path,
        {
            "model_id": spec.model_id,
            "lineage_root": spec.lineage_root,
            "pretrained": "true",
            "parent_model_digest": parent_digest,
            "teacher_checkpoint_sha256": teacher_digest,
            "teacher_forward_count": teacher_forward_count,
            "offline_simulation": "false",
            "frame_contract": "81@16fps@256x256",
            "train_steps": steps,
        },
    )
    result = {
        "status": "success",
        "model_id": spec.model_id,
        "pretrained": True,
        "lineage_root": spec.lineage_root,
        "parent_model_digest": parent_digest,
        "teacher_checkpoint_sha256": teacher_digest,
        "student_checkpoint_sha256": checkpoint_digest,
        "student_checkpoint": str(checkpoint_path),
        "optimizer_steps": steps,
        "teacher_forward_count": teacher_forward_count,
        "offline_simulation": False,
        "frame_contract": {
            "frames": spec.num_frames,
            "fps": spec.native_fps,
            "duration_s": spec.duration_s,
            "width": spec.width,
            "height": spec.height,
        },
        "loss": {
            "total_final": losses[-1],
            "standard_final": standard_losses[-1],
            "distill_final": distill_losses[-1],
            "total_history": losses,
        },
        "teacher": (
            teacher_metadata
            if len(teacher_records) == 1
            else {
                "teacher_checkpoint_sha256": teacher_digest,
                "teacher_forward_count": teacher_forward_count,
                "cache_items": teacher_records,
            }
        ),
        "training_cache_items": [str(path) for path in cache_paths],
        "wall_time_s": float(time.perf_counter() - started),
    }
    _write(result_path, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = _load_mapping(args.config)
        if args.dry_run:
            spec = _training_spec(raw, Path(str(raw["wan_config"])))
            print(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        result = run(raw, output=args.output, result_path=args.result, steps_override=args.steps)
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
