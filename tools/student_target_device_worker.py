#!/usr/bin/env python3
"""Run a real Student artifact through the configured target runtime.

This worker is intentionally an execution-layer adapter.  It does not alter
the campaign controller or acceptance semantics: it exports the checkpoint,
verifies/creates an int8 target artifact, packages that artifact as the
trusted runtime input, runs the real Student sampler plus H3 VAE on the
explicit target CUDA device, and emits artifact-bound EdgeEvidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from harness4h3.student.edge import EdgeEvidence
from harness4h3.student.inference import (
    decode_video_latent,
    load_h3_cache_item,
    sample_student_latent,
    write_video,
)
from harness4h3.student.proposal import StudentProposal, StudentTarget
from harness4h3.student.quantization import load_student_state, quantize_checkpoint
from harness4h3.student.runtime_quantization import load_runtime_quantized_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _copy_artifact(source: Path, destination: Path) -> Path:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError("target artifact source does not exist: %s" % source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise ValueError("target artifact is empty: %s" % destination)
    return destination


def _gpu_index(device: str) -> int:
    value = str(device).strip()
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    return int(value)


def _gpu_sample(gpu_index: int) -> tuple[float, float]:
    command = [
        "nvidia-smi",
        "-i",
        str(int(gpu_index)),
        "--query-gpu=power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    line = next((item.strip() for item in completed.stdout.splitlines() if item.strip()), "")
    fields = [item.strip() for item in line.split(",")]
    if len(fields) != 2:
        raise RuntimeError("nvidia-smi returned malformed power/temperature data: %r" % completed.stdout)
    return float(fields[0]), float(fields[1])


class _GpuSampler:
    def __init__(self, gpu_index: int, interval_s: float):
        self.gpu_index = int(gpu_index)
        self.interval_s = max(0.05, float(interval_s))
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                power, thermal = _gpu_sample(self.gpu_index)
                self.samples.append((time.monotonic(), power, thermal))
            except BaseException as exc:  # surface the measurement failure after the run
                self.error = exc
                self._stop.set()
                return
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, name="target-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> Mapping[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15.0)
        try:
            power, thermal = _gpu_sample(self.gpu_index)
            self.samples.append((time.monotonic(), power, thermal))
        except BaseException as exc:
            if self.error is None:
                self.error = exc
        if self.error is not None:
            raise RuntimeError("target GPU measurement failed: %s" % self.error) from self.error
        if not self.samples:
            raise RuntimeError("target GPU measurement produced no samples")
        energy_j = 0.0
        for previous, current in zip(self.samples, self.samples[1:]):
            elapsed = max(0.0, current[0] - previous[0])
            energy_j += ((previous[1] + current[1]) / 2.0) * elapsed
        return {
            "energy_j": float(energy_j),
            "thermal_c": float(max(item[2] for item in self.samples)),
            "power_samples": [
                {"time_monotonic": float(stamp), "power_w": float(power), "thermal_c": float(thermal)}
                for stamp, power, thermal in self.samples
            ],
        }


def _build_target(args: argparse.Namespace) -> StudentTarget:
    return StudentTarget(
        latent_frames=int(args.latent_frames),
        latent_height=int(args.latent_height),
        latent_width=int(args.latent_width),
    )


def _steady_state_generation(
    proposal: StudentProposal,
    model: torch.nn.Module,
    prompt: torch.Tensor,
    target: StudentTarget,
    comfyui_root: Path,
    vae_name: str,
    output_path: Path,
    device: torch.device,
    *,
    seed: int,
) -> Mapping[str, Any]:
    """Run one generation with the deployed Student and VAE already resident.

    ``generate_video`` deliberately owns model loading for the server
    evaluator.  A target runtime is different: deployment loads the Student
    once and keeps both the Student and VAE resident between invocations.  The
    measured interval must therefore begin at Student sampling and end after
    VAE decode/video materialization, so the power and latency boundaries are
    the same and exclude runtime loading.
    """

    started = time.perf_counter()
    latent = sample_student_latent(model, prompt, proposal, target, device, seed=seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sampling_latency = time.perf_counter() - started

    decode_started = time.perf_counter()
    frames = decode_video_latent(comfyui_root, vae_name, latent)
    write_video(frames, output_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_latency = time.perf_counter() - decode_started
    peak_memory_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )
    return {
        "video_path": str(Path(output_path).resolve()),
        "latency_s": float(sampling_latency + decode_latency),
        "sampling_latency_s": float(sampling_latency),
        "decode_latency_s": float(decode_latency),
        "generation_latency_s": float(sampling_latency + decode_latency),
        "peak_memory_gb": peak_memory_gb,
        "frame_count": int(frames.shape[0]),
        "resolution": [int(frames.shape[2]), int(frames.shape[1])],
        "checkpoint_bytes": 0,
        "device": str(device),
    }


def _run(args: argparse.Namespace) -> int:
    result_path = Path(args.result).resolve()
    output_dir = Path(args.output).resolve()
    started = time.monotonic()
    try:
        checkpoint = Path(args.checkpoint).resolve()
        proposal = StudentProposal.from_dict(json.loads(Path(args.proposal).read_text(encoding="utf-8")))
        target = _build_target(args)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = _copy_artifact(checkpoint, output_dir / "export" / "student.safetensors")
        exported_sha = _sha256(exported)

        quantized = output_dir / "quantize" / "student-int8.safetensors"
        _state, source_metadata = load_student_state(exported)
        if str(source_metadata.get("quantization", "none")) == "int8":
            _copy_artifact(exported, quantized)
            quantization_mode = "verified_existing_int8"
        else:
            quantize_checkpoint(
                exported,
                quantized,
                bits=8,
                metadata={"source_sha256": exported_sha, "target_device_id": args.target_device_id},
            )
            quantization_mode = "runtime_int8_quantization"
        quantized_sha = _sha256(quantized)

        # The trusted runtime consumes safetensors directly.  Compilation is
        # therefore a validated runtime package, not a claimed opaque binary.
        _state, compiled_metadata = load_student_state(quantized)
        if str(compiled_metadata.get("quantization", "")) != "int8":
            raise ValueError("compiled target artifact is not tagged int8")
        compiled = _copy_artifact(quantized, output_dir / "compile" / "student.runtime.safetensors")
        compiled_sha = _sha256(compiled)
        _write_json(
            output_dir / "compile" / "runtime-manifest.json",
            {
                "runtime_backend": "trusted-runtime-v1",
                "target_device_id": args.target_device_id,
                "source_sha256": exported_sha,
                "quantized_sha256": quantized_sha,
                "compiled_sha256": compiled_sha,
                "quantization": "int8",
            },
        )
        deployment_id = "%s:%s" % (args.target_device_id, compiled_sha[:16])
        _write_json(
            output_dir / "deploy" / "deployment.json",
            {"deployment_id": deployment_id, "artifact": str(compiled), "artifact_sha256": compiled_sha},
        )

        benchmark_dir = output_dir / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        selected_device = str(args.device)
        gpu_index = _gpu_index(selected_device)
        runtime_device = torch.device(selected_device)
        torch.cuda.set_device(runtime_device)
        dtype = torch.bfloat16 if proposal.deployment.precision == "bf16" else torch.float16
        prompt = load_h3_cache_item(
            Path(args.cache_dir), target, runtime_device, dtype
        )["prompt"]
        model = load_runtime_quantized_model(proposal, compiled, target, runtime_device)

        # A deployed target runtime keeps the Student and VAE resident after
        # deployment. Warm once outside the measured interval so target
        # latency/energy describe steady-state generation; the measured run
        # still includes the real Student sampler and VAE decode.
        warmup_generation = _steady_state_generation(
            proposal,
            model,
            prompt,
            target,
            Path(args.comfyui_root),
            args.vae_name,
            benchmark_dir / "warmup-generation.mp4",
            runtime_device,
            seed=int(args.seed),
        )
        torch.cuda.synchronize(runtime_device)
        torch.cuda.reset_peak_memory_stats(runtime_device)
        sampler = _GpuSampler(gpu_index, args.measurement_interval_s)
        sampler.start()
        generation = _steady_state_generation(
            proposal,
            model,
            prompt,
            target,
            Path(args.comfyui_root),
            args.vae_name,
            benchmark_dir / "student-generation.mp4",
            runtime_device,
            seed=int(args.seed),
        )
        generation["checkpoint_bytes"] = int(compiled.stat().st_size)
        gpu_measurement = sampler.stop()
        peak_memory_gb = float(generation.get("peak_memory_gb", 0.0))
        if peak_memory_gb <= 0:
            peak_memory_gb = float(torch.cuda.max_memory_allocated(runtime_device) / (1024**3))
        benchmark = {
            "deployment_id": deployment_id,
            "artifact_sha256": compiled_sha,
            "target_device_id": args.target_device_id,
            "runtime_backend": "trusted-runtime-v1",
            "device": selected_device,
            "latency_s": float(generation["generation_latency_s"]),
            "memory_gb": peak_memory_gb,
            "energy_j": float(gpu_measurement["energy_j"]),
            "thermal_c": float(gpu_measurement["thermal_c"]),
            "model_size_gb": float(compiled.stat().st_size / (1024**3)),
            "precision": proposal.deployment.precision,
            "quantization": "int8",
            "resolution": list(generation["resolution"]),
            "frames": int(target.latent_frames),
            "sampling_steps": int(proposal.training.target_steps),
            "generation": dict(generation),
            "warmup_generation": dict(warmup_generation),
            "measurement_mode": "steady_state_resident_student_and_vae",
            "latency_boundary": "student_sampling_plus_vae_decode_and_video_write",
            "energy_boundary": "student_sampling_plus_vae_decode_and_video_write",
            "runtime_resident": True,
            "quantization_mode": "runtime_int8_weight_only",
            "artifact_quantization_mode": quantization_mode,
            "gpu_measurement": gpu_measurement,
        }
        benchmark_path = benchmark_dir / "benchmark.json"
        _write_json(benchmark_path, benchmark)
        reference = benchmark_dir / "edge-evidence.json"
        _write_json(reference, {"deployment_id": deployment_id, "artifact_sha256": compiled_sha, "benchmark": benchmark})

        metadata = {
            "runtime_backend": "trusted-runtime-v1",
            "precision": proposal.deployment.precision,
            "quantization": "int8",
            "resolution": list(generation["resolution"]),
            "frames": int(target.latent_frames),
            "sampling_steps": int(proposal.training.target_steps),
        }
        values = {
            "edge_exported": 1.0,
            "edge_quantized": 1.0,
            "edge_runtime": 1.0,
            "edge_device": 1.0,
            "edge_latency": benchmark["latency_s"],
            "edge_memory": benchmark["memory_gb"],
            "edge_energy": benchmark["energy_j"],
            "edge_thermal": benchmark["thermal_c"],
            "edge_model_size": benchmark["model_size_gb"],
        }
        evidence = [
            EdgeEvidence(name, float(value), compiled_sha, args.target_device_id, str(reference), metadata=metadata).to_dict()
            for name, value in values.items()
        ]
        _write_json(
            result_path,
            {
                "status": "success",
                "deployment_id": deployment_id,
                "artifact_sha256": compiled_sha,
                "benchmark": benchmark,
                "evidence": evidence,
                "offline_simulation": False,
                "wall_time_s": time.monotonic() - started,
            },
        )
        return 0
    except BaseException as exc:
        _write_json(
            result_path,
            {
                "status": "failed",
                "failure_code": "target_device_execution_failed",
                "message": str(exc),
                "offline_simulation": False,
                "wall_time_s": time.monotonic() - started,
            },
        )
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--target-device-id", required=True)
    parser.add_argument("--comfyui-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--vae-name", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--measurement-interval-s", type=float, default=0.1)
    parser.add_argument("--latent-frames", type=int, default=5)
    parser.add_argument("--latent-height", type=int, default=16)
    parser.add_argument("--latent-width", type=int, default=16)
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
