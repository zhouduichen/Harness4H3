"""Measured evaluation for TinyH3 checkpoints."""

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from harness4h3.evaluator.evaluator import EvaluationResult
from harness4h3.h3.state import ModelState
from harness4h3.target.profile import TargetProfile
from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.base import ModelRole
from h3_training.data.dataset import SyntheticH3Dataset, deterministic_batches
from h3_training.data.schema import ModalLatents, ModalPrediction
from h3_training.tiny.factory import load_tiny_checkpoint


class TinyCheckpointEvaluator:
    def __init__(self, dataset_size: int = 8, base_seed: int = 20_000, baseline_quality: Optional[float] = None) -> None:
        self.dataset_size = dataset_size
        self.base_seed = base_seed
        self.baseline_quality = baseline_quality

    def evaluate_checkpoint(self, path: Path, target: Optional[TargetProfile] = None) -> EvaluationResult:
        model, metadata = load_tiny_checkpoint(path)
        model.eval()
        adapter = TinyH3Adapter()
        role = ModelRole("student", model, adapter, False, metadata)
        dataset = SyntheticH3Dataset(self.dataset_size, self.base_seed, model.config)
        generator = torch.Generator().manual_seed(self.base_seed + 999)
        squared_error = 0.0
        element_count = 0
        elapsed = 0.0
        with torch.no_grad():
            for raw in deterministic_batches(dataset, batch_size=2):
                batch = adapter.prepare_batch(raw, generator)
                noisy = adapter.add_noise(batch.latents, batch.noise, batch.timesteps)
                started = time.perf_counter()
                prediction = adapter.predict(role, noisy, batch.timesteps, batch.conditioning)
                elapsed += time.perf_counter() - started
                target_velocity = ModalPrediction(
                    video=batch.latents.video - batch.noise.video,
                    audio=batch.latents.audio - batch.noise.audio,
                )
                for actual, expected in ((prediction.video, target_velocity.video), (prediction.audio, target_velocity.audio)):
                    squared_error += torch.sum((actual - expected) ** 2).item()
                    element_count += actual.numel()
        mse = squared_error / element_count
        quality = 1.0 / (1.0 + mse)
        latency = elapsed / self.dataset_size
        size_gb = Path(path).stat().st_size / (1024 ** 3)
        metrics: Dict[str, Any] = {
            "quality_score": quality,
            "heldout_mse": mse,
            "latency_s": latency,
            "model_size_gb": size_gb,
            "checkpoint_bytes": Path(path).stat().st_size,
            "sampling_steps": metadata["sampling_nfe"],
            "measured": True,
        }
        failures = []
        if target is not None:
            if target.max_model_size_gb is not None and size_gb > target.max_model_size_gb:
                failures.append("model_size")
            if target.max_latency_s is not None and latency > target.max_latency_s:
                failures.append("latency")
            if target.min_quality_score is not None and quality < target.min_quality_score:
                failures.append("quality")
            baseline = self.baseline_quality
            if target.max_quality_drop is not None and baseline is not None and baseline - quality > target.max_quality_drop:
                failures.append("quality_drop")
        return EvaluationResult(
            score=quality if not failures else 0.0,
            metrics=metrics,
            critical_regression=bool(failures),
            failure_type=failures[0] if failures else None,
        )

    def evaluate(self, state: ModelState, target: TargetProfile) -> Tuple[float, Mapping[str, Any]]:
        result = self.evaluate_checkpoint(Path(state.checkpoint_path), target)
        return result.score, result.metrics
