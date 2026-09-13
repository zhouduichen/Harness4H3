"""Creation and loading of TinyH3 model-only checkpoints."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .model import TinyH3Config, TinyH3Model


CHECKPOINT_SCHEMA = 1


def create_tiny_checkpoint(
    path: Path,
    model_id: str = "M0000",
    parent_id: Optional[str] = None,
    sampling_nfe: int = 4,
    seed: int = 0,
    config: TinyH3Config = TinyH3Config(),
    provenance: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = TinyH3Model(config)
    finally:
        torch.random.set_rng_state(prior)
    payload = tiny_checkpoint_payload(model, model_id, parent_id, sampling_nfe, provenance)
    torch.save(payload, path)
    return path


def tiny_checkpoint_payload(
    model: TinyH3Model,
    model_id: str,
    parent_id: Optional[str],
    sampling_nfe: int,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if sampling_nfe <= 0:
        raise ValueError("sampling_nfe must be positive")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "architecture_name": "TinyH3",
        "config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "model_id": model_id,
        "parent_id": parent_id,
        "sampling_nfe": sampling_nfe,
        "provenance": dict(provenance or {"kind": "tiny_reference", "offline_simulation": False}),
    }


def load_tiny_checkpoint(path: Path) -> Tuple[TinyH3Model, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA or payload.get("architecture_name") != "TinyH3":
        raise ValueError("not a supported TinyH3 checkpoint")
    model = TinyH3Model(TinyH3Config(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    metadata = {key: value for key, value in payload.items() if key != "model_state"}
    return model, metadata
