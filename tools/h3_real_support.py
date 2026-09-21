"""Small, dependency-light helpers shared by the real MiniMax-H3 worker."""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Mapping

import torch

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no Linux reflink ioctl.
    fcntl = None


TRAINABLE_TENSOR_NAMES = frozenset(
    {
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)

VIDEO_LATENT_FRAMES = 5
AUDIO_LATENT_FRAMES = 37
VIDEO_ROWS = VIDEO_LATENT_FRAMES * 64
VIDEO_ROW_WIDTH = 96
AUDIO_ROWS = AUDIO_LATENT_FRAMES * 2
AUDIO_ROW_WIDTH = 32
TEXT_LENGTH = 8
TEXT_WIDTH = 5120


def make_smoke_cache(output_dir: Path, seed: int, text_len: int = TEXT_LENGTH) -> Path:
    """Write one deterministic 256x256x22 H3 cache item and its manifest."""

    if text_len <= 0:
        raise ValueError("text_len must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    item = {
        "video": torch.randn((VIDEO_ROWS, VIDEO_ROW_WIDTH), generator=generator, dtype=torch.float32),
        "audio": torch.randn((AUDIO_ROWS, AUDIO_ROW_WIDTH), generator=generator, dtype=torch.float32),
        "prompt": torch.randn((text_len, TEXT_WIDTH), generator=generator, dtype=torch.float32).to(torch.bfloat16),
        "caption": "A deterministic MiniMax-H3 training smoke sample.",
        "height": 256,
        "width": 256,
        "latent_frames": VIDEO_LATENT_FRAMES,
        "audio_frames": AUDIO_LATENT_FRAMES,
        "num_frames": 22,
        "seed": int(seed),
    }
    cache_path = output_dir / "00000000.pt"
    torch.save(item, cache_path)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "a1-t0-smoke-00000000",
                "task": "fl2va",
                "caption": item["caption"],
                "target_video": None,
                "target_audio": None,
                "cache": str(cache_path),
                "synthetic_latent": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Return the safetensors header and the number of bytes before payloads."""

    with Path(path).open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("safetensors file has no complete header length")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > 256 * 1024 * 1024:
            raise ValueError("invalid safetensors header length")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError("safetensors file has a truncated header")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid safetensors header JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header, 8 + header_length


def _tensor_bytes(value: torch.Tensor) -> bytes:
    value = value.detach().to("cpu").contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def _clone_or_copy_checkpoint(parent: Path, child: Path) -> str:
    """Create an immutable child base, using filesystem CoW when available.

    The training worker changes only a few fixed-size output-head ranges.  A
    reflink makes that operation cheap on filesystems that support
    ``FICLONE`` while preserving the safe copy fallback for NFS, ext4
    versions without reflink support, and non-Linux hosts.
    """

    if fcntl is not None:
        try:
            with parent.open("rb") as source, child.open("w+b") as target:
                fcntl.ioctl(target.fileno(), 0x40049409, source.fileno())  # FICLONE
            return "reflink"
        except (OSError, ValueError):
            try:
                child.unlink()
            except FileNotFoundError:
                pass
    shutil.copy2(parent, child)
    return "copy"


def patch_safetensors_tensors(
    parent: Path,
    child: Path,
    replacements: Mapping[str, torch.Tensor],
) -> None:
    """Copy parent and replace fixed-size tensor payloads without changing its header."""

    parent = Path(parent).resolve()
    child = Path(child).resolve()
    if parent == child:
        raise ValueError("child checkpoint must differ from parent")
    if not parent.is_file():
        raise FileNotFoundError(parent)
    header, data_start = read_safetensors_header(parent)
    for name, value in replacements.items():
        if name not in header or name == "__metadata__":
            raise KeyError("replacement tensor is absent from parent: %s" % name)
        entry = header[name]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("data_offsets"), list):
            raise ValueError("invalid data offsets for tensor: %s" % name)
        offsets = entry["data_offsets"]
        if len(offsets) != 2 or not all(isinstance(x, int) for x in offsets):
            raise ValueError("invalid data offsets for tensor: %s" % name)
        payload = _tensor_bytes(value)
        expected = int(offsets[1]) - int(offsets[0])
        if len(payload) != expected:
            raise ValueError("replacement byte size mismatch for tensor: %s" % name)
    child.parent.mkdir(parents=True, exist_ok=True)
    _clone_or_copy_checkpoint(parent, child)
    with child.open("r+b") as stream:
        for name, value in replacements.items():
            start, end = header[name]["data_offsets"]
            payload = _tensor_bytes(value)
            stream.seek(data_start + start)
            stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def tensor_payload_equal(path_a: Path, path_b: Path, name: str) -> bool:
    """Compare one tensor by streaming it through the safetensors reader."""

    from safetensors import safe_open

    with safe_open(str(path_a), framework="pt", device="cpu") as first:
        with safe_open(str(path_b), framework="pt", device="cpu") as second:
            if name not in first.keys() or name not in second.keys():
                return False
            return torch.equal(first.get_tensor(name), second.get_tensor(name))
