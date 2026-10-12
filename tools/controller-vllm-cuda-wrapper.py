#!/usr/bin/env python3
"""Launch vLLM with an explicit CUDA platform on hosts with broken NVML."""

from __future__ import annotations

import sys

import vllm.platforms as platforms
from vllm.platforms.cuda import CudaPlatform


platforms.current_platform = CudaPlatform()

from vllm.entrypoints.cli.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
