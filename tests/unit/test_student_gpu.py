from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness4h3.student.gpu import GPUResourceUnavailable, select_free_cuda_devices


def test_gpu_lease_assigns_distinct_devices_by_memory(monkeypatch):
    monkeypatch.setattr(
        "harness4h3.student.gpu.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0,45444\n1,30000\n2,12000\n"),
    )
    assert select_free_cuda_devices((44.0, 20.0), wait_s=0) == ("cuda:0", "cuda:1")


def test_gpu_lease_reports_contention_without_wait(monkeypatch):
    monkeypatch.setattr(
        "harness4h3.student.gpu.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0,30000\n"),
    )
    with pytest.raises(GPUResourceUnavailable, match="fewer than 2 GPUs"):
        select_free_cuda_devices((44.0, 20.0), wait_s=0)
