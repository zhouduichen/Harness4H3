from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness4h3.student.gpu import GPUResourceUnavailable, RuntimeResourceGate, select_free_cuda_devices, select_role_gpu_allocation


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


def test_role_allocation_assigns_distinct_student_device(monkeypatch):
    monkeypatch.setattr(
        "harness4h3.student.gpu.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0,60000\n1,55000\n2,25000\n"),
    )
    allocation = select_role_gpu_allocation(
        2,
        20.0,
        20.0,
        wait_s=0,
        worker_min_free_memory_gb=80.0,
    )
    assert allocation.teacher_devices == ("cuda:0", "cuda:1")
    assert allocation.student_device == "cuda:2"
    assert set(allocation.teacher_devices).isdisjoint({allocation.student_device})
    assert sum(allocation.free_memory_gb.values()) >= 80.0


def test_role_allocation_enforces_aggregate_worker_floor(monkeypatch):
    monkeypatch.setattr(
        "harness4h3.student.gpu.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0,30000\n1,25000\n2,22000\n"),
    )
    with pytest.raises(GPUResourceUnavailable, match="aggregate minimum"):
        select_role_gpu_allocation(2, 20.0, 20.0, wait_s=0, worker_min_free_memory_gb=100.0)


def test_runtime_resource_gate_uses_post_lease_free_memory_and_margin():
    RuntimeResourceGate(2.0).check(
        estimated_training_peak_memory_gb=10.0,
        actual_free_memory_gb=12.0,
        device="cuda:3",
    )
    with pytest.raises(GPUResourceUnavailable, match="safety margin"):
        RuntimeResourceGate(2.0).check(
            estimated_training_peak_memory_gb=10.1,
            actual_free_memory_gb=12.0,
            device="cuda:3",
        )
