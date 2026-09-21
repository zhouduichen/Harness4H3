from __future__ import annotations

from types import SimpleNamespace

from harness4h3.student.gpu import select_role_gpu_allocation


def test_worker_role_assignment_has_distinct_teacher_and_student_devices(monkeypatch):
    monkeypatch.setattr(
        "harness4h3.student.gpu.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0,50000\n1,48000\n2,30000\n3,10000\n"),
    )
    allocation = select_role_gpu_allocation(
        2,
        20.0,
        25.0,
        wait_s=0,
        worker_min_free_memory_gb=70.0,
    )
    assert len(allocation.teacher_devices) == 2
    assert allocation.student_device == "cuda:2"
    assert allocation.all_devices == ("cuda:0", "cuda:1", "cuda:2")
