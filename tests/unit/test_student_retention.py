from __future__ import annotations

from harness4h3.student.retention import apply_retention, retain_after_evaluation


def test_retention_preserves_active_parent_and_deletes_only_rejected_child(tmp_path):
    parent = tmp_path / "parent.safetensors"
    child_dir = tmp_path / "student_0001"
    child = child_dir / "student.safetensors"
    parent.write_bytes(b"parent")
    child_dir.mkdir()
    child.write_bytes(b"child")
    decision = retain_after_evaluation(child, "student_0001", outcome="rejected", protected={parent})
    assert decision.delete_path == str(child.resolve())
    assert parent.exists()
    applied = apply_retention(decision)
    assert not child.exists()
    assert applied.reason == "candidate_deleted"


def test_retention_refuses_symlink_or_unexpected_layout(tmp_path):
    target = tmp_path / "actual.safetensors"
    target.write_bytes(b"child")
    child_dir = tmp_path / "student_0001"
    child_dir.mkdir()
    link = child_dir / "student.safetensors"
    link.symlink_to(target)
    decision = retain_after_evaluation(link, "student_0001", outcome="rejected")
    assert decision.delete_path is None
    assert decision.reason == "symlink_refused"
