from types import SimpleNamespace

from harness4h3.remote.checkpoint_retention import RemoteCheckpointRetention
from harness4h3.remote.ssh import RemoteConfig, SSHClient


class FakeRetentionClient(SSHClient):
    def __init__(self):
        super().__init__(
            RemoteConfig(
                "fake",
                "/srv/harness",
                "/srv/models",
                "/srv/comfy",
                results_root="/srv/models/results",
            ),
            runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        self.removed = []
        self.failed_files = []

    def remove_file(self, path):
        self.removed.append(path)
        return {"status": "deleted"}

    def find(self, pattern, root=None):
        return list(self.failed_files)


def test_rejected_remote_child_is_deleted_but_accepted_child_is_kept():
    client = FakeRetentionClient()
    retention = RemoteCheckpointRetention(client, "/srv/models/results")
    child = "/srv/models/results/continuous/M0001/M0001.safetensors"

    rejected = retention.apply(child, "M0001", "rejected_candidate", "/srv/models/results/continuous/M0000/M0000.safetensors")
    accepted = retention.apply(
        "/srv/models/results/continuous/M0002/M0002.safetensors",
        "M0002",
        "accepted_candidate",
    )

    assert rejected.deleted is True
    assert rejected.retained is False
    assert accepted.retained is True
    assert client.removed == [child]


def test_remote_retention_refuses_outside_and_malformed_paths():
    client = FakeRetentionClient()
    retention = RemoteCheckpointRetention(client, "/srv/models/results")

    outside = retention.apply("/srv/models/other/M0001.safetensors", "M0001", "rejected_candidate")
    malformed = retention.apply(
        "/srv/models/results/continuous/M0001/other.safetensors",
        "M0001",
        "rejected_candidate",
    )

    assert outside.retained is True and outside.deleted is False
    assert malformed.retained is True and malformed.deleted is False
    assert client.removed == []


def test_failed_worker_cleanup_targets_only_child_and_partial_files():
    client = FakeRetentionClient()
    retention = RemoteCheckpointRetention(client, "/srv/models/results")
    result = retention.cleanup_failed("/srv/models/results/continuous/M0003", "M0003")

    assert result.deleted is True
    assert result.retained is False
    assert client.removed == [
        "/srv/models/results/continuous/M0003/M0003.safetensors",
        "/srv/models/results/continuous/M0003/M0003.safetensors.part",
    ]


def test_failed_worker_cleanup_removes_named_failed_checkpoint_but_not_evidence():
    client = FakeRetentionClient()
    client.failed_files = [
        "/srv/models/results/continuous/M0003/M0003.failed-invalid-parent-id.safetensors",
        "/srv/models/results/continuous/M0003/M0003.failed-invalid-parent-id.safetensors.evidence.json",
        "/srv/models/results/continuous/M0002/M0003.failed-other.safetensors",
    ]
    retention = RemoteCheckpointRetention(client, "/srv/models/results")

    result = retention.cleanup_failed("/srv/models/results/continuous/M0003", "M0003")

    assert result.deleted is True
    assert client.removed == [
        "/srv/models/results/continuous/M0003/M0003.safetensors",
        "/srv/models/results/continuous/M0003/M0003.safetensors.part",
        "/srv/models/results/continuous/M0003/M0003.failed-invalid-parent-id.safetensors",
    ]


def test_superseded_candidate_uses_the_same_bounded_delete_path():
    client = FakeRetentionClient()
    retention = RemoteCheckpointRetention(client, "/srv/models/results")
    result = retention.apply(
        "/srv/models/results/continuous/M0004/M0004.safetensors",
        "M0004",
        "superseded_candidate",
    )

    assert result.deleted is True
    assert result.reason == "superseded_candidate_deleted"
