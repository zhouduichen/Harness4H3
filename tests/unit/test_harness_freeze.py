from __future__ import annotations

from harness4h3 import HARNESS_CHANGE_POLICY, HARNESS_STATUS, HARNESS_VERSION, __version__


def test_harness_is_explicitly_marked_research_preview():
    assert __version__ == "0.4.0"
    assert HARNESS_VERSION == "Harness4H3-v0.4"
    assert HARNESS_STATUS == "research_preview"
    assert HARNESS_CHANGE_POLICY == "research_changes_allowed"
