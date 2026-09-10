from __future__ import annotations

from harness4h3 import HARNESS_CHANGE_POLICY, HARNESS_STATUS, HARNESS_VERSION, __version__


def test_phase_i_harness_is_explicitly_frozen():
    assert __version__ == "1.0.0"
    assert HARNESS_VERSION == "Harness4H3-v1.0"
    assert HARNESS_STATUS == "frozen"
    assert HARNESS_CHANGE_POLICY == "bugfix_only"
