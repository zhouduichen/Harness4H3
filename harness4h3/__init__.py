"""Harness4H3 public package."""

__version__ = "1.0.0"

# Phase-I experiments run against one immutable optimization environment. Keep
# this identifier in evidence and trajectories so improvements are attributed
# to the model/runtime candidate, not to an implicitly changing Harness.
HARNESS_VERSION = "Harness4H3-v1.0"
HARNESS_STATUS = "frozen"
HARNESS_CHANGE_POLICY = "bugfix_only"


class Harness4H3Error(Exception):
    """Base class for expected, user-facing harness failures."""
