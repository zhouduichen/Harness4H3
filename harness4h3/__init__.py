"""Harness4H3 public package."""

__version__ = "0.4.0"

# Research-preview experiments record this identifier so improvements are
# attributed to the model/runtime candidate and the evolving protocol remains
# visible in evidence.
HARNESS_VERSION = "Harness4H3-v0.4"
HARNESS_STATUS = "research_preview"
HARNESS_CHANGE_POLICY = "research_changes_allowed"


class Harness4H3Error(Exception):
    """Base class for expected, user-facing harness failures."""
