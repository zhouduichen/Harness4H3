"""Compatibility imports for the frozen workflow-evolution implementation.

EvoGen Phase I does not import this module.  It remains available so existing
deployments and archived experiments can still be replayed.
"""

from ..legacy.workflow_evolution import (  # noqa: F401
    Diagnosis,
    EvolutionController,
    EvolutionOutcome,
    diagnose,
    propose_mutation,
    should_promote,
)

__all__ = [
    "Diagnosis",
    "EvolutionController",
    "EvolutionOutcome",
    "diagnose",
    "propose_mutation",
    "should_promote",
]
