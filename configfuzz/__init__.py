"""ConfigFuzz constraint-inference prototype."""

from .corpus import ConstraintCorpus, ManualConstraintRule
from .model import Constraint, ConstraintKind, ConstraintSet, Evidence, EvidenceKind

__all__ = [
    "Constraint",
    "ConstraintCorpus",
    "ConstraintKind",
    "ConstraintSet",
    "Evidence",
    "EvidenceKind",
    "ManualConstraintRule",
]

__version__ = "0.1.0"
