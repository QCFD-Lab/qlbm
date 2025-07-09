"""Classes implementing the logic of the collision operator in the Space-Time Data encoding."""

from ....lattice.eqc.eqc import EquivalenceClass
from .d2q4_old import SpaceTimeD2Q4CollisionOperator
from ....lattice.eqc.eqc_generator import (
    EquivalenceClassGenerator,
)

__all__ = [
    "SpaceTimeD2Q4CollisionOperator",
    "EquivalenceClass",
    "EquivalenceClassGenerator",
]
