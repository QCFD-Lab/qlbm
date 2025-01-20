"""Modular qlbm quantum circuit components for the CQLBM algorithm :cite:p:`collisionless`."""

from .bounceback_reflection import (
    BounceBackReflectionOperator,
    BounceBackWallComparator,
)
from .cqlbm import CQLBM
from .primitives import (
    CollisionlessInitialConditions,
    CollisionlessInitialConditions3DSlim,
    Comparator,
    ComparatorMode,
    EdgeComparator,
    GridMeasurement,
    SpeedSensitiveAdder,
)
from .specular_reflection import SpecularReflectionOperator, SpecularWallComparator
from .streaming import (
    CollisionlessStreamingOperator,
    ControlledIncrementer,
    PhaseShift,
    SpeedSensitivePhaseShift,
    StreamingAncillaPreparation,
)

__all__ = [
    "ComparatorMode",
    "Comparator",
    "SpeedSensitiveAdder",
    "StreamingAncillaPreparation",
    "ControlledIncrementer",
    "GridMeasurement",
    "EdgeComparator",
    "CollisionlessInitialConditions",
    "CollisionlessInitialConditions3DSlim",
    "PhaseShift",
    "SpeedSensitivePhaseShift",
    "CollisionlessStreamingOperator",
    "SpecularReflectionOperator",
    "SpecularWallComparator",
    "BounceBackReflectionOperator",
    "BounceBackWallComparator",
    "CQLBM",
]
