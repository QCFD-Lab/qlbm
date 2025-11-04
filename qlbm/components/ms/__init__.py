"""Modular qlbm quantum circuit components for the CQLBM algorithm :cite:p:`collisionless`."""

from .bounceback_reflection import (
    BounceBackReflectionOperator,
    BounceBackWallComparator,
)
from .cqlbm import CQLBM
from .primitives import (
    MSInitialConditions,
    MSInitialConditions3DSlim,
    Comparator,
    ComparatorMode,
    EdgeComparator,
    GridMeasurement,
    SpeedSensitiveAdder,
)
from .specular_reflection import SpecularReflectionOperator, SpecularWallComparator
from .streaming import (
    ControlledIncrementer,
    MSStreamingOperator,
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
    "MSInitialConditions",
    "MSInitialConditions3DSlim",
    "PhaseShift",
    "SpeedSensitivePhaseShift",
    "MSStreamingOperator",
    "SpecularReflectionOperator",
    "SpecularWallComparator",
    "BounceBackReflectionOperator",
    "BounceBackWallComparator",
    "CQLBM",
]
