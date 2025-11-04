"""Modular and extendible quantum circuits that perform parts of the QLBM algorithm."""

from .base import (
    MSOperator,
    LBMAlgorithm,
    LBMOperator,
    LBMPrimitive,
    QuantumComponent,
    SpaceTimeOperator,
)
from .common import (
    EmptyPrimitive,
    EQCCollisionOperator,
    EQCPermutation,
    EQCRedistribution,
    HammingWeightAdder,
)
from .lqlga import (
    LQLGA,
    GenericLQLGACollisionOperator,
    LQGLAInitialConditions,
    LQLGAGridVelocityMeasurement,
    LQLGAMGReflectionOperator,
    LQLGAReflectionOperator,
    LQLGAStreamingOperator,
)
from .ms import (
    CQLBM,
    BounceBackReflectionOperator,
    MSInitialConditions,
    GridMeasurement,
    MSStreamingOperator,
    SpecularReflectionOperator,
)
from .ms.primitives import Comparator, ComparatorMode, SpeedSensitiveAdder
from .ms.streaming import (
    ControlledIncrementer,
    PhaseShift,
    SpeedSensitivePhaseShift,
    StreamingAncillaPreparation,
)

__all__ = [
    "QuantumComponent",
    "LBMPrimitive",
    "GenericLQLGACollisionOperator",
    "LBMOperator",
    "MSOperator",
    "SpaceTimeOperator",
    "LBMAlgorithm",
    "ComparatorMode",
    "Comparator",
    "SpeedSensitiveAdder",
    "PhaseShift",
    "SpeedSensitivePhaseShift",
    "EmptyPrimitive",
    "StreamingAncillaPreparation",
    "ControlledIncrementer",
    "GridMeasurement",
    "MSInitialConditions",
    "MSStreamingOperator",
    "SpecularReflectionOperator",
    "BounceBackReflectionOperator",
    "CQLBM",
    "GenericLQLGACollisionOperator",
    "LQGLAInitialConditions",
    "LQLGA",
    "LQLGAGridVelocityMeasurement",
    "LQLGAMGReflectionOperator",
    "LQLGAReflectionOperator",
    "LQLGAStreamingOperator",
    "EQCCollisionOperator",
    "EQCPermutation",
    "EQCRedistribution",
    "HammingWeightAdder",
]
