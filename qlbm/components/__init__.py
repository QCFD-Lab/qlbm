"""Modular and extendible quantum circuits that perform parts of the QLBM algorithm."""

from .ab import (
    ABQLBM,
    ABDiscreteUniformInitialConditions,
    ABGridMeasurement,
    ABInitialConditions,
    ABParallelDiscreteUniformInitialConditions,
    ABReflectionOperator,
    ABReflectionPermutation,
    ABStreamingOperator,
    ABZoneAgnosticReflectionOperator,
    ABZoneAgnosticReflectionOracle,
)
from .base import (
    LBMAlgorithm,
    LBMOperator,
    LBMPrimitive,
    MSOperator,
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
from .common.adders import ParameterizedDraperAdder, ParameterizedPhaseShift, PhaseShift
from .cqlbm import CQLBM
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
    MSQLBM,
    BounceBackReflectionOperator,
    GridMeasurement,
    MSInitialConditions,
    MSStreamingOperator,
    SpecularReflectionOperator,
)
from .ms.primitives import Comparator, ComparatorMode
from .ms.streaming import (
    ControlledIncrementer,
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
    "ParameterizedDraperAdder",
    "PhaseShift",
    "ParameterizedPhaseShift",
    "EmptyPrimitive",
    "StreamingAncillaPreparation",
    "ControlledIncrementer",
    "GridMeasurement",
    "MSInitialConditions",
    "MSStreamingOperator",
    "SpecularReflectionOperator",
    "BounceBackReflectionOperator",
    "MSQLBM",
    "GenericLQLGACollisionOperator",
    "LQGLAInitialConditions",
    "LQLGA",
    "CQLBM",
    "LQLGAGridVelocityMeasurement",
    "LQLGAMGReflectionOperator",
    "LQLGAReflectionOperator",
    "LQLGAStreamingOperator",
    "EQCCollisionOperator",
    "EQCPermutation",
    "EQCRedistribution",
    "HammingWeightAdder",
    "ABQLBM",
    "ABInitialConditions",
    "ABGridMeasurement",
    "ABReflectionOperator",
    "ABReflectionPermutation",
    "ABStreamingOperator",
    "ABDiscreteUniformInitialConditions",
    "ABParallelDiscreteUniformInitialConditions",
    "ABZoneAgnosticReflectionOperator",
    "ABZoneAgnosticReflectionOracle",
]
