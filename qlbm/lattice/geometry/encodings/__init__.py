"""Algorithm-specific geometrical data encodings."""

from .ms import (
    DimensionalReflectionData,
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
)
from .lqlga import LQLGAPointwiseReflectionData, LQLGAReflectionData
from .spacetime import (
    SpaceTimeDiagonalReflectionData,
    SpaceTimePWReflectionData,
    SpaceTimeVolumetricReflectionData,
)

__all__ = [
    "DimensionalReflectionData",
    "ReflectionPoint",
    "ReflectionWall",
    "ReflectionResetEdge",
    "SpaceTimePWReflectionData",
    "SpaceTimeVolumetricReflectionData",
    "SpaceTimeDiagonalReflectionData",
    "LQLGAPointwiseReflectionData",
    "LQLGAReflectionData",
]
