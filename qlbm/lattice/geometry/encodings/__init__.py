"""Algorithm-specific geometrical data encodings."""

from .lqlga import LQLGAPointwiseReflectionData, LQLGAReflectionData
from .ms import (
    DimensionalReflectionData,
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
)
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
