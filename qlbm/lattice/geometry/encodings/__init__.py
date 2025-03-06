"""Algorithm-specific geometrical data encodings."""

from .collisionless import (
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
]
