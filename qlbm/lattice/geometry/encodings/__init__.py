"""Algorithm-specific geometrical data encodings."""

from .collisionless import (
    DimensionalReflectionData,
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
)
from .spacetime import SpaceTimeReflectionData, SpaceTimeVolumetricReflectionData

__all__ = [
    "DimensionalReflectionData",
    "ReflectionPoint",
    "ReflectionWall",
    "ReflectionResetEdge",
    "SpaceTimeReflectionData",
    "SpaceTimeVolumetricReflectionData",
]
