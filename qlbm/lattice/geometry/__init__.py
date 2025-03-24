"""Geometrical data, including shapes and algorithm-specific data structures."""

from .encodings import (
    DimensionalReflectionData,
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
    SpaceTimeDiagonalReflectionData,
    SpaceTimePWReflectionData,
    SpaceTimeVolumetricReflectionData,
)
from .shapes import Block, Circle

__all__ = [
    "DimensionalReflectionData",
    "ReflectionPoint",
    "ReflectionWall",
    "ReflectionResetEdge",
    "SpaceTimeDiagonalReflectionData",
    "SpaceTimePWReflectionData",
    "SpaceTimeVolumetricReflectionData",
    "Block",
    "Circle",
]
