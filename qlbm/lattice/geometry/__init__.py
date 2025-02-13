"""Geometrical data, including shapes and algorithm-specific data structures."""

from .encodings import (
    DimensionalReflectionData,
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
    SpaceTimeReflectionData,
    SpaceTimeVolumetricReflectionData,
)
from .shapes import Block

__all__ = [
    "DimensionalReflectionData",
    "ReflectionPoint",
    "ReflectionWall",
    "ReflectionResetEdge",
    "SpaceTimeReflectionData",
    "SpaceTimeVolumetricReflectionData",
    "Block",
]
