"""Geometrical data, including shapes and algorithm-specific data structures."""

from .encodings import (
    DimensionalReflectionData,
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
    SpaceTimePWReflectionData,
    SpaceTimeVolumetricReflectionData,
)
from .shapes import Block, Circle

__all__ = [
    "DimensionalReflectionData",
    "ReflectionPoint",
    "ReflectionWall",
    "ReflectionResetEdge",
    "SpaceTimePWReflectionData",
    "SpaceTimeVolumetricReflectionData",
    "Block",
    "Circle",
]
