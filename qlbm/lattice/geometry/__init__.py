"""Geometrical data, including shapes and algorithm-specific data structures."""

from .encodings import (
    DimensionalReflectionData,
    LQLGAPointwiseReflectionData,
    LQLGAReflectionData,
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
    "LQLGAPointwiseReflectionData",
    "LQLGAReflectionData",
]
