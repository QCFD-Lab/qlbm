"""Primitives for preparing the initial state for the :class:`.SpaceTimeQLBM` algorithm."""

from .pointwise import PointWiseSpaceTimeInitialConditions
from .volumetric import VolumetricSpaceTimeInitialConditions

__all__ = [
    "PointWiseSpaceTimeInitialConditions",
    "VolumetricSpaceTimeInitialConditions",
]
