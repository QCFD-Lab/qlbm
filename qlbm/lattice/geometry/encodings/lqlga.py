from abc import ABC
from typing import Tuple


class LQLGAReflectionData(ABC):
    """Base class for all space-time reflectiopn data structures."""

    pass


class LQLGAPointwiseReflectionData(LQLGAReflectionData):
    """
    Data structure representing pointwise space-time reflection information for LQLGA geometries.

    Attributes:
        gridpoints (Tuple[Tuple[int, ...], Tuple[int, ...]]):
            A tuple containing two grid points (as tuples of integers) that are
            related by a reflection operation.
        velocity_indices_to_swap (Tuple[int, int]):
            A tuple of two velocity indices that should be swapped during the
            reflection operation.
    """

    def __init__(
        self,
        gridpoints: Tuple[Tuple[int, ...], Tuple[int, ...]],
        velocity_indices_to_swap: Tuple[int, int],
    ) -> None:
        self.gridpoints = gridpoints
        self.velocity_indices_to_swap = velocity_indices_to_swap
