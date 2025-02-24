"""Base classes for geometrical shapes."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from stl import mesh

from qlbm.lattice.geometry.encodings.spacetime import (
    SpaceTimePWReflectionData,
    SpaceTimeVolumetricReflectionData,
)
from qlbm.lattice.spacetime.properties_base import SpaceTimeLatticeBuilder


class Shape(ABC):
    """Base class for all geometrical shapes."""

    @abstractmethod
    def stl_mesh(self) -> mesh.Mesh:
        """
        Provides the ``stl`` representation of the block.

        Returns
        -------
        ``stl.mesh.Mesh``
            The mesh representing the block.
        """
        pass

    @abstractmethod
    def to_json(self):
        """
        Serializes the shape to JSON format.

        Returns
        -------
        str
            The JSON representation of the block.
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Produces a dictionary representation of the block.

        Returns
        -------
        Dict[str, List[int] | str]
            A dictionary representation of the bounds and boundary conditions of the block.
        """
        pass


class SpaceTimeShape(Shape):
    """Base class for all shapes compatible with the :class:`.STQLBM` algorithm."""

    @abstractmethod
    def get_d2q4_surfaces(self) -> List[List[List[Tuple[int, ...]]]]:
        """
        Get all surfaces of the block in 2 dimensions.

        The information is formatted as ``List[List[List[Tuple[int, ...]]]]``.
        The outermost list is by dimension.
        The middle list contains two lists pertaining to the lower and upper bounds of the block in that dimenison.
        The innermost list contains the gridpoints that make up the surface encoded as tuples.

        Returns
        -------
        List[List[List[Tuple[int, ...]]]]
            The block surfaces in two dimensions.
        """

    @abstractmethod
    def contains_gridpoint(self, gridpoint: Tuple[int, ...]) -> bool:
        """
        Whether the block contains a given gridpoint within its volume.

        Parameters
        ----------
        gridpoint : Tuple[int, ...]
            The gridpoint to check for.

        Returns
        -------
        bool
            Whether the gridpoint is within the block.
        """
        pass

    @abstractmethod
    def get_spacetime_reflection_data_d1q2(
        self,
        properties: SpaceTimeLatticeBuilder,
        num_steps: int | None = None,
    ) -> List[SpaceTimePWReflectionData]:
        """Calculate space-time reflection data for :math:`D_1Q_2` :class:`.STQLBM` lattice.

        Parameters
        ----------
        properties : SpaceTimeLatticeBuilder
            The lattice discretization properties.
        num_steps int | None, optional
            Number of timesteps to calculate reflections for. If None, uses ``properties.num_timesteps``. Defaults to None.

        Returns
        -------
        List[SpaceTimeReflectionData]
            The information encoding the reflections to be performed.
        """
        pass

    @abstractmethod
    def get_spacetime_reflection_data_d2q4(
        self,
        properties: SpaceTimeLatticeBuilder,
        num_steps: int | None = None,
    ) -> List[SpaceTimePWReflectionData]:
        """
        Calculate space-time reflection data for :math:`D_2Q_4` :class:`.STQLBM` lattice.

        Parameters
        ----------
        properties : SpaceTimeLatticeBuilder
            The lattice discretization properties.
        num_steps : int | None, optional
            Number of timesteps to calculate reflections for. If None, uses ``properties.num_timesteps``. Defaults to None.

        Returns
        -------
        List[SpaceTimeReflectionData]
            The information encoding the reflections to be performed.
        """
        pass

    @abstractmethod
    def get_d2q4_volumetric_reflection_data(
        self,
        properties: SpaceTimeLatticeBuilder,
        num_steps: int | None = None,
    ) -> List[SpaceTimeVolumetricReflectionData]:
        """Calculate volumetric reflection data for :math:`D_2Q_4` :class:`.STQLBM` lattice.

        Parameters
        ----------
        properties : SpaceTimeLatticeBuilder
            The lattice discretization properties.
        num_steps int | None, optional
            Number of timesteps to calculate reflections for. If None, uses ``properties.num_timesteps``. Defaults to None.

        Returns
        -------
        List[SpaceTimeVolumetricReflectionData]
            The information encoding the reflections to be performed.
        """
        pass
