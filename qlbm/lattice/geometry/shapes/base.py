"""Base classes for geometrical shapes."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from stl import mesh

from qlbm.lattice.geometry.encodings.spacetime import (
    SpaceTimePWReflectionData,
    SpaceTimeVolumetricReflectionData,
)
from qlbm.lattice.spacetime.properties_base import SpaceTimeLatticeBuilder
from qlbm.tools.utils import flatten, get_qubits_to_invert


class Shape(ABC):
    """Base class for all geometrical shapes."""

    def __init__(self, num_grid_qubits: List[int], boundary_condition: str):
        super().__init__()

        self.num_grid_qubits = num_grid_qubits
        self.boundary_condition = boundary_condition
        self.num_dims = len(num_grid_qubits)
        # The number of qubits used to offset "higher" dimensions
        self.previous_qubits: List[int] = [
            sum(num_grid_qubits[previous_dim] for previous_dim in range(dim))
            for dim in range(self.num_dims)
        ]

    @abstractmethod
    def stl_mesh(self) -> mesh.Mesh:
        """
        Provides the ``stl`` representation of the shape.

        Returns
        -------
        ``stl.mesh.Mesh``
            The mesh representing the shape.
        """
        pass

    @abstractmethod
    def to_json(self):
        """
        Serializes the shape to JSON format.

        Returns
        -------
        str
            The JSON representation of the shape.
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Produces a dictionary representation of the shape.

        Returns
        -------
        Dict[str, List[int] | str]
            A dictionary representation of the bounds and boundary conditions of the shape.
        """
        pass


class SpaceTimeShape(Shape):
    """Base class for all shapes compatible with the :class:`.SpaceTimeQLBM` algorithm."""

    def __init__(self, num_grid_qubits: List[int], boundary_condition: str):
        super().__init__(num_grid_qubits, boundary_condition)

    def get_spacetime_reflection_data_d2q4_from_points(
        self,
        properties: SpaceTimeLatticeBuilder,
        gridpoints: List[Tuple[int, ...]],
        streaming_line_velocities: List[int],
        symmetric_non_reflection_dimension_increment: List[Tuple[int, Tuple[int, ...]]],
        num_steps: int,
    ) -> List[SpaceTimePWReflectionData]:
        """
        Traverses the stencil of a collection of grid points and computes the corresponding :class:`.SpaceTimePWReflectionData`.

        Parameters
        ----------
        properties : SpaceTimeLatticeBuilder
            The lattice discretizationproperties.
        gridpoints : List[Tuple[int, ...]]
            The gridpoints to compute the reflection data for.
        streaming_line_velocities : List[int]
            The streaming lines to traverse.
        symmetric_non_reflection_dimension_increment : List[Tuple[int, Tuple[int, ...]]]
            A list of increments in the non-traversing direction(s) that when applied iteratively cover the stencil.
        num_steps : int
            The number of steps to compute the reflection data for.

        Returns
        -------
        List[SpaceTimePWReflectionData]
            The reflection data for each position in the stencil and each lattice gridpoint.
        """
        reflection_list: List[SpaceTimePWReflectionData] = []
        # Each surface is made up of several gridpoints
        for gridpoint in gridpoints:
            for reflection_direction in streaming_line_velocities:
                increment = properties.get_reflection_increments(reflection_direction)

                # Each gridpoint is part of the stencil of several others
                for (
                    offset,
                    starting_increment,
                ) in symmetric_non_reflection_dimension_increment:
                    origin: Tuple[int, ...] = tuple(
                        a + b for a, b in zip(gridpoint, starting_increment)
                    )
                    for t in range(num_steps - abs(offset)):
                        num_gps_in_dim = [2**n for n in self.num_grid_qubits]
                        gridpoint_encoded = tuple(
                            a
                            + (
                                t + 1
                                if (
                                    streaming_line_velocities[0] == reflection_direction
                                )
                                else t
                            )
                            * b
                            for a, b in zip(origin, increment)
                        )

                        # Add periodic boundary conditions
                        gridpoint_encoded = tuple(
                            x + num_gps_in_dim if x < 0 else x % num_gps_in_dim
                            for x, num_gps_in_dim in zip(
                                gridpoint_encoded, num_gps_in_dim
                            )
                        )
                        distance_from_origin = tuple(
                            (t + 1) * x + y
                            for x, y in zip(increment, starting_increment)
                        )

                        # The qubits to invert for this gridpoint
                        qubits_to_invert = flatten(
                            [
                                [
                                    self.previous_qubits[dim] + q
                                    for q in get_qubits_to_invert(
                                        gridpoint_encoded[dim],
                                        self.num_grid_qubits[dim],
                                    )
                                ]
                                for dim in range(2)
                            ]
                        )
                        opposite_reflection_direction = (
                            streaming_line_velocities[1]
                            if reflection_direction == streaming_line_velocities[0]
                            else streaming_line_velocities[0]
                        )

                        reflection_list.append(
                            SpaceTimePWReflectionData(
                                gridpoint_encoded,
                                qubits_to_invert,
                                [opposite_reflection_direction],
                                distance_from_origin,
                                properties,
                            )
                        )

        return reflection_list

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
        """Calculate space-time reflection data for :math:`D_1Q_2` :class:`.SpaceTimeQLBM` lattice.

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
        Calculate space-time reflection data for :math:`D_2Q_4` :class:`.SpaceTimeQLBM` lattice.

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
        """Calculate volumetric reflection data for :math:`D_2Q_4` :class:`.SpaceTimeQLBM` lattice.

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

    @abstractmethod
    def name(self) -> str:
        """
        The name of the shape.

        Returns
        -------
        str
            The name of the shape.
        """
        pass
