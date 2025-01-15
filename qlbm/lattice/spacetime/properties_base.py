"""
Base module defining the properties and structure of STQBM lattice.

This module provides abstract base classes and enums for building and managing
static STQBM lattice properties simulations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from logging import Logger, getLogger
from typing import Dict, List, Tuple, cast

from qiskit import QuantumRegister


class LatticeDiscretization(Enum):
    """
    The Lattice Boltzmann discretization used in the simulation.

    The only supported discretizations currently are D1Q2 and D2Q4.
    """

    D1Q2 = (1,)
    D2Q4 = (2,)


class VonNeumannNeighborType(Enum):
    """
    The type of neighbor in the von Neumann neighborhood.

    Neighbors are either the origin (0), intermediate (1), or extreme (2).
    Extreme neighbors are those at the very edges of the von Neumann neighborhood.
    """

    ORIGIN = (0,)
    INTERMEDIATE = (1,)
    EXTREME = (2,)


class VonNeumannNeighbor:
    """Class encoding the information of a von Neumann neighbor in STQBM data encoding."""

    def __init__(
        self,
        relative_coordinates: Tuple[int, int],
        neighborhood_index: int,
        neighbor_type: VonNeumannNeighborType,
    ):
        self.coordinates_relative = relative_coordinates
        self.coordinates_inverse = [-1 * coord for coord in relative_coordinates]
        self.neighbor_index = neighborhood_index
        self.neighbor_type = neighbor_type

    def __eq__(self, other):
        """Compare two VonNeumannNeighbor instances for equality.

        Two neighbors are qual if their coordinates, index, and type are equal.

        Parameters
        ----------
        other : Any
            The object to compare with.

        Returns
        -------
        bool
            True if objects are equal, False otherwise.
        """
        if not isinstance(other, VonNeumannNeighbor):
            return NotImplemented

        return (
            all(
                c0 == c1
                for c0, c1 in zip(self.coordinates_relative, other.coordinates_relative)
            )
            and all(
                c0 == c1
                for c0, c1 in zip(self.coordinates_inverse, other.coordinates_inverse)
            )
            and self.neighbor_index == other.neighbor_index
            and self.neighbor_type == other.neighbor_type
        )

    def get_absolute_values(
        self, origin: Tuple[int, ...], relative: bool
    ) -> Tuple[int, ...]:
        """
        Get the absolute coordinates of the neighbor.

        Parameters
        ----------
        origin : Tuple[int, ...]
            The absolute coordinates of the origin.
        relative : bool
            Whether the coordinates are relative to the origin. If this is false, the coordinates are inversed in the result.

        Returns
        -------
        Tuple[int, ...]
            The absolute coordinates of the neighbor.
        """
        return cast(
            Tuple[int, ...],
            tuple(
                c0 + c1
                for c0, c1 in zip(
                    (
                        self.coordinates_relative
                        if relative
                        else self.coordinates_inverse
                    ),
                    origin,
                )
            ),
        )


class SpaceTimeLatticeBuilder(ABC):
    """
    Abstract base class for building STQBM lattice structures..

    This class provides the interface and common functionality for creating and managing
    lattice configurations. Each object stores discretization-specific information
    such as the number of qubits required for different components of the system
    and performs calculations based on the discretization.
    """

    origin: VonNeumannNeighbor

    def __init__(
        self,
        num_timesteps: int,
        include_measurement_qubit: bool = False,
        use_volumetric_ops: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        self.num_timesteps = num_timesteps
        self.include_measurement_qubit = include_measurement_qubit
        self.use_volumetric_ops = use_volumetric_ops
        self.logger = logger

    @abstractmethod
    def get_num_previous_grid_qubits(self, dim: int) -> int:
        """
        Get the number of qubits required to encode lower dimensions of the grid.

        Parameters
        ----------
        dim : int
            The dimension of the grid for which to compute the number of previous qubits.

        Returns
        -------
        int
            The number of qubits required to encode the lower dimensions of the grid.
        """
        pass

    @abstractmethod
    def get_discretization(self) -> LatticeDiscretization:
        """
        Get the discretization of the lattice.

        Returns
        -------
        LatticeDiscretization
            The discretization of the lattice.
        """
        pass

    @abstractmethod
    def get_num_velocities_per_point(self) -> int:
        """
        Get the number of discrete velocities for each grid location.

        Returns
        -------
        int
            The number of discrete velocities for each grid location.
        """
        pass

    @abstractmethod
    def get_num_ancilla_qubits(self) -> int:
        """
        Get the number of ancilla qubits required by the simulation.

        Returns
        -------
        int
            The number of ancilla qubits needed for the simulation.
        """
        pass

    @abstractmethod
    def get_num_grid_qubits(self) -> int:
        """
        Get the number of qubits required to encode the grid.

        Returns
        -------
            int: The nubmer of qubits required to encode the grid.
        """
        pass

    @abstractmethod
    def get_num_velocity_qubits(self, num_timesteps: int | None = None) -> int:
        """
        Get the number of required qubits to encode the velocity components in the quantum circuit.

        Parameters
        ----------
        num_timesteps : int | None, optional
            Number of timesteps for which to calculate the number of velocities. If None, uses class default, by default None

        Returns
        -------
        int
            Number of qubits required to encode velocity components
        """
        pass

    def get_num_total_qubits(self) -> int:
        """
        Get the total number of qubits required the lattice.

        Returns
        -------
        int
            The sum of ancillary qubits, grid qubits, and velocity qubits used in the system.
        """
        return (
            self.get_num_ancilla_qubits()
            + self.get_num_grid_qubits()
            + self.get_num_velocity_qubits()
        )

    @abstractmethod
    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        """Returns a tuple of lists of quantum registers this lattice configuration requires.

        ach list in the tuple represents a specific type or group of quantum registers: grid, velocity, and ancilla.

        Returns
        -------
        Tuple[List[QuantumRegister], ...]
            A tuple containing lists of quantum registers.
        """
        pass

    def get_neighbor_distance_indices(self, distance_from_origin: int) -> List[int]:
        """
        Get indices of neighboring points at a specific distance from the origin.

        This method returns a list of indices corresponding to neighboring points that are
        exactly at the specified Manhattan distance from the origin point in the lattice.

        Parameters
        ----------
        distance_from_origin : int
            The Manhattan distance from the origin point.

        Returns
        -------
        List[int]
            A list of indices corresponding to neighboring points at the specified distance.
            Returns [0] if distance_from_origin is 0.

        Notes
        -----
        The indices are calculated based on the number of velocity qubits and
        velocities per point in the lattice structure.
        """
        if distance_from_origin == 0:
            return [0]

        total_neighbors = int(
            (
                self.get_num_velocity_qubits(distance_from_origin)
                / self.get_num_velocities_per_point()
            )
        )
        neighbors_lower_distance = int(
            (
                self.get_num_velocity_qubits(distance_from_origin)
                / self.get_num_velocities_per_point()
            )
        )

        return list(range(neighbors_lower_distance, total_neighbors))

    def get_num_gridpoints_within_distance(self, distance: int) -> int:
        """
        Get the number of grid points within a given distance.

        Parameters
        ----------
        distance : int
            The distance from the origin.

        Returns
        -------
        int
            The number of grid points within the specified distance.
        """
        return (
            self.get_num_velocity_qubits(distance)
            // self.get_num_velocities_per_point()
        )

    @abstractmethod
    def get_index_of_neighbor(self, distance: Tuple[int, ...]) -> int:
        """
        Get the index of a neighbor at a given distance.

        Parameters
        ----------
        distance : Tuple[int, ...]
            The :math:`d`-dimensional distance tuple representing the neighbor's position.

        Returns
        -------
        int
            The index of the neighbor.
        """
        pass

    @abstractmethod
    def get_streaming_lines(
        self, dimension: int, direction: bool, timestep: int | None = None
    ) -> List[List[int]]:
        """
        Get the streaming lines along a given axis and direction.

        A streaming line defines the order in which qubits can be consistently swapped to achieve streaming.

        Parameters
        ----------
        dimension : int
            The dimension along which to stream: 0, 1, or 2.
        direction : bool
            Whether the streaming occurs from the negative (False) or positive (True) direction.
        timestep : int | None, optional
            The time step for which to perform streaming, by default None

        Returns
        -------
        List[List[int]]
            The ordered qubits to swap for each level of the stencil.
        """
        pass

    @abstractmethod
    def get_neighbor_indices(
        self,
    ) -> Tuple[
        Dict[int, List[VonNeumannNeighbor]],
        Dict[int, Dict[int, List[VonNeumannNeighbor]]],
    ]:
        """
        Get all of the information encoding the neighborhood structure of the lattice grid.

        We differentiate between two kinds of grid points, based on their relative position as neighbors relative to the origin.
        `Extreme points` are grid points that, in the expanding Manhattan distance stencil, have 3 neighbors with higher Manhattan distances.
        All other points (except the origin) are `intermediate points`, which have 2 neighbors with higher Manhattan distances and 2 neighbors with lower ones.
        Both extreme and intermediate points are further broken down into `classes`, based on which side of the origin they are on.
        The classes follow the same numerical ordering as the local :math:`D_2Q_4` discretization:
        0 encodes right, 1 up, 2 left, 3 down.
        These differences are relevant when constructing the :class:`.SpaceTimeStreamingOperator`.
        The structure of the output of this function contains two dictionaries, one for each kind of neighbor:

        #. A dictionary containing extreme points. The keys of the dictionary are Manhattan distances, and the entries are the indices of the neighbors, ordered by class.
        #. A dictionary containing intermediate points. The keys of the dictionary are Manhattan distances, and the entries are themselves dictionaries. For each nested dictionary, the key is the class, and the value is a list consisting of the point indices belonging to that class.

        Returns
        -------
        Tuple[ Dict[int, List[VonNeumannNeighbor]], Dict[int, Dict[int, List[VonNeumannNeighbor]]], ]
            The information encoding the lattice neighborhood structure.
        """
        pass

    @abstractmethod
    def get_reflected_index_of_velocity(self, velocity_index: int) -> int:
        """
        Gets the index that corresponds to the the velocity _after_ reflection has occurred.

        Parameters
        ----------
        velocity_index : int
            The velocity direction before reflection.

        Returns
        -------
        int
            The index of the velocity opposing the input.
        """
        pass

    @abstractmethod
    def get_reflection_increments(self, velocity_index: int) -> Tuple[int, ...]:
        """
        Gets the increment specific to a velocity direction.

        Parameters
        ----------
        velocity_index : int
            A discrete velocity direction.

        Returns
        -------
        Tuple[int, ...]
            The :math:`d-`dimensional increment.
        """
        pass
