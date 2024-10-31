from abc import ABC, abstractmethod
from enum import Enum
from logging import Logger, getLogger
from typing import Dict, List, Tuple, cast

from qiskit import QuantumRegister


class LatticeDiscretization(Enum):
    D1Q2 = (1,)
    D2Q4 = (2,)


class VonNeumannNeighborType(Enum):
    ORIGIN = (0,)
    INTERMEDIATE = (1,)
    EXTREME = (2,)


class VonNeumannNeighbor:
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
        self, origin: Tuple[int, int], relative: bool
    ) -> Tuple[int, int]:
        return cast(
            Tuple[int, int],
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
    def __init__(
        self,
        num_timesteps: int,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        self.num_timesteps = num_timesteps
        self.logger = logger

    @abstractmethod
    def get_discretization(self) -> LatticeDiscretization:
        pass

    @abstractmethod
    def get_num_velocities_per_point(self) -> int:
        pass

    @abstractmethod
    def get_num_ancilla_qubits(self) -> int:
        pass

    @abstractmethod
    def get_num_grid_qubits(self) -> int:
        pass

    @abstractmethod
    def get_num_velocity_qubits(self, num_timesteps: int | None = None) -> int:
        pass

    def get_num_total_qubits(self) -> int:
        return (
            self.get_num_ancilla_qubits()
            + self.get_num_grid_qubits()
            + self.get_num_velocity_qubits()
        )

    @abstractmethod
    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        pass

    def get_neighbor_distance_indices(self, distance_from_origin: int) -> List[int]:
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
        return (
            self.get_num_velocity_qubits(distance)
            // self.get_num_velocities_per_point()
        )

    @abstractmethod
    def get_index_of_neighbor(self, distance: Tuple[int, ...]) -> int:
        pass

    @abstractmethod
    def get_streaming_lines(
        self, dimension: int, direction: bool, timestep: int | None = None
    ) -> List[List[int]]:
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
