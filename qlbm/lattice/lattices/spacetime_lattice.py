from enum import Enum
from logging import Logger, getLogger
from typing import Dict, List, Tuple, cast

from qiskit import QuantumCircuit, QuantumRegister

from qlbm.lattice.blocks import Block
from qlbm.tools.exceptions import CircuitException, LatticeException
from qlbm.tools.utils import dimension_letter, flatten

from .base import Lattice


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

    def velocity_index_to_swap(self, point_class: int, timestep: int) -> int:
        if timestep == 1:
            return (point_class + 2) % 4
        else:
            raise CircuitException("Not implemented.")


class SpaceTimeLattice(Lattice):
    """Lattice implementing the Space-Time data encoding."""

    # ! Only works for D2Q4

    # Points with 3 neighbors with higher Manhattan distances
    # In order:
    #   ^   |   ^   |   ^   |   x   |
    # x   > | <   > | <   x | <   > |
    #   v   |   x   |   v   |   v   |
    extreme_point_classes: List[Tuple[int, Tuple[int, int]]] = [
        (0, (1, 0)),
        (1, (0, 1)),
        (2, (-1, 0)),
        (3, (0, -1)),
    ]

    # Points with 2 neighbors with higher Manhattan distances
    # In order:
    #   ^   |   ^   |   x   |   x   |
    # x   > | <   x | <   x | x   > |
    #   x   |   x   |   v   |   v   |
    intermediate_point_classes: List[Tuple[int, Tuple[int, int]]] = [
        (0, (-1, 1)),
        (1, (-1, -1)),
        (2, (1, -1)),
        (3, (1, 1)),
    ]

    # The origin point's neighbors all have higher Manhattan distances (1)
    #   ^   |
    # <   > |
    #   v   |
    origin_point_class: List[int] = [0]

    def __init__(
        self,
        num_timesteps: int,
        lattice_data: str | Dict,  # type: ignore
        logger: Logger = getLogger("qlbm/"),
    ):
        super().__init__(lattice_data, logger)
        self.num_gridpoints, self.num_velocities, self.blocks = self.parse_input_data(
            lattice_data
        )  # type: ignore
        # Adding the length corrects for the fact that the parser subtracts 1
        # from the input to get the correct bit length.
        self.total_gridpoints: int = (
            sum(self.num_gridpoints) + len(self.num_gridpoints)
        ) * (sum(self.num_gridpoints) + len(self.num_gridpoints))
        self.num_velocities_per_point: int = sum(2**v for v in self.num_velocities)
        self.num_timesteps = num_timesteps
        self.num_grid_qubits: int = sum(
            num_gridpoints_in_dim.bit_length()
            for num_gridpoints_in_dim in self.num_gridpoints
        )
        self.num_velocity_qubits: int = self.num_required_velocity_qubits(num_timesteps)
        self.num_ancilla_qubits = 0
        self.num_total_qubits = (
            self.num_ancilla_qubits + self.num_grid_qubits + self.num_velocity_qubits
        )
        self.block_list: List[Block] = []
        temporary_registers = self.get_registers()
        (
            self.grid_registers,
            self.velocity_registers,
        ) = temporary_registers
        self.registers = tuple(flatten(temporary_registers))
        self.circuit = QuantumCircuit(*self.registers)
        self.extreme_point_indices, self.intermediate_point_indices = (
            self.get_neighbor_indices()
        )
        self.num_dimensions = len(self.num_gridpoints)

        logger.info(self.__str__())

    def grid_index(self, dim: int | None = None) -> List[int]:
        if dim is None:
            return list(range(self.num_grid_qubits))

        if dim >= self.num_dimensions or dim < 0:
            raise LatticeException(
                f"Cannot index grid register for dimension {dim} in {self.num_dimensions}-dimensional lattice."
            )

        previous_qubits = sum(self.num_gridpoints[d].bit_length() for d in range(dim))

        return list(
            range(
                previous_qubits, previous_qubits + self.num_gridpoints[dim].bit_length()
            )
        )

    def velocity_index(
        self,
        point_neighborhood_index: int,
        velocity_direction: int | None = None,
    ) -> List[int]:
        if velocity_direction is None:
            return list(
                range(
                    self.num_grid_qubits
                    + point_neighborhood_index * self.num_velocities_per_point,
                    self.num_grid_qubits
                    + (point_neighborhood_index + 1) * (self.num_velocities_per_point),
                )
            )
        return [
            self.num_grid_qubits
            + point_neighborhood_index * self.num_velocities_per_point
            + velocity_direction
        ]

    def grid_neighbors(
        self, coordinates: Tuple[int, int], up_to_distance: int
    ) -> List[List[int]]:
        return [
            [
                (coordinates[0] + x_offset) % (self.num_gridpoints[0] + 1),
                (coordinates[1] + y_offset) % (self.num_gridpoints[1] + 1),
            ]
            for x_offset in range(-up_to_distance, up_to_distance + 1)
            for y_offset in range(
                abs(x_offset) - up_to_distance, up_to_distance + 1 - abs(x_offset)
            )
        ]

    def num_required_velocity_qubits(
        self,
        num_timesteps: int,
    ) -> int:
        # Generalization of equation 17 of the paper
        # ! TODO generalize to 3D
        return min(
            self.total_gridpoints * self.num_velocities_per_point,
            int(
                self.num_velocities_per_point
                * self.num_velocities_per_point
                * num_timesteps
                * (num_timesteps + 1)
                * 0.5
                + self.num_velocities_per_point
            ),
        )

    def num_gridpoints_within_distance(self, manhattan_distance: int) -> int:
        return int(
            self.num_required_velocity_qubits(manhattan_distance)
            / self.num_velocities_per_point
        )

    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        # Grid qubits
        grid_registers = [
            QuantumRegister(gp.bit_length(), name=f"g_{dimension_letter(c)}")
            for c, gp in enumerate(self.num_gridpoints)
        ]

        # Velocity qubits
        velocity_registers = [
            QuantumRegister(
                self.num_required_velocity_qubits(
                    self.num_timesteps,
                ),  # The number of velocity qubits required at time t
                name="v",
            )
        ]

        return (grid_registers, velocity_registers)

    def von_neumann_neighbor_indices(
        self, manhattan_distance_from_origin: int
    ) -> List[int]:
        if manhattan_distance_from_origin == 0:
            return [0]

        total_neighbors = int(
            (
                self.num_required_velocity_qubits(manhattan_distance_from_origin)
                / self.num_velocities_per_point
            )
        )
        neighbors_lower_distance = int(
            (
                self.num_required_velocity_qubits(manhattan_distance_from_origin)
                / self.num_velocities_per_point
            )
        )

        return list(range(neighbors_lower_distance, total_neighbors))

    def get_neighbor_indices(
        self,
    ) -> Tuple[
        Dict[int, List[VonNeumannNeighbor]],
        Dict[int, Dict[int, List[VonNeumannNeighbor]]],
    ]:
        # ! This ONLY works for D2Q4!
        extreme_point_neighbor_indices: Dict[int, List[VonNeumannNeighbor]] = {}
        intermediate_point_neighbor_indices: Dict[
            int, Dict[int, List[VonNeumannNeighbor]]
        ] = {
            manhattan_distance: {}
            for manhattan_distance in range(2, self.num_timesteps + 1)
        }

        for manhattan_distance in range(1, self.num_timesteps + 1):
            total_neighbors_within_distance: int = self.num_gridpoints_within_distance(
                manhattan_distance
            )

            previous_extreme_point_neighbors: List[VonNeumannNeighbor] = (
                extreme_point_neighbor_indices[manhattan_distance - 1]
                if manhattan_distance > 1
                else (
                    [
                        VonNeumannNeighbor(
                            (0, 0),
                            1,
                            VonNeumannNeighborType.ORIGIN,
                        )
                        for _ in range(4)
                    ]
                )
            )

            extreme_point_neighbor_indices[manhattan_distance] = [
                VonNeumannNeighbor(
                    cast(
                        Tuple[int, int],
                        tuple(
                            c0 + c1
                            for c0, c1 in zip(
                                previous_extreme_point_neighbors[
                                    neighbor_class
                                ].coordinates_relative,
                                self.extreme_point_classes[neighbor_class][1],
                            )
                        ),
                    ),
                    extreme_point_index,
                    VonNeumannNeighborType.EXTREME,
                )
                for neighbor_class, extreme_point_index in enumerate(
                    range(
                        previous_extreme_point_neighbors[0].neighbor_index
                        + self.num_velocities_per_point * (manhattan_distance - 1),
                        total_neighbors_within_distance,
                        manhattan_distance,
                    )
                )
            ]

            if manhattan_distance < 2:
                continue

            intermediate_point_neighbor_indices[manhattan_distance] = {
                intermediate_point_class[0]: [
                    VonNeumannNeighbor(
                        cast(
                            Tuple[int, int],
                            tuple(
                                c0 + (relative_intermediate_point_index + 1) * c1
                                for c0, c1 in zip(
                                    extreme_point_neighbor_indices[manhattan_distance][
                                        neighbor_class
                                    ].coordinates_relative,
                                    self.intermediate_point_classes[neighbor_class][1],
                                )
                            ),
                        ),
                        absolute_intermediate_point_index,
                        VonNeumannNeighborType.INTERMEDIATE,
                    )
                    for relative_intermediate_point_index, absolute_intermediate_point_index in enumerate(
                        range(
                            extreme_point_neighbor_indices[manhattan_distance][
                                intermediate_point_class[0]
                            ].neighbor_index
                            + 1,
                            (
                                extreme_point_neighbor_indices[manhattan_distance][
                                    intermediate_point_class[0] + 1
                                ].neighbor_index
                                if neighbor_class
                                != len(self.intermediate_point_classes) - 1
                                else total_neighbors_within_distance
                            ),
                        )
                    )
                ]
                for neighbor_class, intermediate_point_class in enumerate(
                    self.intermediate_point_classes
                )
            }

        return extreme_point_neighbor_indices, intermediate_point_neighbor_indices

    def logger_name(self) -> str:
        gp_string = ""
        for c, gp in enumerate(self.num_gridpoints):
            gp_string += f"{gp+1}"
            if c < len(self.num_gridpoints) - 1:
                gp_string += "x"
        return f"{self.num_dimensions}d-{gp_string}-q4"
