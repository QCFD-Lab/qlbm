from logging import Logger, getLogger
from typing import Dict, List, Tuple, cast

from qiskit import QuantumRegister

from qlbm.lattice.blocks import Block
from qlbm.tools.utils import dimension_letter

from .builder_base import (
    LatticeDiscretization,
    SpaceTimeLatticeBuilder,
    VonNeumannNeighbor,
    VonNeumannNeighborType,
)


class D2Q4SpaceTimeLatticeBuilder(SpaceTimeLatticeBuilder):
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
        num_gridpoints: List[int],
        blocks: Dict[str, List[Block]],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(num_timesteps, logger)
        self.num_gridpoints = num_gridpoints
        self.blocks = blocks

    def get_discretization(self) -> LatticeDiscretization:
        return LatticeDiscretization.D2Q4

    def get_num_velocities_per_point(self) -> int:
        return 4

    def get_num_ancilla_qubits(self) -> int:
        return 0

    def get_num_grid_qubits(self) -> int:
        return sum(
            num_gridpoints_in_dim.bit_length()
            for num_gridpoints_in_dim in self.num_gridpoints
        )

    def get_num_velocity_qubits(self, num_timesteps: int | None = None) -> int:
        total_gridpoints = (sum(self.num_gridpoints) + len(self.num_gridpoints)) * (
            sum(self.num_gridpoints) + len(self.num_gridpoints)
        )
        velocities_per_gp = self.get_num_velocities_per_point()

        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        return min(
            total_gridpoints * velocities_per_gp,
            int(
                velocities_per_gp
                * velocities_per_gp
                * num_timesteps
                * (num_timesteps + 1)
                * 0.5
                + velocities_per_gp
            ),
        )

    def get_num_total_qubits(self) -> int:
        return (
            self.get_num_ancilla_qubits()
            + self.get_num_grid_qubits()
            + self.get_num_velocity_qubits()
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
                self.get_num_velocity_qubits(
                    self.num_timesteps,
                ),  # The number of velocity qubits required at time t
                name="v",
            )
        ]

        return (grid_registers, velocity_registers)

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

    def get_index_of_neighbor(self, distance: Tuple[int, ...]) -> int:
        if distance[0] == 0 and distance[1] == 0:
            return 0

        distance_from_origin = sum(map(abs, distance))
        is_extreme_point = distance[0] == 0 or distance[1] == 0
        num_points_lower_distances = self.get_num_gridpoints_within_distance(
            distance_from_origin - 1
        )
        quadrant: int = self.coordinates_to_quadrant(distance)  # type: ignore

        if is_extreme_point:
            return num_points_lower_distances + quadrant * distance_from_origin
        else:
            ordering_dim = 0 if quadrant in [1, 3] else 1
            distance_across_ordering_dim = abs(distance[ordering_dim])
            return (
                num_points_lower_distances
                + quadrant * distance_from_origin
                + distance_across_ordering_dim
            )

    def get_streaming_lines(
        self, dimension: int, direction: bool, timestep: int | None = None
    ) -> List[List[int]]:
        neighbors_in_line = []
        if timestep is None:
            timestep = self.num_timesteps
        for offset in range(-timestep + 1, timestep):
            start = -self.num_timesteps + abs(offset)
            end = self.num_timesteps - abs(offset)
            step = 1

            if (dimension == 0 and direction) or (dimension == 1 and not direction):
                start, end = end, start
                step *= -1

            neighbors_in_line.append(
                [
                    self.get_index_of_neighbor(
                        (
                            i if dimension == 0 else offset,
                            offset if dimension == 0 else i,
                        )
                    )
                    for i in range(start, end + step, step)
                ]
            )
        return neighbors_in_line

    def get_neighbor_indices(self):
        extreme_point_neighbor_indices: Dict[int, List[VonNeumannNeighbor]] = {}
        intermediate_point_neighbor_indices: Dict[
            int, Dict[int, List[VonNeumannNeighbor]]
        ] = {
            manhattan_distance: {}
            for manhattan_distance in range(2, self.num_timesteps + 1)
        }

        for manhattan_distance in range(1, self.num_timesteps + 1):
            total_neighbors_within_distance: int = (
                self.get_num_gridpoints_within_distance(manhattan_distance)
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
                        + self.get_num_velocities_per_point()
                        * (manhattan_distance - 1),
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

    def coordinates_to_quadrant(self, distance: Tuple[int, int]) -> int:
        """
        Maps a given point to the quadrant it belongs to.
        Quadrants are ordered in counterclockwise fashion, starting on the top right at 0:
         1 | 0
        _______
         2 | 3

        Points along lines that intersect with the origin (i.e., (0, 5), (-8, 0)) belong to quadrants as well:

        #. :math:`x>0,y=0 \to q_0`
        #. :math:`x=0,y>0 \to q_1`
        #. :math:`x=<0,y=0 \to q_2`
        #. :math:`x=0,y<0 \to q_3`

        Parameters
        ----------
        distance : Tuple[int, int]
            The coordinates of the point to place in a quadrant, expressed as its distance from the origin in the x and y axes.

        Returns
        -------
        int
            The quadrant the point belongs to.
        """

        if distance[1] == 0:
            if distance[0] > 0:
                return 0
            return 2

        if distance[0] == 0:
            if distance[1] > 0:
                return 1
            return 3

        if distance[1] > 0:
            if distance[0] > 0:
                return 0
            return 1

        if distance[0] < 0:
            return 2
        return 3
