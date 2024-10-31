from logging import Logger, getLogger
from typing import Dict, List, Tuple, cast

from qiskit import QuantumRegister

from qlbm.lattice.blocks import Block
from qlbm.lattice.lattices.spacetime.builder_base import SpaceTimeLatticeBuilder
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter

from .builder_base import (
    LatticeDiscretization,
    SpaceTimeLatticeBuilder,
    VonNeumannNeighbor,
    VonNeumannNeighborType,
)


class D1Q2SpaceTimeLatticeBuilder(SpaceTimeLatticeBuilder):
    # Points to the right of the origin are marked as 0
    # And points to the left are marked as 1
    # This abstraction is more useful for more complex
    # Stencils in higher dimensions
    extreme_point_classes: List[Tuple[int, Tuple[int, int]]] = [
        (0, (1, 0)),
        (1, (-1, 0)),
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
        return LatticeDiscretization.D1Q2

    def get_num_velocities_per_point(self) -> int:
        return 2

    def get_num_ancilla_qubits(self) -> int:
        return 0

    def get_num_grid_qubits(self) -> int:
        return self.num_gridpoints[0].bit_length()

    def get_num_velocity_qubits(self, num_timesteps: int | None = None) -> int:
        total_gridpoints = self.num_gridpoints[0] + 1
        velocities_per_gp = self.get_num_velocities_per_point()

        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        return min(
            total_gridpoints * velocities_per_gp,
            velocities_per_gp * num_timesteps * 2 + velocities_per_gp,
        )

    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        # Grid qubits
        grid_registers = [
            QuantumRegister(self.num_gridpoints[0].bit_length(), name="g_x")
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

    def get_index_of_neighbor(self, distance: Tuple[int, ...]) -> int:
        if (d := distance[0]) == 0:
            return 0

        return self.get_num_gridpoints_within_distance(abs(d) - 1) + (d < 0)

    def get_streaming_lines(
        self, dimension: int, direction: bool, timestep: int | None = None
    ) -> List[List[int]]:
        if dimension != 0:
            raise LatticeException(
                f"Dimension {dimension} unsupported, D1Q2 lattices only support dimension 0."
            )

        step = 1 if direction else -1

        if timestep is None:
            timestep = self.num_timesteps

        start, end, step = (
            (timestep, -timestep, -1)
            if direction
            else (
                -timestep,
                timestep,
                1,
            )
        )

        return [
            [self.get_index_of_neighbor((i, 0)) for i in range(start, end + step, step)]
        ]

    def get_neighbor_indices(
        self,
    ) -> Tuple[
        Dict[int, List[VonNeumannNeighbor]],
        Dict[int, Dict[int, List[VonNeumannNeighbor]]],
    ]:
        extreme_point_neighbor_indices: Dict[int, List[VonNeumannNeighbor]] = {}

        for manhattan_distance in range(1, self.num_timesteps + 1):
            previous_extreme_point_neighbors: List[VonNeumannNeighbor] = (
                extreme_point_neighbor_indices[manhattan_distance - 1]
                if manhattan_distance > 1
                else (
                    [
                        VonNeumannNeighbor(
                            (0, 0),
                            0,
                            VonNeumannNeighborType.ORIGIN,
                        )
                        for _ in range(2)
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
                        previous_extreme_point_neighbors[-1].neighbor_index + 1,
                        previous_extreme_point_neighbors[-1].neighbor_index + 3,
                    )
                )
            ]

        return extreme_point_neighbor_indices, {}
