""":math:`D_3Q_6` STQBM builder."""

from itertools import product
from logging import Logger, getLogger
from typing import Dict, List, Tuple

from qiskit import QuantumRegister
from typing_extensions import override

from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    SpaceTimeLatticeBuilder,
)
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter


class D3Q6SpaceTimeLatticeBuilder(SpaceTimeLatticeBuilder):
    """
    :math:`D_3Q_6` lattice builder.

    WIP.
    """

    velocity_reflection: Dict[int, int] = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}

    def __init__(
        self,
        num_timesteps: int,
        num_gridpoints: List[int],
        include_measurement_qubit: bool = False,
        use_volumetric_ops: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(
            num_timesteps,
            include_measurement_qubit=include_measurement_qubit,
            use_volumetric_ops=use_volumetric_ops,
            logger=logger,
        )
        self.num_gridpoints = num_gridpoints

    @override
    def get_discretization(self) -> LatticeDiscretization:
        return LatticeDiscretization.D3Q6

    @override
    def get_num_velocities_per_point(self) -> int:
        return 6

    @override
    def get_num_ancilla_qubits(self) -> int:
        return (6 if self.use_volumetric_ops else 0) + (
            1 if self.include_measurement_qubit else 0
        )

    @override
    def get_num_grid_qubits(self) -> int:
        return sum(
            num_gridpoints_in_dim.bit_length()
            for num_gridpoints_in_dim in self.num_gridpoints
        )

    @override
    def get_num_previous_grid_qubits(self, dim: int) -> int:
        # ! TODO add exception
        return sum(self.num_gridpoints[i].bit_length() for i in range(dim))

    @override
    def get_num_velocity_qubits(self, num_timesteps: int | None = None) -> int:
        total_gridpoints = (
            (sum(self.num_gridpoints) + len(self.num_gridpoints))
            * (sum(self.num_gridpoints) + len(self.num_gridpoints))
            * (sum(self.num_gridpoints) + len(self.num_gridpoints))
        )
        velocities_per_gp = self.get_num_velocities_per_point()

        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        return min(
            total_gridpoints * velocities_per_gp,
            int(
                8 * velocities_per_gp * velocities_per_gp * velocities_per_gp
                + 12 * velocities_per_gp * velocities_per_gp
                + 16 * velocities_per_gp
                + velocities_per_gp
            ),
        )

    @override
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

        ancilla_measurement_register = (
            [QuantumRegister(1, "a_m")] if self.include_measurement_qubit else []
        )

        ancilla_comparator_registers = (
            [
                QuantumRegister(1, f"a_{bound}{dimension_letter(dim)}")
                for dim, bound in product([0, 1], ["l", "u"])
            ]
            if self.use_volumetric_ops
            else []
        )

        # Ancilla qubits
        ancilla_registers = ancilla_measurement_register + ancilla_comparator_registers

        return (grid_registers, velocity_registers, ancilla_registers)

    @override
    def get_index_of_neighbor(self, distance: Tuple[int, ...]) -> int:
        raise LatticeException("Not implemented")

    @override
    def get_streaming_lines(
        self, dimension: int, direction: bool, timestep: int | None = None
    ) -> List[List[int]]:
        raise LatticeException("Not implemented")

    @override
    def get_neighbor_indices(self):
        # ! TODO!!!
        return [], []

    @override
    def get_reflected_index_of_velocity(self, velocity_index: int) -> int:
        if velocity_index not in list(range(4)):
            raise LatticeException(
                f"D3Q6 discretization only supports velocities with indices 0-5. Index {velocity_index} unsupported."
            )

        return self.velocity_reflection[velocity_index]

    @override
    def get_reflection_increments(self, velocity_index: int) -> Tuple[int, ...]:
        if velocity_index not in list(range(4)):
            raise LatticeException(
                f"D3Q6 discretization only supports velocities with indices 0-5. Index {velocity_index} unsupported."
            )

        raise LatticeException("Not implemented")
