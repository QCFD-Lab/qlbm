from logging import Logger, getLogger
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, XGate

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice, VonNeumannNeighbor
from qlbm.tools.utils import bit_value, flatten


class SpaceTimeInitialConditions(LBMPrimitive):
    def __init__(
        self,
        lattice: SpaceTimeLattice,
        grid_data: List[Tuple[Tuple[int, int], Tuple[bool, bool, bool, bool]]] = [
            ((2, 5), (True, True, True, True)),
            ((3, 4), (False, True, False, True)),
        ],
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)

        self.lattice = lattice
        self.grid_data = grid_data

        self.circuit = self.create_circuit()

    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        circuit.h(self.lattice.grid_index())

        # Set the state for the origin
        for grid_point_data in self.grid_data:
            circuit.compose(self.set_grid_value(grid_point_data[0]), inplace=True)
            circuit.compose(
                MCMT(
                    XGate(),
                    num_ctrl_qubits=self.lattice.num_grid_qubits,
                    num_target_qubits=sum(
                        grid_point_data[1]
                    ),  # The sum is equal to the number of velocities set to true
                ),
                qubits=list(
                    self.lattice.grid_index()
                    + flatten(
                        [
                            self.lattice.velocity_index(0, c)
                            for c, is_velocity_enabled in enumerate(grid_point_data[1])
                            if is_velocity_enabled
                        ]
                    )
                ),
                inplace=True,
            )
            circuit.compose(self.set_grid_value(grid_point_data[0]), inplace=True)

            # Set the velocity state for neighbors in increasing velocity
            for manhattan_distance in range(1, self.lattice.num_timesteps + 1):
                for neighbor in self.lattice.extreme_point_indices[manhattan_distance]:
                    circuit.compose(
                        self.set_neighbor_velocity(
                            grid_point_data[0], grid_point_data[1], neighbor
                        ),
                        inplace=True,
                    )

                # No intermediate points at Manhattan distance 1
                if manhattan_distance < 2:
                    continue

                for neighbor in flatten(
                    list(
                        self.lattice.intermediate_point_indices[
                            manhattan_distance
                        ].values()
                    )
                ):
                    circuit.compose(
                        self.set_neighbor_velocity(
                            grid_point_data[0], grid_point_data[1], neighbor
                        ),
                        inplace=True,
                    )

        return circuit

    def set_grid_value(self, point_coordinates: Tuple[int, int]) -> QuantumCircuit:
        # ! TODO: rename, refactor into primitive
        circuit = self.lattice.circuit.copy()

        for dim, num_gp in enumerate(self.lattice.num_gridpoints):
            for qubit_index in range(num_gp.bit_length()):
                if not bit_value(point_coordinates[dim], qubit_index):
                    circuit.x(self.lattice.grid_index(dim)[0] + qubit_index)

        return circuit

    def set_neighbor_velocity(
        self,
        point_coordinates: Tuple[int, int],
        velocity_values: Tuple[bool, bool, bool, bool],
        neighbor: VonNeumannNeighbor,
    ) -> QuantumCircuit:
        # ! TODO: rename, refactor into primitive
        circuit = self.lattice.circuit.copy()
        absolute_neighbor_coordinates = neighbor.get_absolute_values(
            point_coordinates, relative=False
        )
        circuit.compose(
            self.set_grid_value(absolute_neighbor_coordinates), inplace=True
        )
        circuit.compose(
            MCMT(
                XGate(),
                num_ctrl_qubits=self.lattice.num_grid_qubits,
                num_target_qubits=sum(
                    velocity_values
                ),  # The sum is equal to the number of velocities set to true
            ),
            inplace=True,
            qubits=list(
                self.lattice.grid_index()
                + flatten(
                    [
                        self.lattice.velocity_index(neighbor.neighbor_index, c)
                        for c, is_velocity_enabled in enumerate(velocity_values)
                        if is_velocity_enabled
                    ]
                )
            ),
        )
        circuit.compose(
            self.set_grid_value(absolute_neighbor_coordinates), inplace=True
        )

        return circuit

    def __str__(self) -> str:
        return f"[Primitive SpaceTimeInitialConditions with data={self.grid_data} and lattice={self.lattice}]"
