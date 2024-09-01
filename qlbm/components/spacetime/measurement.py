from logging import Logger, getLogger

from qiskit import ClassicalRegister

from qlbm.components.base import SpaceTimeOperator
from qlbm.lattice import SpaceTimeLattice


class SpaceTimeGridVelocityMeasurement(SpaceTimeOperator):
    def __init__(
        self,
        lattice: SpaceTimeLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice

        self.circuit = self.create_circuit()

    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        qubits_to_measure = self.lattice.grid_index() + self.lattice.velocity_index(0)
        circuit.add_register(
            ClassicalRegister(
                self.lattice.num_grid_qubits + self.lattice.num_velocities_per_point
            )
        )

        circuit.measure(
            qubits_to_measure,
            list(range(len(qubits_to_measure))),
        )

        return circuit

    def __str__(self) -> str:
        # TODO: Implement
        return "Space Gird Measurement"
