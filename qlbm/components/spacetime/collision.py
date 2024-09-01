from logging import Logger, getLogger
from math import pi
from time import perf_counter_ns

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import MCMT, RYGate

from qlbm.components.base import SpaceTimeOperator
from qlbm.lattice import SpaceTimeLattice


class SpaceTimeCollisionOperator(SpaceTimeOperator):
    def __init__(
        self,
        lattice: SpaceTimeLattice,
        timestep: int,
        gate_to_apply: Gate = RYGate(pi / 2),
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice
        self.timestep = timestep
        self.gate_to_apply = gate_to_apply

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        collision_circuit = self.local_collision_circuit(reset_state=False)
        collision_circuit.compose(
            MCMT(
                self.gate_to_apply,
                self.lattice.num_velocities_per_point - 1,
                1,
            ),
            qubits=list(range(1, self.lattice.num_velocities_per_point)) + [0],
            inplace=True,
        )

        # Create the local circuit once
        collision_circuit.compose(
            self.local_collision_circuit(reset_state=True), inplace=True
        )

        # Append the collision circuit at each step
        for velocity_qubit_indices in range(
            self.lattice.num_grid_qubits,
            self.lattice.num_grid_qubits
            + self.lattice.num_required_velocity_qubits(self.timestep),
            self.lattice.num_velocities_per_point,
        ):
            circuit.compose(
                collision_circuit,
                inplace=True,
                qubits=range(velocity_qubit_indices, velocity_qubit_indices + 4),
            )
        return circuit

    def local_collision_circuit(self, reset_state: bool) -> QuantumCircuit:
        circuit = QuantumCircuit(self.lattice.num_velocities_per_point)

        if not reset_state:
            circuit.cx(control_qubit=0, target_qubit=2)
            circuit.x(0)
            circuit.cx(control_qubit=1, target_qubit=3)
            circuit.cx(control_qubit=0, target_qubit=1)
            circuit.x(list(range(circuit.num_qubits)))
        # Same circuit, but mirrored
        else:
            circuit.x(list(range(circuit.num_qubits)))
            circuit.cx(control_qubit=0, target_qubit=1)
            circuit.cx(control_qubit=1, target_qubit=3)
            circuit.x(0)
            circuit.cx(control_qubit=0, target_qubit=2)

        return circuit

    def __str__(self) -> str:
        # TODO: Implement
        return "Space Time Collision Operator"
