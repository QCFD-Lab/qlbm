"""Collision operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LQLGAOperator, SpaceTimeOperator
from qlbm.components.common.cbse_collision.cbse_collision import EQCCollisionOperator
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice


class GenericLQLGACollisionOperator(LQLGAOperator):
    def __init__(
        self,
        lattice: LQLGALattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        local_collision_circuit = EQCCollisionOperator(
            self.lattice.discretization
        ).circuit
        circuit = self.lattice.circuit.copy()

        for velocity_qubit_indices in range(
            0,
            self.lattice.num_base_qubits,
            self.lattice.num_velocities_per_point,
        ):
            circuit.compose(
                local_collision_circuit,
                inplace=True,
                qubits=range(
                    velocity_qubit_indices,
                    velocity_qubit_indices + self.lattice.num_velocities_per_point,
                ),
            )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[GenericSpaceTimeCollisionOperator for discretization {self.lattice.discretization}]"
