"""WIP."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMOperator
from qlbm.components.common.primitives import TruncatedQFT
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException


class ABEAveragedCollisionOperator(LBMOperator):
    """WIP."""

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        if self.lattice.discretization == LatticeDiscretization.D1Q3:
            return self.__create_circuit_d1q3()

        raise LatticeException("ABE only currently supported in D1Q3")

    def __create_circuit_d1q3(self):
        circuit = self.lattice.circuit.copy()

        circuit.compose(
            TruncatedQFT(
                self.lattice.num_velocity_qubits,
                self.lattice.num_velocities_per_point,
                self.logger,
            ).circuit,
            qubits=self.lattice.velocity_index(),
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Operator ABEAveragedCollision with lattice {self.lattice}]"
