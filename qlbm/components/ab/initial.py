from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.components.common.primitives import TruncatedQFT
from qlbm.lattice.lattices.abe_lattice import ABLattice


class ABEInitialConditions(LBMPrimitive):
    """TODO."""

    def __init__(
        self,
        lattice: ABLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

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
        return f"[Primitive ABEInitialConditions with lattice {self.lattice}]"
