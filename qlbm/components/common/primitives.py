from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit

from qlbm.components.base import LBMPrimitive
from qlbm.lattice import Lattice


class EmptyPrimitive(LBMPrimitive):
    def __init__(
        self,
        lattice: Lattice,
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

    def create_circuit(self) -> QuantumCircuit:
        return QuantumCircuit(*self.lattice.registers)

    def __str__(self) -> str:
        return f"[Primitive EmptyPrimitive with lattice {self.lattice}]"
