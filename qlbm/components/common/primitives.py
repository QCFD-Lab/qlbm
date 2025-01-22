"""Common primitives used for multiple encodings."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice import Lattice


class EmptyPrimitive(LBMPrimitive):
    """
    Empty primitive used for effectively not specifying parts of the QLBM algorithm.

    Useful in situations where testing the end-to-end implementation of the algorithm
    where one part of the algorithm is left out or not yet implemented.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` or :class:`.SpaceTimeLattice` based on which the number of qubits is inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
    """

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

    @override
    def create_circuit(self) -> QuantumCircuit:
        return QuantumCircuit(*self.lattice.registers)

    @override
    def __str__(self) -> str:
        return f"[Primitive EmptyPrimitive with lattice {self.lattice}]"
