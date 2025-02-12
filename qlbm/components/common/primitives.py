"""Common primitives used for multiple encodings."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, XGate
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


class MCSwap(LBMPrimitive):
    """
    Decomposition of a Multi-Controlled Swap Gate into 1 multi-controlled :math:`X` gate and 2 single-controlled :math:`X` gates.

    Decomposition taken from :cite:t:`mcswap`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` or :class:`.SpaceTimeLattice` based on which the number of qubits is inferred.
    :attr:`control_qubits`    The qubits that control the swap gate.
    :attr:`target_qubits`     The two qubits to be swapped.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
    """

    def __init__(
        self,
        lattice: Lattice,
        control_qubits: List[int],
        target_qubits: Tuple[int, int],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.lattice = lattice
        self.control_qubits = control_qubits
        self.target_qubits = target_qubits

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        circuit.cx(self.target_qubits[1], self.target_qubits[0])
        circuit.compose(
            MCMT(XGate(), len(self.control_qubits) + 1, len(self.target_qubits) - 1),
            qubits=self.control_qubits + list(self.target_qubits),
            inplace=True,
        )
        circuit.cx(self.target_qubits[1], self.target_qubits[0])

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive MCSwap with lattice {self.lattice}]"
