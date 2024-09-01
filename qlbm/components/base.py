from abc import ABC, abstractmethod
from io import TextIOBase
from logging import Logger, getLogger
from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.qasm2 import dump as dump_qasm2
from qiskit.qasm3 import dump as dump_qasm3

from qlbm.lattice import CollisionlessLattice, Lattice, SpaceTimeLattice


class QuantumComponent(ABC):
    circuit: QuantumCircuit
    logger: Logger

    def __init__(
        self,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__()
        self.logger = logger

    @abstractmethod
    def create_circuit(self) -> QuantumCircuit:
        pass

    def __repr__(self) -> str:
        return self.circuit.__repr__()

    @abstractmethod
    def __str__(self) -> str:
        return self.circuit.__str__()

    def circ_dim(self) -> Tuple[int, int]:
        return len(self.circuit.qubits), len(self.circuit.qubits)

    def width(self) -> int:
        return self.circuit.width()

    def size(self) -> int:
        return self.circuit.size()

    def dump_qasm3(self, stream: TextIOBase) -> None:
        return dump_qasm3(self.circuit, stream)

    def dump_qasm2(self, stream: TextIOBase) -> None:
        return dump_qasm2(self.circuit, stream)

    def draw(self, output: str, filename: str | None = None):  # type: ignore
        return self.circuit.draw(output=output, filename=filename)  # type: ignore


class LBMPrimitive(QuantumComponent):
    logger: Logger

    def __init__(
        self,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)


class LBMOperator(QuantumComponent):
    lattice: Lattice

    def __init__(
        self,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice


class CQLBMOperator(LBMOperator):
    lattice: CollisionlessLattice

    def __init__(
        self,
        lattice: CollisionlessLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice


class SpaceTimeOperator(LBMOperator):
    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice


class LBMAlgorithm(QuantumComponent):
    lattice: Lattice

    def __init__(
        self,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
