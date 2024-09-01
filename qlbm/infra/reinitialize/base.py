from abc import ABC, abstractmethod
from logging import Logger, getLogger

from qiskit import QuantumCircuit as QiskitQC
from qiskit.quantum_info import Statevector
from qiskit.result import Counts
from qiskit_aer.backends.aerbackend import AerBackend
from qulacs import QuantumCircuit as QulacsQC

from qlbm.infra.compiler import CircuitCompiler
from qlbm.lattice import Lattice


class Reinitializer(ABC):
    lattice: Lattice

    def __init__(
        self,
        lattice: Lattice,
        compiler: CircuitCompiler,
        logger: Logger = getLogger("qlbm"),
    ):
        self.lattice = lattice
        self.compiler = compiler
        self.logger = logger

    @abstractmethod
    def reinitialize(
        self,
        statevector: Statevector,
        counts: Counts,
        backend: AerBackend | None,
        optimization_level: int = 0,
    ) -> QiskitQC | QulacsQC:
        pass

    @abstractmethod
    def requires_statevector(self) -> bool:
        pass
