from logging import Logger, getLogger

from qiskit import QuantumCircuit as QiskitQC
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector
from qiskit.result import Counts
from qiskit_aer.backends.aer_simulator import AerBackend
from qulacs import QuantumCircuit as QulacsQC

from qlbm.infra.compiler import CircuitCompiler
from qlbm.lattice import CollisionlessLattice

from .base import Reinitializer


class CollisionlessReinitializer(Reinitializer):
    lattice: CollisionlessLattice
    statevector: Statevector
    counts: Counts

    def __init__(
        self,
        lattice: CollisionlessLattice,
        compiler: CircuitCompiler,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice, compiler, logger)
        self.lattice = lattice
        self.logger = logger

    def reinitialize(
        self,
        statevector: Statevector,
        counts: Counts,
        backend: AerBackend | None = None,
        optimization_level: int = 0,
    ) -> QiskitQC | QulacsQC:
        circuit = self.lattice.circuit.copy()
        circuit.compose(
            Initialize(statevector),
            inplace=True,
            qubits=range(circuit.num_qubits),
        )
        return circuit

    def requires_statevector(self) -> bool:
        return True
