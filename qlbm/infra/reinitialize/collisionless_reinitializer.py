""":class:`.CQLBM`-specific implementation of the :class:`.Reinitializer`."""

from logging import Logger, getLogger

from qiskit import QuantumCircuit as QiskitQC
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector
from qiskit.result import Counts
from qiskit_aer.backends.aer_simulator import AerBackend
from qulacs import QuantumCircuit as QulacsQC
from typing_extensions import override

from qlbm.infra.compiler import CircuitCompiler
from qlbm.lattice import CollisionlessLattice

from .base import Reinitializer


class CollisionlessReinitializer(Reinitializer):
    r"""
    :class:`.CQLBM`-specific implementation of the :class:`.Reinitializer`.

    Compatible with both :class:`.QiskitRunner`\ s and :class:`.QulacsRunner`\ s.
    To generate a new set of initial conditions for the CQLBM algorithm,
    the reinitializer simply returns the quantum state computed
    at the end of the previous simulation.
    This allows the reuse of a single quantum circuit for the simulation
    of arbitrarily many time steps.
    No copy of the statevector is required.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.CollisionlessLattice` of the simulated system.
    :attr:`compiler`            The compiler that converts the novel initial conditions circuits.
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``
    =========================== ======================================================================
    """

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
        """
        Returns the provided ``statevector`` as a new Qiskit ``Initialize`` object that can be prepended to the time step circuit to resume simulation.

        Parameters
        ----------
        statevector : Statevector
            The statevector at the end of the simulation.
        counts : Counts
            Ignored.
        backend : AerBackend | None
            Ignored.
        optimization_level : int, optional
            Ignored.

        Returns
        -------
        QiskitQC | QulacsQC
            A Qiskit ``Initialize`` object.
        """
        circuit = self.lattice.circuit.copy()
        circuit.compose(
            Initialize(statevector),
            inplace=True,
            qubits=range(circuit.num_qubits),
        )
        return circuit

    @override
    def requires_statevector(self) -> bool:
        return True
