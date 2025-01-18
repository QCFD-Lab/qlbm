"""Base class for all algorithm-specific reinitializers."""

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
    """
    Base class for all algorithm-specific reinitializers.

    A ``Reinitializer`` uses the information at information available
    at the end of the simulation of 1 or more time steps
    to new initial conditions for the following time steps.
    Such information includes the quantum state and counts extracted from it.
    Novel initial conditions are inferred automatically based on
    the requirements of the algorithm under simulation, and
    an on-the-fly :class:`.CircuitCompiler` automatically converts
    them to the appropriate format to enable compatibility with the already transpiled circuits.
    For convenience, all reinitializers provide a uniform :meth:`reinitialize` interface,
    which takes as input both the quantum state and the counts performed during simulation.
    Its implementation may choose to ignore one of those inputs, depending on the
    algorithm and implementation.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.Lattice` of the simulated system.
    :attr:`compiler`            The compiler that converts the novel initial conditions circuits.
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``
    =========================== ======================================================================
    """

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
        """
        Parses the input statevector and counts, constructs a new initial conditions circuit, and transpiles it to the given backend.

        Parameters
        ----------
        statevector : Statevector
            The statevector at the end of the simulation.
        counts : Counts
            The counts extracted from the statevector at the end of the simulation.
        backend : AerBackend | None
            The backend to compile to.k
        optimization_level : int, optional
            The optimization level to pass to the circuit compiler, by default 0.

        Returns
        -------
        QiskitQC | QulacsQC
            The compiled initial conditions circuit to use for the next time step.
        """
        pass

    @abstractmethod
    def requires_statevector(self) -> bool:
        """
        Whether the reinitializer requires a copy of the statevector.

        Omotting the statevector may significantly increase the perfomance of reinitialization.

        Returns
        -------
        bool
            Whether a statevector is needed.
        """
        pass
