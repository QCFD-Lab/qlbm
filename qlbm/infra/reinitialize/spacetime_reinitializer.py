from logging import Logger, getLogger
from typing import List, Tuple, cast

from qiskit import QuantumCircuit as QiskitQC
from qiskit.quantum_info import Statevector
from qiskit.result import Counts
from qiskit_aer.backends.aer_simulator import AerBackend
from qulacs import QuantumCircuit as QulacsQC

from qlbm.components.spacetime import SpaceTimeInitialConditions
from qlbm.infra.compiler import CircuitCompiler
from qlbm.lattice import SpaceTimeLattice

from .base import Reinitializer


class SpaceTimeReinitializer(Reinitializer):
    """
    :class:`.SpaceTimeQLBM`-specific implementation of the :class:`.Reinitializer`.
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
    :attr:`lattice`             The :class:`.SpaceTimeLattice` of the simulated system.
    :attr:`compiler`            The compiler that converts the novel initial conditions circuits.
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``
    =========================== ======================================================================
    """

    lattice: SpaceTimeLattice
    counts: Counts

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        compiler: CircuitCompiler,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice, compiler, logger)
        self.lattice = lattice
        self.logger = logger
        self.x_grid_qubits = self.lattice.num_gridpoints[0].bit_length()
        self.y_grid_qubits = self.lattice.num_gridpoints[1].bit_length()

    def reinitialize(
        self,
        statevector: Statevector,
        counts: Counts,
        backend: AerBackend | None,
        optimization_level: int = 0,
    ) -> QiskitQC | QulacsQC:
        """
        Converts the input ``counts`` into a new :class:`.SpaceTimeInitialConditions`
        object that can be prepended to the time step circuit to resume simulation.

        Parameters
        ----------
        statevector : Statevector
            Ignored.
        counts : Counts
            The counts obtained from :class:`.SpacetimeGridVelocityMeasurement` at the end of the simulation.
        backend : AerBackend | None
            The backend used for simulation.
        optimization_level : int, optional
            The compiler optimization level.

        Returns
        -------
        QiskitQC | QulacsQC
            The suitably compiles initial conditions circuit.
        """
        return self.compiler.compile(
            SpaceTimeInitialConditions(
                self.lattice, self.counts_to_velocity_pairs(counts)
            ),
            backend=backend,
            optimization_level=optimization_level,
        )

    def counts_to_velocity_pairs(
        self,
        counts: Counts,
    ) -> List[Tuple[Tuple[int, int], Tuple[bool, bool, bool, bool]]]:
        """
        Converts all counts into their grid and velocity components.

        Parameters
        ----------
        counts : Counts
            The Qiskit ``Count`` output of the simulation.

        Returns
        -------
        List[Tuple[Tuple[int, int], Tuple[bool, bool, bool, bool]]]
            The input counts split into their grid position and velocity profile.
        """
        return [self.split_count(count) for count in counts if int(count[:4], 2) > 0]

    def split_count(
        self, count: str
    ) -> Tuple[Tuple[int, int], Tuple[bool, bool, bool, bool]]:
        """
        Splits a given ``Count`` into its position and velocity components.
        Counts are assumed to be obtained from :class:`.SpacetimeGridVelocityMeasurement` objects,
        and split format is the same as the input to :class:`.SpaceTimeInitialConditions`.

        Parameters
        ----------
        count : str
            The Qiskit ``Count`` output of the simulation.

        Returns
        -------
        Tuple[Tuple[int, int], Tuple[bool, bool, bool, bool]]
            The input count split into its grid position and velocity profile.
        """
        inverse_count = count[::-1]
        return (
            (
                int(inverse_count[: self.x_grid_qubits][::-1], 2),
                int(
                    inverse_count[
                        self.x_grid_qubits : self.x_grid_qubits + self.y_grid_qubits
                    ][::-1],
                    2,
                ),
            ),
            cast(
                Tuple[bool, bool, bool, bool],
                tuple(
                    bool(int(x, 2))
                    for x in inverse_count[self.x_grid_qubits + self.y_grid_qubits :]
                ),
            ),
        )

    def requires_statevector(self) -> bool:
        return False
