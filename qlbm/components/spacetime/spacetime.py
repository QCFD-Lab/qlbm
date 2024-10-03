from logging import Logger, getLogger

from qiskit import QuantumCircuit

from qlbm.components.base import LBMAlgorithm
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice

from .collision import SpaceTimeCollisionOperator
from .streaming import SpaceTimeStreamingOperator


class SpaceTimeQLBM(LBMAlgorithm):
    """
    The end-to-end algorithm of the Space-Time Quantum Lattice Boltzmann Algorithm
    described in :cite:`spacetime`.
    This implementation currently only supports 1 time step on the :math:`D_2Q_4`
    lattice discretization.
    Additional steps are possible by means of reinitialization.

    The algorithm is composed of two main steps, the implementation of which (in general) varies per individual time step:

    #. Streaming performed by the :class:`.SpaceTimeStreamingOperator` moves the particles on the grid by means of swap gates over velocity qubits.
    #. Collision performed by the :class:`.SpaceTimeCollisionOperator` does not move particles on the grid, but locally alters the velocity qubits at each grid point, if applicable.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.spacetime import SpaceTimeQLBM
        from qlbm.lattice import SpaceTimeLattice

        # Build an example lattice
        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            },
        )

        # Draw the end-to-end algorithm for 1 time step
        SpaceTimeQLBM(lattice=lattice).draw("mpl")
    """

    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice, logger)

        self.lattice = lattice

        self.circuit = self.create_circuit()

    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        circuit.compose(
            SpaceTimeStreamingOperator(self.lattice, 1, self.logger).circuit,
            inplace=True,
        )

        circuit.compose(
            SpaceTimeCollisionOperator(self.lattice, 1, logger=self.logger).circuit,
            inplace=True,
        )

        return circuit

    def __str__(self) -> str:
        return f"[Space Time QLBM on lattice={self.lattice}]"
