"""The end-to-end algorithm of the Space-Time Quantum Lattice Boltzmann Algorithm described in :cite:`spacetime`."""

from logging import Logger, getLogger

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMAlgorithm
from qlbm.components.spacetime.reflection import PointWiseSpaceTimeReflectionOperator
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice

from .collision import SpaceTimeCollisionOperator
from .streaming import SpaceTimeStreamingOperator


class SpaceTimeQLBM(LBMAlgorithm):
    """The end-to-end algorithm of the Space-Time Quantum Lattice Boltzmann Algorithm described in :cite:`spacetime`.

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
        filter_inside_blocks: bool = True,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice, logger)

        self.lattice = lattice
        self.filter_inside_blocks = filter_inside_blocks
        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        for timestep in range(self.lattice.num_timesteps, 0, -1):
            circuit.compose(
                SpaceTimeStreamingOperator(self.lattice, timestep, self.logger).circuit,
                inplace=True,
            )

            circuit.compose(
                PointWiseSpaceTimeReflectionOperator(
                    self.lattice,
                    timestep,
                    self.lattice.blocks["bounceback"],
                    self.filter_inside_blocks,
                    self.logger,
                ).circuit,
                inplace=True,
            )

            # There is no collision in 1D
            if self.lattice.num_dims > 1:
                circuit.compose(
                    SpaceTimeCollisionOperator(
                        self.lattice, timestep, logger=self.logger
                    ).circuit,
                    inplace=True,
                )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Space Time QLBM on lattice={self.lattice}]"
