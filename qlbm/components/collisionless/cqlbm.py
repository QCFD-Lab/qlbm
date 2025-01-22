"""The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMAlgorithm
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import get_time_series

from .bounceback_reflection import BounceBackReflectionOperator
from .specular_reflection import SpecularReflectionOperator
from .streaming import CollisionlessStreamingOperator, StreamingAncillaPreparation


class CQLBM(LBMAlgorithm):
    """The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`.

    This implementation supports 2D and 3D simulations with with cuboid objects
    with either bounce-back or specular reflection boundary conditions.

    The algorithm is composed of three steps that are repeated according to a CFL counter:

    #. Streaming performed by the :class:`.CollisionlessStreamingOperator` increments or decrements the positions of particles on the grid.
    #. :class:`.BounceBackReflectionOperator` and :class:`.SpecularReflectionOperator` reflect the particles that come in contact with :class:`.Block` obstacles encoded in the :class:`.CollisionlessLattice`.
    #. The :class:`.StreamingAncillaPreparation` resets the state of the ancilla qubits for the next CFL counter substep.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
    """

    def __init__(
        self,
        lattice: CollisionlessLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice: CollisionlessLattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        # Assumes equal velocities in all dimensions
        # ! TODO adapt to DnQm discretization
        time_series = get_time_series(2 ** self.lattice.num_velocities[0].bit_length())
        circuit = QuantumCircuit(
            *self.lattice.registers,
        )

        for velocities_to_increment in time_series:
            circuit.compose(
                CollisionlessStreamingOperator(
                    self.lattice,
                    velocities_to_increment,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )
            if self.lattice.blocks["specular"]:
                circuit.compose(
                    SpecularReflectionOperator(
                        self.lattice,
                        self.lattice.blocks["specular"],
                        logger=self.logger,
                    ).circuit,
                    inplace=True,
                )

            if self.lattice.blocks["bounceback"]:
                circuit.compose(
                    BounceBackReflectionOperator(
                        self.lattice,
                        self.lattice.blocks["bounceback"],
                        logger=self.logger,
                    ).circuit,
                    inplace=True,
                )

            for dim in range(self.lattice.num_dims):
                circuit.compose(
                    StreamingAncillaPreparation(
                        self.lattice,
                        velocities_to_increment,
                        dim,
                        logger=self.logger,
                    ).circuit,
                    inplace=True,
                )
        return circuit

    @override
    def __str__(self) -> str:
        return f"[Algorithm CQLBM with lattice {self.lattice}]"
