"""The end-to-end algorithm of the Space-Time Quantum Lattice Boltzmann Algorithm described in :cite:`spacetime`."""

from logging import Logger, getLogger

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMAlgorithm
from qlbm.components.lqlga.collision import GenericLQLGACollisionOperator
from qlbm.components.lqlga.reflection import LQLGAReflectionOperator
from qlbm.components.lqlga.streaming import LQLGAStreamingOperator
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice


class LQLGA(LBMAlgorithm):
    r"""
    Implementation of the Linear Quantum Lattice Gas Algorithm (LQLGA).

    For a lattice with :math:`N_g` gridpoints and :math:`q` discrete velocities,
    LQLGA requires exactly :math:`N_g \cdot q` qubits.

    That is exactly equal to the number of classical bits required for one
    deterministic run of the classical LGA algorithm.

    More information about this algorithm can be found in :cite:t:`lqlga1`, :cite:t:`lqlga2`, and :cite:t:`spacetime2`.
    """

    lattice: LQLGALattice

    def __init__(
        self,
        lattice: LQLGALattice,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice, logger)

        self.lattice = lattice

        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        circuit.compose(
            GenericLQLGACollisionOperator(self.lattice, self.logger).circuit,
            inplace=True,
        )

        circuit.compose(
            LQLGAStreamingOperator(self.lattice, self.logger).circuit, inplace=True
        )

        circuit.compose(
            LQLGAReflectionOperator(
                self.lattice, self.lattice.shapes["bounceback"], self.logger
            ).circuit,
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[LQLGA on lattice={self.lattice}]"
