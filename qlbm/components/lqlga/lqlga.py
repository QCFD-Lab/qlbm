"""The end-to-end algorithm of the Space-Time Quantum Lattice Boltzmann Algorithm described in :cite:`spacetime`."""

from logging import Logger, getLogger

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMAlgorithm
from qlbm.components.lqlga.collision import GenericLQLGACollisionOperator
from qlbm.components.lqlga.streaming import LQLGAStreamingOperator
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice


class LQLGA(LBMAlgorithm):
    lattice: LQLGALattice

    def __init__(
        self,
        lattice: LQLGALattice,
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

        circuit.compose(
            LQLGAStreamingOperator(self.lattice, self.logger).circuit, inplace=True
        )

        circuit.compose(
            GenericLQLGACollisionOperator(self.lattice, self.logger).circuit,
            inplace=True,
        )

    @override
    def __str__(self) -> str:
        return f"[LQLGA on lattice={self.lattice}]"
