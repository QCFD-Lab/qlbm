from logging import Logger, getLogger

from qiskit import QuantumCircuit

from qlbm.components.base import LBMAlgorithm
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice

from .collision import SpaceTimeCollisionOperator
from .streaming import SpaceTimeStreamingOperator


class SpaceTimeQLBM(LBMAlgorithm):
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
