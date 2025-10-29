"""The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.abe.averaged_collision import ABEAveragedCollisionOperator
from qlbm.components.base import LBMAlgorithm
from qlbm.lattice import CollisionlessLattice
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.abe_lattice import ABELattice
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import get_time_series

from .streaming import ABEStreamingOperator


class ABECQLBM(LBMAlgorithm):
    """TODO."""

    def __init__(
        self,
        lattice: ABELattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice: ABELattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        circuit = QuantumCircuit(
            *self.lattice.registers,
        )

        circuit.compose(
            ABEStreamingOperator(
                self.lattice,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )

        # circuit.compose(
        #     ABEAveragedCollisionOperator(
        #         self.lattice,
        #         logger=self.logger,
        #     ).circuit,
        #     inplace=True,
        # )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Algorithm ABECQLBM with lattice {self.lattice}]"
