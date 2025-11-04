"""The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.abe.reflection import ABEReflectionOperator
from qlbm.components.base import LBMAlgorithm
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.abe_lattice import ABLattice
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import flatten

from .streaming import ABEStreamingOperator


class ABECQLBM(LBMAlgorithm):
    """TODO."""

    def __init__(
        self,
        lattice: ABLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice: ABLattice = lattice

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

        for bc in ["bounceback", "specular"]:
            if self.lattice.shapes[bc]:
                if not all(
                    isinstance(shape, Block)
                    for shape in self.lattice.shapes["specular"]
                ):
                    raise LatticeException(
                        f"All shapes with the {bc} boundary condition must be cuboids for the CQLBM algorithm. "
                    )

        circuit.compose(
            ABEReflectionOperator(
                self.lattice,
                flatten(list(self.lattice.shapes.values())),  # type: ignore
                logger=self.logger,
            ).circuit,
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Algorithm ABECQLBM with lattice {self.lattice}]"
