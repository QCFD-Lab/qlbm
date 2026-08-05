"""The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.ab.reflection import (
    ABZoneAgnosticReflectionOperator,
)
from qlbm.components.ab.reflection.standard_reflection import ABReflectionOperator
from qlbm.components.base import LBMAlgorithm
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.tools.exceptions import CircuitException

from .streaming import ABStreamingOperator


class ABQLBM(LBMAlgorithm):
    """
    Implementation of the **A** mplitude **B** ased QLBM (ABQLBM).

    The algorithm consists of interleaving steps of streaming and boundary conditions.
    Note that there is **no** collision in this algorithm as of yet.
    Details of the general framework can be found in :cite:`collisionless`.
    The ABQLBM works with :math:`D_dQ_q` discretizations only.
    For multi-speed alternatives, see :class:`.MSQLBM`.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABQLBM
        from qlbm.lattice import ABLattice

        # Example with streaming only for simplicity.
        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABQLBM(lattice).draw("mpl")
    """

    def __init__(
        self,
        lattice: ABLattice,
        use_agnostic_bcs: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice: ABLattice = lattice

        self.use_agnostic_bcs = use_agnostic_bcs

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
            ABStreamingOperator(
                self.lattice,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )

        if self.use_agnostic_bcs and self.lattice.has_multiple_geometries():
            raise CircuitException(
                "Zone-agnostic boundary conditions are not supported "
                "with multiple geometries. Use use_agnostic_bcs=False "
                "or specify a single geometry."
            )

        if self.use_agnostic_bcs:
            circuit.compose(
                ABZoneAgnosticReflectionOperator(
                    self.lattice,
                    None,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )
        else:
            circuit.compose(
                ABReflectionOperator(
                    self.lattice,
                    None,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Algorithm ABQLBM with lattice {self.lattice}]"
