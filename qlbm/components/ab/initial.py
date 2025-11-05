"""Quantum circuits used for setting the initial state in the :class:`ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.components.common.primitives import TruncatedQFT
from qlbm.lattice.lattices.ab_lattice import ABLattice


class ABInitialConditions(LBMPrimitive):
    """
    Initial conditions for the :class:`ABQLBM` algorithm.

    This component creates an equal magnitude superposition of all velocity
    basis states at position ``(0, 0)`` using the :class:`TruncatedQFT`.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABInitialConditions
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABInitialConditions(lattice).draw("mpl")

    You can also get the low-level decomposition of the circuit as:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABInitialConditions
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABInitialConditions(lattice).circuit.decompose(reps=2).draw("mpl")
    """

    def __init__(
        self,
        lattice: ABLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        circuit.compose(
            TruncatedQFT(
                self.lattice.num_velocity_qubits,
                self.lattice.num_velocities_per_point,
                self.logger,
            ).circuit,
            qubits=self.lattice.velocity_index(),
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABEInitialConditions with lattice {self.lattice}]"
