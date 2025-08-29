"""Measurement operator for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from logging import Logger, getLogger

from qiskit import ClassicalRegister
from typing_extensions import override

from qlbm.components.base import LQLGAOperator
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice


class LQLGAGridVelocityMeasurement(LQLGAOperator):
    # TODO: Improve documentation
    """
    Measurement operator for the :class:`.LQLGA` algorithm.

    This operator measures the velocity qubits at each grid point in the LQLGA lattice.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.lqlga import LQLGAGridVelocityMeasurement
        from qlbm.lattice import LQLGALattice

        lattice = LQLGALattice(
            {
                "lattice": {
                    "dim": {"x": 5},
                    "velocities": "D1Q3",
                },
                "geometry": [],
            },
        )

        LQLGAGridVelocityMeasurement(lattice=lattice).draw("mpl")
    """

    def __init__(
        self,
        lattice: LQLGALattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice

        self.circuit = self.create_circuit()

    @override
    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        qubits_to_measure = list(range(self.lattice.num_base_qubits))
        circuit.add_register(ClassicalRegister(self.lattice.num_base_qubits))

        circuit.measure(
            qubits_to_measure,
            list(range(len(qubits_to_measure))),
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[LQLGAGridVelocityMeasurement for lattice {self.lattice}]"
