"""Measurement operator for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from logging import Logger, getLogger
from typing import Tuple

from qiskit import ClassicalRegister
from qiskit.circuit.library import MCMTGate, XGate
from typing_extensions import override

from qlbm.components.base import LQLGAOperator, SpaceTimeOperator
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.tools.utils import flatten, get_qubits_to_invert


class LQLGAGridVelocityMeasurement(LQLGAOperator):
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
