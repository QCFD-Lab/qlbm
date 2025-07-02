"""Permutation step in QLGA collision."""

from logging import Logger, getLogger
from typing import override

from qiskit import QuantumCircuit

from qlbm.components.base import LBMPrimitive
from qlbm.components.spacetime.collision.eqc_discretizations import EquivalenceClass
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException


class SpaceTimeEQCPermutation(LBMPrimitive):
    """Permutes states belonging to equivalence classes onto pre-determined basis states.

    Precedes redistribution.

    WIP.
    """

    def __init__(
        self,
        equivalence_class: EquivalenceClass,
        inverse: bool = False,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)
        self.equivalence_class = equivalence_class
        self.inverse = inverse

        self.circuit = self.create_circuit()

    @override
    def create_circuit(self):
        if self.equivalence_class.discretization == LatticeDiscretization.D2Q4:
            return self.__create_circuit_d2q4()
        elif self.equivalence_class.discretization == LatticeDiscretization.D3Q6:
            return self.__create_circuit_d3q6()
        else:
            raise CircuitException(
                f"Collision not yet supported for discretization {self.equivalence_class.discretization}."
            )

    def __create_circuit_d2q4(self):
        circuit = QuantumCircuit(4)

        if not self.inverse:
            circuit.cx(1, 2)
            circuit.cx(0, 1)
            circuit.cx(0, 3)
        else:
            circuit.cx(0, 3)
            circuit.cx(0, 1)
            circuit.cx(1, 2)

        return circuit

    def __create_circuit_d3q6(self):
        circuit = QuantumCircuit(6)

        match self.equivalence_class.id():
            case (2, [0, 0, 0]):
                circuit.cx(2, 3)
                circuit.cx(5, 4)

                circuit.cx(1, 2)
                circuit.cx(1, 3)
                circuit.cx(1, 5)

                circuit.cx(0, 2)
                circuit.cx(0, 4)
                circuit.cx(0, 5)
            case (4, [0, 0, 0]):
                circuit.ccx(4, 5, 3)
                circuit.ccx(0, 2, 4)
                circuit.ccx(0, 1, 2)
                circuit.ccx(0, 1, 5)
                circuit.x(1)
                circuit.cx(1, 0)
            case (3, [1, 0, 0]):
                circuit.cx(0, 3)
                circuit.cx(1, 2)
                circuit.ccx(0, 5, 4)
                circuit.cx(1, 5)
                circuit.swap(0, 1)
            case (3, [-1, 0, 0]):
                circuit.cx(1, 2)
                circuit.cx(5, 4)
                circuit.cx(1, 5)
                circuit.x(0)
                circuit.swap(0, 1)
            case (3, [0, 1, 0]):
                circuit.cx(0, 2)
                circuit.cx(5, 3)
                circuit.cx(0, 5)
                circuit.x(4)
            case (3, [0, -1, 0]):
                circuit.cx(5, 3)
                circuit.cx(0, 2)
                circuit.cx(0, 5)
                circuit.x(1)
            case (3, [0, 0, 1]):
                circuit.cx(1, 3)
                circuit.cx(0, 1)
                circuit.cx(0, 4)
                circuit.x(5)
            case (3, [0, 0, -1]):
                circuit.cx(0, 1)
                circuit.cx(4, 3)
                circuit.cx(0, 4)
                circuit.x(2)
            case _:
                raise CircuitException(
                    f"Collision not yet supported for discretization {self.equivalence_class.discretization} and equivalence class {self.equivalence_class.id()}."
                )

        return circuit if not self.inverse else circuit.inverse()

    @override
    def __str__(self) -> str:
        return f"[SpaceTimeEQCPermutation for eqc {self.equivalence_class}]"
