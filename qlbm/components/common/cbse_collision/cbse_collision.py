"""Collision operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.components.common.cbse_collision.cbse_permutation import (
    EQCPermutation,
)
from qlbm.components.common.cbse_collision.cbse_redistribution import (
    EQCRedistribution,
)
from qlbm.lattice.eqc.eqc import EquivalenceClass
from qlbm.lattice.eqc.eqc_generator import (
    EquivalenceClassGenerator,
)
from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)


class EQCCollisionOperator(LBMPrimitive):
    """
    Collision operator based on the equivalence class abstraction described in section 5 of :cite:`spacetime2`.

    Consists of a permutation, redistribution, and inverse permutation of the velocity qubits.
    This operator is designed to be applied to a single velocity register, which can be repeated depending on the encoding.
    Used in the :class:`.GenericSpaceTimeCollisionOperator` and :class:`.GenericLQLGACollisionOperator`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`discretization`    The discretization for which this collision operator is defined.
    :attr:`num_velocities`    The number of velocities in the discretization.
    ========================= ======================================================================
    """

    discretization: LatticeDiscretization
    """
    The discretization of the lattice for which this collision operator is defined.
    """

    num_velocities: int
    """
    The number of velocities in the discretization."""

    def __init__(
        self,
        discretization: LatticeDiscretization,
    ) -> None:
        super().__init__()
        self.discretization = discretization
        self.num_velocities = LatticeDiscretizationProperties.get_num_velocities(
            self.discretization
        )

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        """
        Applies the collision operator of all equivalence classes onto one velocity register.

        Returns
        -------
        QuantumCircuit
            The circuit performing the complete collision operator.
        """
        circuit = QuantumCircuit(self.num_velocities)

        for eqc in EquivalenceClassGenerator(
            self.discretization
        ).generate_equivalence_classes():
            circuit.compose(self.create_circuit_one_eqc(eqc), inplace=True)

        return circuit

    def create_circuit_one_eqc(
        self, equivalence_class: EquivalenceClass
    ) -> QuantumCircuit:
        """
        Creates the PRP-based collision operator for one equivalence class.

        Parameters
        ----------
        equivalence_class : EquivalenceClass
            The equivalence class to collide.

        Returns
        -------
        QuantumCircuit
            The circuit performing the collision.
        """
        circuit = QuantumCircuit(self.num_velocities)
        circuit.compose(
            EQCPermutation(equivalence_class, logger=self.logger).circuit,
            inplace=True,
        )

        circuit.compose(
            EQCRedistribution(equivalence_class, logger=self.logger).circuit,
            inplace=True,
        )
        circuit.compose(
            EQCPermutation(equivalence_class, inverse=True, logger=self.logger).circuit,
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[EQCCollisionOperator for equi {self.discretization}]"
