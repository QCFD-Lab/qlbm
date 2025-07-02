"""Redistribution step in QLGA collision."""

from logging import Logger, getLogger
from typing import override

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qlbm.components.base import LBMPrimitive
from qlbm.components.spacetime.collision.eqc_discretizations import EquivalenceClass
from qlbm.lattice.spacetime.properties_base import LatticeDiscretizationProperties


class SpaceTimeEQCRedistribution(LBMPrimitive):
    """
    Redistributes the amplitudes of the states belonging to an equivalence class evenly across all other equivalent states.

    WIP.
    """

    def __init__(
        self, equivalence_class: EquivalenceClass, logger: Logger = getLogger("qlbm")
    ):
        super().__init__(logger)
        self.equivalence_class = equivalence_class

        self.circuit = self.create_circuit()

    @override
    def create_circuit(self):
        nv = LatticeDiscretizationProperties.get_num_velocities(
            self.equivalence_class.discretization
        )
        circuit = QuantumCircuit(nv)

        n = self.equivalence_class.size()
        nq = np.ceil(np.log2(n)).astype(int)

        QFT = np.array(
            [
                [np.exp(2j * np.pi * i * j / n) / np.sqrt(n) for j in range(n)]
                for i in range(n)
            ]
        )

        U = np.eye(2**nq, dtype=complex)
        U[:n, :n] = QFT
        op = Operator(U)
        assert op.is_unitary()

        qft_block_circ = QuantumCircuit(nq)
        qft_block_circ.append(op, list(range(nq)))

        circuit.compose(
            qft_block_circ.control(nv - int(nq), label=rf"Coll({n}, {nq})"),
            qubits=list(range(nv - 1, -1, -1)),
            inplace=True,
        )

        return circuit.decompose()

    @override
    def __str__(self):
        return super().__str__()
