"""Utilities for the Amplitude-Based QLBM."""

from logging import Logger, getLogger
from time import perf_counter_ns

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.ab_lattice import ABLattice


class BinaryToOHPermutation(LBMPrimitive):
    """
    Permutes the first :math:`q` basis states of the binary encoding into the :math:`q` one-hot states of the OH encoding.

    This operator is implemented as a decomposed permutation matrix.
    As such, its decomposition will be exponentially expensive in the number of qubits.
    By default, the unitary acts the :math:`q` qubits of the of the one hot encoding (in a :math:`D_dQ_q` discretization).

    Example usage:
    .. plot::
        :include-source:

        from qlbm.components.ab import BinaryToOHPermutation
        from qlbm.lattice import OHLattice

        lattice = OHLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            }
        )

        BinaryToOHPermutation(lattice).draw("mpl")
    """

    lattice: ABLattice

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

        n = self.lattice.num_velocity_qubits
        dim = 2**n

        perm = [-1] * dim
        used_rows = set()

        for j in range(n):
            row = 1 << j  # 2^j
            perm[j] = row
            used_rows.add(row)

        # Fill in the rest of the permutation arbitrarily but bijectively.
        remaining_rows = [r for r in range(dim) if r not in used_rows]
        k = 0
        for col in range(n, dim):
            perm[col] = remaining_rows[k]
            k += 1

        U = np.zeros((dim, dim), dtype=complex)
        for col in range(dim):
            row = perm[col]
            U[row, col] = 1.0

        op = Operator(U)

        circuit = QuantumCircuit(n)
        circuit.unitary(op, range(n), label="binary_to_onehot")

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive BinaryOHPermutation with lattice {self.lattice}]"
