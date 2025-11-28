"""Quantum circuits used for setting the initial state in the :class:`ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.base import LBMPrimitive
from qlbm.components.common.primitives import TruncatedQFT
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.tools.exceptions import LatticeException


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

        match self.lattice.get_encoding():
            case ABEncodingType.AB:
                circuit.compose(
                    TruncatedQFT(
                        self.lattice.num_velocity_qubits,
                        self.lattice.num_velocities_per_point,
                        self.logger,
                    ).circuit,
                    qubits=self.lattice.velocity_index(),
                    inplace=True,
                )

                circuit.h(self.lattice.grid_index(1))
            case ABEncodingType.OH:
                nq = int(np.ceil(np.log2(self.lattice.num_velocity_qubits)))
                circuit.compose(
                    TruncatedQFT(
                        nq,
                        self.lattice.num_velocity_qubits,
                        self.logger,
                    ).circuit,
                    qubits=self.lattice.velocity_index()[:nq],
                    inplace=True,
                )

                circuit.compose(
                    self.__oh_permutation(),
                    qubits=self.lattice.velocity_index(),
                    inplace=True,
                )
            case _:
                raise LatticeException(
                    f"Encoding {self.lattice.get_encoding()} not supported."
                )

        if self.lattice.has_multiple_geometries():
            circuit.h(self.lattice.marker_index())

        return circuit

    def __oh_permutation(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        n = self.lattice.num_velocity_qubits
        dim = 2**n

        perm = [-1] * dim
        used_rows = set()

        for j in range(9):
            row = 1 << j  # 2^j
            perm[j] = row
            used_rows.add(row)

        # Fill in the rest of the permutation arbitrarily but bijectively.
        remaining_rows = [r for r in range(dim) if r not in used_rows]
        k = 0
        for col in range(9, dim):
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
        return f"[Primitive ABEInitialConditions with lattice {self.lattice}]"
