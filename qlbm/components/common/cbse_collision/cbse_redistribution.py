"""Permutations of states belonging to equivalence classes, based on the computational basis state encoding."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import override

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.eqc.eqc import EquivalenceClass
from qlbm.lattice.spacetime.properties_base import LatticeDiscretizationProperties
from qlbm.tools.utils import is_two_pow


class EQCRedistribution(LBMPrimitive):
    """
    Redistribution operator for equivalence classes in the CBSE encoding.

    The operator is mathematically described in section 4 of :cite:`spacetime2`.
    Redistribution is applied before and after permutations, and consists of a controlled
    unitary operator composed of a discrete Fourier transform (DFT)-block matrix.


    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`equivalence_class`  The equivalence class of the operator.
    :attr:`decompose_block`    Whether to decompose the DFT block into a circuit.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common import EQCRedistribution
        from qlbm.lattice import LatticeDiscretization
        from qlbm.lattice.eqc import EquivalenceClassGenerator

        # Generate some equivalence classes
        eqcs = EquivalenceClassGenerator(
            LatticeDiscretization.D3Q6
        ).generate_equivalence_classes()

        # Select one at random and draw its circuit in the schematic form
        EQCRedistribution(eqcs.pop(), decompose_block=False).circuit.draw("mpl")

    The `decompose_block` parameter can be set to ``True`` to decompose the DFT block into a circuit:

    .. plot::
        :include-source:

        from qlbm.components.common import EQCRedistribution
        from qlbm.lattice import LatticeDiscretization
        from qlbm.lattice.eqc import EquivalenceClassGenerator

        # Generate some equivalence classes
        eqcs = EquivalenceClassGenerator(
            LatticeDiscretization.D3Q6
        ).generate_equivalence_classes()

        # Select one at random and draw its decomposed circuit
        EQCRedistribution(eqcs.pop(), decompose_block=True).circuit.draw("mpl")

    """

    equivalence_class: EquivalenceClass
    """
    The equivalence class for which the redistribution is defined.
    """

    decompose_block: bool
    """
    Whether to decompose the DFT block into a circuit.
    If set to ``False``, the block is returned as a matrix. Otherwise, it is decomposed into a circuit.
    Defaults to ``True``.
    """

    def __init__(
        self,
        equivalence_class: EquivalenceClass,
        decompose_block: bool = True,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)
        self.equivalence_class = equivalence_class
        self.decompose_block = decompose_block

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        nv = LatticeDiscretizationProperties.get_num_velocities(
            self.equivalence_class.discretization
        )
        circuit = QuantumCircuit(nv)
        n = self.equivalence_class.size()
        nq = np.ceil(np.log2(n)).astype(int)

        redistribution_circuit = QuantumCircuit(nq)
        if is_two_pow(n):
            redistribution_circuit.ry(np.pi / 2, list(range(nq)), label="RY(π/2)")
        else:
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

            redistribution_circuit.append(op, list(range(nq)))

        circuit.compose(
            redistribution_circuit.control(
                nv - int(nq),
                label=rf"MCRY(π/2, {nq})" if is_two_pow(n) else rf"Coll({n}, {nq})",
            ),
            qubits=list(range(nv - 1, -1, -1)),
            inplace=True,
        )

        if not is_two_pow(n):
            return circuit.decompose() if self.decompose_block else circuit
        else:
            return circuit

    @override
    def __str__(self):
        return f"[EQCRedistribution({self.equivalence_class})]"
