from itertools import product

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.components.common.primitives import AdditionConversion
from qlbm.tools.utils import bit_value


@pytest.mark.parametrize(
    "nq,state_in,state_out", list(product([4], [1, 4, 7, 11, 14], [0, 2, 8, 9, 12]))
)
def test_addition_conversion(nq, state_in, state_out):
    sim = AerSimulator()

    qc = QuantumCircuit(nq + 1)
    for q in range(nq):
        if bit_value(state_in, q):
            qc.x(q)

    qc.compose(
        AdditionConversion(nq, state_in, state_out).circuit,
        inplace=True,
    )
    qc.measure_all()
    tqc = transpile(qc, sim, optimization_level=0)

    counts = sim.run(tqc, shots=128).result().get_counts()

    assert all(int(c, 2) == state_out for c in counts.keys()), (
        f"{state_in} handled incorrectly. Expected {state_out}, got {counts}."
    )
