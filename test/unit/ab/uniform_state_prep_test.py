import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.components.common.primitives import UniformStatePrep


@pytest.mark.parametrize(
    "num_states",
    list(range(1, 10)),
)
def test_uniform_State_prep(num_states):
    nq = 5
    sim = AerSimulator()

    qc = QuantumCircuit(nq)
    qc.compose(
        UniformStatePrep(nq, num_states).circuit,
        inplace=True,
    )
    tqc = transpile(qc, sim)
    tqc.save_statevector()
    result = sim.run(tqc).result()
    state = result.get_statevector(tqc)

    expected = 1.0 / np.sqrt(num_states)

    assert np.allclose(np.abs(state)[:num_states], expected, atol=1e-8), (
        "Uniform state prep results in wrong magnitudes for the first k basis states"
    )
    assert np.allclose(np.abs(state)[num_states:], 0.0, atol=1e-8), (
        "Uniform state prep results in wrong magnitudes for the trailing basis states"
    )
