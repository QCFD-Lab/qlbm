import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.reflection import ABBounceBackReflectionPermutation
from qlbm.components.ab.reflection.common import ABSpecularReflectionPermutation
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.utils import bit_value


@pytest.mark.parametrize(
    "permutation_outcome_pairs",
    [(0, 0), (1, 3), (2, 4), (3, 1), (4, 2), (5, 7), (6, 8), (7, 5), (8, 6)]
    + [(i, i) for i in range(9, 16)],
)
def test_reflectionpermutation_outcomes_d2q9_bounceback(permutation_outcome_pairs):
    nq = 4
    sim = AerSimulator()

    qc = QuantumCircuit(nq)
    for q in range(nq):
        if bit_value(permutation_outcome_pairs[0], q):
            qc.x(q)

    qc.compose(
        ABBounceBackReflectionPermutation(
            nq, LatticeDiscretization.D2Q9, ABEncodingType.AB
        ).circuit,
        inplace=True,
    )
    qc.measure_all()
    tqc = transpile(qc, sim, optimization_level=0)

    counts = sim.run(tqc, shots=128).result().get_counts()

    assert all(
        int(c, 2) == permutation_outcome_pairs[1] for c in counts.keys()
    ), f"{permutation_outcome_pairs} handled incorrectly. Expected {permutation_outcome_pairs}, got {counts}."


@pytest.mark.parametrize(
    "permutation_outcome_pairs",
    [(0, 0), (1, 3), (3, 1), (5, 6), (6, 5), (8, 7), (7, 8), (2, 2), (4, 4)]
    + [(i, i) for i in range(9, 16)],
)
def test_reflectionpermutation_outcomes_d2q9_specular_rx(permutation_outcome_pairs):
    nq = 4
    sim = AerSimulator()

    qc = QuantumCircuit(nq)
    for q in range(nq):
        if bit_value(permutation_outcome_pairs[0], q):
            qc.x(q)

    qc.compose(
        ABSpecularReflectionPermutation(
            nq,
            LatticeDiscretization.D2Q9,
            ABEncodingType.AB,
            reflect_in_dim=(True, False),
        ).circuit,
        inplace=True,
    )
    qc.measure_all()
    tqc = transpile(qc, sim, optimization_level=0)

    counts = sim.run(tqc, shots=128).result().get_counts()

    assert all(
        int(c, 2) == permutation_outcome_pairs[1] for c in counts.keys()
    ), f"{permutation_outcome_pairs} handled incorrectly. Expected {permutation_outcome_pairs}, got {counts}."


@pytest.mark.parametrize(
    "permutation_outcome_pairs",
    [(0, 0), (1, 1), (3, 3), (2, 4), (4, 2), (5, 8), (8, 5), (6, 7), (7, 6)]
    + [(i, i) for i in range(9, 16)],
)
def test_reflectionpermutation_outcomes_d2q9_specular_ry(permutation_outcome_pairs):
    nq = 4
    sim = AerSimulator()

    qc = QuantumCircuit(nq)
    for q in range(nq):
        if bit_value(permutation_outcome_pairs[0], q):
            qc.x(q)

    qc.compose(
        ABSpecularReflectionPermutation(
            nq,
            LatticeDiscretization.D2Q9,
            ABEncodingType.AB,
            reflect_in_dim=(False, True),
        ).circuit,
        inplace=True,
    )
    qc.measure_all()
    tqc = transpile(qc, sim, optimization_level=0)

    counts = sim.run(tqc, shots=128).result().get_counts()

    assert all(
        int(c, 2) == permutation_outcome_pairs[1] for c in counts.keys()
    ), f"{permutation_outcome_pairs} handled incorrectly. Expected {permutation_outcome_pairs}, got {counts}."


@pytest.mark.parametrize(
    "permutation_outcome_pairs",
    [(0, 0), (1, 3), (2, 4), (3, 1), (4, 2), (5, 7), (6, 8), (7, 5), (8, 6)]
    + [(i, i) for i in range(9, 16)], 
)
def test_reflectionpermutation_outcomes_d2q9_specular_rxry(permutation_outcome_pairs):
    nq = 4
    sim = AerSimulator()

    qc = QuantumCircuit(nq)
    for q in range(nq):
        if bit_value(permutation_outcome_pairs[0], q):
            qc.x(q)

    qc.compose(
        ABSpecularReflectionPermutation(
            nq,
            LatticeDiscretization.D2Q9,
            ABEncodingType.AB,
            reflect_in_dim=(True, True),
        ).circuit,
        inplace=True,
    )
    qc.measure_all()
    tqc = transpile(qc, sim, optimization_level=0)

    counts = sim.run(tqc, shots=128).result().get_counts()

    assert all(
        int(c, 2) == permutation_outcome_pairs[1] for c in counts.keys()
    ), f"{permutation_outcome_pairs} handled incorrectly. Expected {permutation_outcome_pairs}, got {counts}."
