from itertools import product

import pytest
from qiskit import ClassicalRegister, transpile
from qiskit_aer import AerSimulator

from qlbm.components.ab.initial import ABDiscreteUniformInitialConditions


@pytest.mark.parametrize(
    "velocities,lattice_fixture",
    list(
        product(
            [[0], [0, 1], [6, 7, 8], [4, 5, 0, 7], list(range(9))],
            ["ab_lattice_d2q9_8x8"],
        )
    ),
)
def test_initial_ab_no_gird_superposition(velocities, lattice_fixture, request):
    lattice = request.getfixturevalue(lattice_fixture)
    sim = AerSimulator()

    qc = lattice.circuit.copy()
    qc.add_register(ClassicalRegister(4))
    qc.compose(
        ABDiscreteUniformInitialConditions(
            lattice, velocities, ([], [])
        ).circuit,
        inplace=True,
    )

    qc.measure(lattice.velocity_index(), list(range(4)))
    tqc = transpile(qc, sim, optimization_level=0)

    counts = sim.run(tqc, shots=256).result().get_counts()

    output_velocities = list(set([int(c, 2) for c in counts.keys()]))

    assert sorted(output_velocities) == sorted(velocities), (
        f"Expected output velocities to be {velocities}, got {output_velocities}"
    )


@pytest.mark.parametrize(
    "velocities,lattice_fixture",
    list(
        product(
            [[0], [0, 1], [6, 7, 8], [4, 5, 0, 7], list(range(9))],
            ["oh_lattice_d2q9_8x8"],
        )
    ),
)
def test_initial_oh_no_gird_superposition(velocities, lattice_fixture, request):
    lattice = request.getfixturevalue(lattice_fixture)
    sim = AerSimulator()

    qc = lattice.circuit.copy()
    qc.add_register(ClassicalRegister(9))
    qc.compose(
        ABDiscreteUniformInitialConditions(
            lattice, velocities, ([], [])
        ).circuit,
        inplace=True,
    )

    qc.measure(lattice.velocity_index(), list(range(9)))
    tqc = transpile(qc, sim, optimization_level=0)

    counts = sim.run(tqc, shots=256).result().get_counts()

    output_velocities = list(set([int(c, 2) for c in counts.keys()]))

    assert sorted(output_velocities) == sorted([2**v for v in velocities]), (
        f"Expected output velocities to be {velocities}, got {output_velocities}"
    )
