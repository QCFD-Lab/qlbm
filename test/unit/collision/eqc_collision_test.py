from typing import List

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.components.spacetime.collision.eqc_collision import (
    EQCCollisionOperator,
)
from qlbm.lattice.eqc.eqc_generator import (
    EquivalenceClassGenerator,
)
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.utils import flatten


def int_to_bool_list(num, num_bits):
    return bit_string_to_bool_list(format(num, f"0{num_bits}b"))


def bit_string_to_bool_list(bitstring):
    return [x == "1" for x in bitstring]


@pytest.fixture
def d2q4_equivalence_class_bitstrings() -> List[List[str]]:
    return [
        eqc.get_bitstrings()
        for eqc in EquivalenceClassGenerator(
            LatticeDiscretization.D2Q4
        ).generate_equivalence_classes()
    ]


@pytest.fixture
def d3q6_equivalence_class_bitstrings() -> List[List[str]]:
    return sorted(
        [
            eqc.get_bitstrings()
            for eqc in EquivalenceClassGenerator(
                LatticeDiscretization.D3Q6
            ).generate_equivalence_classes()
        ],
        key=lambda x: x[0],
    )  # Sort by first element


def verify_simulation_outcome(
    input_state: str,
    expected_outcomes: List[str],
    collision_circuit: QuantumCircuit,
    num_shots=128,
    verify_negative_cases: bool = True,
):
    init_circuit = QuantumCircuit(collision_circuit.num_qubits)
    # for c, b in enumerate(bit_string_to_bool_list(input_state[::-1])):
    #     if b:
    #         init_circuit.x(collision_circuit.num_qubits - c - 1)
    for c, b in enumerate(bit_string_to_bool_list(input_state)):
        if b:
            init_circuit.x(c)
    qc_final = init_circuit.compose(collision_circuit)
    qc_final.measure_all()
    sim = AerSimulator()
    counts = sim.run(transpile(qc_final, sim), shots=num_shots).result().get_counts()

    assert all(b[::-1] in counts for b in expected_outcomes), (
        f"{expected_outcomes} not covered by counts {counts}"
    )

    if verify_negative_cases:
        all_biststrings = [
            format(i, f"0{collision_circuit.num_qubits}b")
            for i in range(2**collision_circuit.num_qubits)
        ]
        assert all(
            b[::-1] not in counts for b in all_biststrings if b not in expected_outcomes
        )


@pytest.mark.parametrize(
    "equivalence_class_index",
    list(
        range(
            len(
                EquivalenceClassGenerator(
                    LatticeDiscretization.D2Q4
                ).generate_equivalence_classes()
            )
        )
    ),
)
def test_d2q4_collision_positive_cases(
    d2q4_equivalence_class_bitstrings, equivalence_class_index
):
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
            "geometry": [],
        },
    )

    local_circuit = EQCCollisionOperator(
        lattice.properties.get_discretization()
    ).circuit
    assert local_circuit.num_qubits == 4
    eqc = d2q4_equivalence_class_bitstrings[equivalence_class_index]
    for velocity_cfg in eqc:
        verify_simulation_outcome(
            velocity_cfg, eqc, local_circuit, verify_negative_cases=True
        )


def test_d2q4_collision_negative_cases(d2q4_equivalence_class_bitstrings):
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
            "geometry": [],
        },
    )

    local_circuit = EQCCollisionOperator(
        lattice.properties.get_discretization()
    ).circuit
    assert local_circuit.num_qubits == 4

    for b in [
        format(i, f"0{local_circuit.num_qubits}b")
        for i in range(2**local_circuit.num_qubits)
        if format(i, f"0{local_circuit.num_qubits}b")
        not in flatten(d2q4_equivalence_class_bitstrings)
    ]:
        verify_simulation_outcome(
            b,
            [b],
            local_circuit,
            verify_negative_cases=True,
        )


@pytest.mark.parametrize(
    "equivalence_class_index",
    list(
        range(
            len(
                EquivalenceClassGenerator(
                    LatticeDiscretization.D3Q6
                ).generate_equivalence_classes()
            )
        )
    ),
)
def test_d3q6_collision_positive_cases(
    d3q6_equivalence_class_bitstrings, equivalence_class_index
):
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {
                "dim": {"x": 2, "y": 2, "z": 2},
                "velocities": "D3Q6",
            },
            "geometry": [],
        },
    )

    local_circuit = EQCCollisionOperator(
        lattice.properties.get_discretization()
    ).circuit
    assert local_circuit.num_qubits == 6
    eqc = d3q6_equivalence_class_bitstrings[equivalence_class_index]
    for velocity_cfg in eqc:
        verify_simulation_outcome(
            velocity_cfg, eqc, local_circuit, verify_negative_cases=True
        )

    # [['010011', '100101'], ['010110', '001101'], ['011011', '110110', '101101'], ['100100', '001001', '010010'], ['100110', '001011'], ['101100', '011010'], ['110010', '101001'], ['110100', '011001']]
