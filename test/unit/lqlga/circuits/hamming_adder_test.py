import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.result import Counts

from qlbm.components.common import HammingWeightAdder


def bit_string_to_bool_list(bitstring):
    return [x == "1" for x in bitstring]


def hamming_weight(bitstring):
    return len([True for x in bitstring if x == "1"])


def get_count_from_circuit(circuit, num_shots=128) -> Counts:
    sim = AerSimulator()
    tqc = transpile(circuit, sim)

    res = sim.run(tqc, num_shots=num_shots).result()
    return res.get_counts()


def test_hamming_adder_all_0s():
    adder = HammingWeightAdder(3, 5).circuit
    adder.measure_all()
    counts = get_count_from_circuit(adder)

    assert len(counts) == 1
    assert (int(s, 2) == 0 for s in counts)


def test_hamming_adder_1plus0():
    circuit = QuantumCircuit(8)
    circuit.x(3)
    adder = HammingWeightAdder(4, 4).circuit
    adder.measure_all()
    circuit.compose(adder, inplace=True)
    counts = get_count_from_circuit(circuit)

    assert len(counts) == 1
    assert all(int(s[4:][::-1], 2) == 1 for s in counts)


def test_hamming_adder_2plus0():
    # Hamming weight 2 in the x register
    # Number 0 in the y register
    circuit = QuantumCircuit(8)
    circuit.x(0)
    circuit.x(3)
    adder = HammingWeightAdder(4, 4).circuit
    adder.measure_all()
    circuit.compose(adder, inplace=True)
    counts = get_count_from_circuit(circuit)

    assert len(counts) == 1
    print(counts)
    assert all(int(s[:4], 2) == 2 for s in counts)


def test_hamming_adder_2plus4():
    # Hamming weight 2 in the x register
    # Number 4 in the y register
    circuit = QuantumCircuit(8)
    circuit.x(0)
    circuit.x(3)
    circuit.x(6)
    adder = HammingWeightAdder(4, 4).circuit
    adder.measure_all()
    circuit.compose(adder, inplace=True)
    counts = get_count_from_circuit(circuit)

    assert len(counts) == 1
    print(counts)
    assert all(int(s[:4], 2) == 6 for s in counts)


def test_hamming_adder_twice():
    # Hamming weight 2 in the x register
    # Number 0 in the y register
    circuit = QuantumCircuit(8)
    circuit.x(0)
    circuit.x(3)
    adder = HammingWeightAdder(4, 4).circuit
    circuit.compose(adder, inplace=True)
    circuit.compose(adder, inplace=True)
    circuit.measure_all()
    counts = get_count_from_circuit(circuit)

    assert len(counts) == 1
    assert all(int(s[:4], 2) == 4 for s in counts)


def test_hamming_adder_superposition_x():
    circuit = QuantumCircuit(8)
    circuit.h([0, 1, 2, 3])
    circuit.x(6)
    adder = HammingWeightAdder(4, 4).circuit
    adder.measure_all()
    circuit.compose(adder, inplace=True)
    counts = get_count_from_circuit(circuit)
    print(counts)
    assert len(counts) == 16
    assert all(int(s[:4], 2) == (hamming_weight(s[4:]) + 4) for s in counts)
