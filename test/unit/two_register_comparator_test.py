import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlbm.components.common.comparators import TwoRegisterComparator
from qlbm.tools.utils import ComparatorMode


def _state_index(x_value: int, y_value: int, out_value: int, num_qubits: int) -> int:
    return x_value + (y_value << num_qubits) + (out_value << (2 * num_qubits))


def _extract_register_value(state_index: int, start: int, num_qubits: int) -> int:
    # Reconstruct an integer from a contiguous little-endian qubit slice.
    return sum(((state_index >> (start + bit)) & 1) << bit for bit in range(num_qubits))


@pytest.mark.parametrize(
    "mode",
    [ComparatorMode.LT, ComparatorMode.LE, ComparatorMode.GT, ComparatorMode.GE],
)
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_two_register_comparator_all_modes(mode: ComparatorMode, num_qubits: int):
    # Build one comparator circuit per (mode, width) and reuse it for all inputs.
    comparator = TwoRegisterComparator(num_qubits=num_qubits, mode=mode)
    operator = mode.to_operator()

    # Exhaustively verify all (x, y) inputs for this register width.
    for x_value in range(2**num_qubits):
        for y_value in range(2**num_qubits):
            qc = QuantumCircuit(2 * num_qubits + 1)

            # Prepare |x>|y>|0> in computational basis (LSB at the lower qubit index).
            for bit in range(num_qubits):
                if (x_value >> bit) & 1:
                    qc.x(bit)
                if (y_value >> bit) & 1:
                    qc.x(num_qubits + bit)

            # Apply the reversible two-register comparator.
            qc.compose(comparator.circuit, inplace=True)

            # The circuit is deterministic on basis input; a single basis state should have unit amplitude.
            state = Statevector.from_instruction(qc)
            amplitudes = state.data
            max_index = max(range(len(amplitudes)), key=lambda idx: abs(amplitudes[idx]))
            max_amplitude = amplitudes[max_index]

            # Ensure no superposition/leakage due to incorrect uncomputation.
            assert abs(abs(max_amplitude) - 1.0) < 1e-9

            x_after = _extract_register_value(max_index, 0, num_qubits)
            y_after = _extract_register_value(max_index, num_qubits, num_qubits)
            out_after = (max_index >> (2 * num_qubits)) & 1

            # Comparator must preserve both input registers exactly.
            assert x_after == x_value
            assert y_after == y_value

            # Output ancilla must encode the selected inequality mode.
            assert out_after == int(operator(x_value, y_value))


@pytest.mark.parametrize(
    "mode",
    [ComparatorMode.LT, ComparatorMode.LE, ComparatorMode.GT, ComparatorMode.GE],
)
def test_two_register_comparator_superposition_inputs(mode: ComparatorMode):
    num_qubits = 2
    num_total_qubits = 2 * num_qubits + 1
    comparator = TwoRegisterComparator(num_qubits=num_qubits, mode=mode)
    operator = mode.to_operator()

    def assert_expected_superposition(circuit: QuantumCircuit):
        input_state = Statevector.from_instruction(circuit)

        # Build expected output by routing each |x,y,0> amplitude to |x,y,f(x,y)>.
        expected = [0j] * (2**num_total_qubits)
        for x_value in range(2**num_qubits):
            for y_value in range(2**num_qubits):
                source_idx = _state_index(x_value, y_value, 0, num_qubits)
                target_idx = _state_index(
                    x_value,
                    y_value,
                    int(operator(x_value, y_value)),
                    num_qubits,
                )
                expected[target_idx] = input_state.data[source_idx]

        actual_state = input_state.evolve(comparator.circuit)
        expected_state = Statevector(expected)

        assert actual_state.equiv(expected_state)

    # Superposition over x only (y fixed to 2).
    x_superposed = QuantumCircuit(num_total_qubits)
    x_superposed.h(0)
    x_superposed.h(1)
    x_superposed.x(num_qubits + 1)
    assert_expected_superposition(x_superposed)

    # Superposition over y only (x fixed to 1).
    y_superposed = QuantumCircuit(num_total_qubits)
    y_superposed.x(0)
    y_superposed.h(num_qubits)
    y_superposed.h(num_qubits + 1)
    assert_expected_superposition(y_superposed)

    # Superposition over both registers.
    both_superposed = QuantumCircuit(num_total_qubits)
    both_superposed.h(0)
    both_superposed.h(1)
    both_superposed.h(num_qubits)
    both_superposed.h(num_qubits + 1)
    assert_expected_superposition(both_superposed)