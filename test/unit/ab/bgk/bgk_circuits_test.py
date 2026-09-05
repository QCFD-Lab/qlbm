import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from qlbm.components.ab import (
    ABBGKQLBM,
    ABAngleEncodedEquilibrium,
    ABBGKCollisionOperator,
    ABBGKMeasurement,
    ABBranchAngleEncoding,
    ABBranchStatePreparation,
    ABLocalBGKCollision,
)
from qlbm.components.ab.collision.angle_encoding import append_multi_controlled_ry
from qlbm.tools.exceptions import CircuitException


def statevector_of(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(Statevector(circuit).data).real


def test_branch_state_preparation_creates_the_branch_amplitudes(encoding):
    amplitudes = statevector_of(ABBranchStatePreparation(encoding).circuit)

    assert amplitudes[: encoding.NUM_BRANCHES] == pytest.approx(encoding.beta)
    assert amplitudes[encoding.NUM_BRANCHES :] == pytest.approx(0.0)


def test_branch_angle_encoding_completes_the_branch_state(encoding, velocity_samples):
    for velocity_x, velocity_y in velocity_samples:
        angle_x, angle_y = encoding.angles(velocity_x, velocity_y)

        circuit = QuantumCircuit(encoding.NUM_COLLISION_QUBITS)
        circuit.compose(
            ABBranchStatePreparation(encoding).circuit, qubits=[2, 3, 4], inplace=True
        )
        circuit.compose(
            ABBranchAngleEncoding(float(angle_x), float(angle_y), encoding).circuit,
            inplace=True,
        )

        assert statevector_of(circuit) == pytest.approx(
            encoding.branch_state(float(angle_x), float(angle_y))
        )


def test_angle_encoded_equilibrium_matches_the_reference_state(
    encoding, velocity_samples
):
    for velocity_x, velocity_y in velocity_samples:
        component = ABAngleEncodedEquilibrium(velocity_x, velocity_y, encoding)

        assert component.circuit.num_qubits == encoding.NUM_COLLISION_QUBITS
        assert statevector_of(component.circuit) == pytest.approx(
            encoding.branch_state(*encoding.angles(velocity_x, velocity_y))
        )


def test_local_collision_produces_the_equilibrium(encoding, velocity_samples):
    for velocity_x, velocity_y in velocity_samples:
        amplitudes = statevector_of(
            ABLocalBGKCollision(velocity_x, velocity_y, encoding).circuit
        )

        assert encoding.alpha * amplitudes[encoding.physical_state_indices()] == (
            pytest.approx(encoding.equilibrium(1.0, velocity_x, velocity_y))
        )
        assert amplitudes[encoding.unused_state_indices()] == pytest.approx(0.0)


def test_multi_controlled_ry_skips_negligible_rotations():
    circuit = QuantumCircuit(2)
    append_multi_controlled_ry(
        circuit, 1e-15, [circuit.qubits[0]], [1], circuit.qubits[1]
    )

    assert circuit.size() == 0


def test_multi_controlled_ry_without_controls_is_unconditional():
    circuit = QuantumCircuit(1)
    append_multi_controlled_ry(circuit, np.pi, [], [], circuit.qubits[0])

    assert statevector_of(circuit) == pytest.approx([0.0, 1.0])


def test_multi_controlled_ry_rejects_mismatched_controls():
    circuit = QuantumCircuit(2)

    with pytest.raises(CircuitException, match="control values"):
        append_multi_controlled_ry(
            circuit, 1.0, [circuit.qubits[0]], [1, 0], circuit.qubits[1]
        )


def test_multi_controlled_ry_rejects_invalid_control_values():
    circuit = QuantumCircuit(2)

    with pytest.raises(CircuitException, match="either 0 or 1"):
        append_multi_controlled_ry(
            circuit, 1.0, [circuit.qubits[0]], [2], circuit.qubits[1]
        )


def test_collision_operator_only_touches_the_collision_qubits(ab_bgk_lattice_4x4):
    operator = ABBGKCollisionOperator(ab_bgk_lattice_4x4)
    instruction = operator.circuit.data[0]

    assert operator.circuit.size() == 1
    assert [
        operator.circuit.find_bit(qubit).index for qubit in instruction.qubits
    ] == ab_bgk_lattice_4x4.collision_index()


def test_collision_operator_is_unitary(ab_bgk_lattice_4x4):
    operator = ABBGKCollisionOperator(ab_bgk_lattice_4x4)

    assert Operator(operator.circuit).is_unitary()


def test_collision_operator_acts_on_every_gridpoint(
    ab_bgk_lattice_4x4, taylor_green_field_4x4
):
    from qlbm.components.ab import ABBGKInitialConditions

    encoding = ab_bgk_lattice_4x4.encoding
    density, velocity_x, velocity_y = taylor_green_field_4x4
    initial = ABBGKInitialConditions(
        ab_bgk_lattice_4x4, density, velocity_x, velocity_y
    )

    collided = Statevector(initial.statevector).evolve(
        ABBGKCollisionOperator(ab_bgk_lattice_4x4).circuit
    )
    populations = encoding.populations_from_probabilities(
        np.abs(np.asarray(collided))[ab_bgk_lattice_4x4.physical_basis_indices()] ** 2,
        initial.density_norm,
    )

    assert populations == pytest.approx(
        encoding.equilibrium(density, velocity_x, velocity_y)
    )


def test_algorithm_contains_collision_streaming_and_reflection(
    ab_bgk_lattice_4x4, ab_bgk_lattice_4x4_obstacle
):
    periodic = ABBGKQLBM(ab_bgk_lattice_4x4)
    with_obstacle = ABBGKQLBM(ab_bgk_lattice_4x4_obstacle)

    assert periodic.circuit.num_qubits == ab_bgk_lattice_4x4.circuit.num_qubits
    assert with_obstacle.circuit.size() > periodic.circuit.size()


def test_algorithm_preserves_the_norm(ab_bgk_lattice_4x4, taylor_green_field_4x4):
    from qlbm.components.ab import ABBGKInitialConditions

    density, velocity_x, velocity_y = taylor_green_field_4x4
    initial = ABBGKInitialConditions(
        ab_bgk_lattice_4x4, density, velocity_x, velocity_y
    )

    evolved = Statevector(initial.statevector).evolve(
        ABBGKQLBM(ab_bgk_lattice_4x4).circuit
    )

    assert np.linalg.norm(np.asarray(evolved)) == pytest.approx(1.0)


def test_measurement_samples_grid_velocity_and_marker(ab_bgk_lattice_4x4):
    measurement = ABBGKMeasurement(ab_bgk_lattice_4x4)
    expected = (
        ab_bgk_lattice_4x4.grid_index()
        + ab_bgk_lattice_4x4.velocity_index()
        + ab_bgk_lattice_4x4.marker_index()
    )

    assert measurement.circuit.num_clbits == len(expected)
    assert [
        measurement.circuit.find_bit(instruction.qubits[0]).index
        for instruction in measurement.circuit.data
    ] == expected
