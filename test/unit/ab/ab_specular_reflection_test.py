"""Statevector-level tests for ABSpecularReflectionOperator."""

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlbm.components.ab.reflection.standard_reflection import (
    ABReflectionOperator,
    ABSpecularReflectionOperator,
)
from qlbm.lattice import ABLattice

_SIMULATOR = AerSimulator(method="statevector")


def _simulate_statevector(circuit: QuantumCircuit) -> Statevector:
    """Run a circuit on AerSimulator and return the final statevector."""
    qc = circuit.copy()
    qc.save_statevector()
    tqc = transpile(qc, _SIMULATOR, optimization_level=0)
    result = _SIMULATOR.run(tqc).result()
    return result.data(0)["statevector"]


def _make_specular_lattice(
    dim_x=8, dim_y=8, x_bounds=(2, 5), y_bounds=(2, 5)
) -> ABLattice:
    """Create an ABLattice with a specular cuboid obstacle."""
    return ABLattice(
        {
            "lattice": {"dim": {"x": dim_x, "y": dim_y}, "velocities": "d2q9"},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": list(x_bounds),
                    "y": list(y_bounds),
                    "boundary": "specular",
                }
            ],
        }
    )


def _make_multi_geometry_specular_lattice() -> ABLattice:
    """Create a multi-geometry ABLattice with specular obstacles."""
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
        }
    )

    lattice.set_geometries(
        [
            [
                {
                    "shape": "cuboid",
                    "x": [2, 5],
                    "y": [2, 5],
                    "boundary": "specular",
                }
            ],
            [
                {
                    "shape": "cuboid",
                    "x": [1, 3],
                    "y": [1, 3],
                    "boundary": "specular",
                }
            ],
        ]
    )

    return lattice


def _encode_basis_state(lattice: ABLattice, x: int, y: int, v: int, marker: int = 0):
    """Encode a computational basis state on the lattice circuit."""
    circuit = lattice.circuit.copy()

    for i, q in enumerate(lattice.grid_index(0)):
        if (x >> i) & 1:
            circuit.x(q)

    for i, q in enumerate(lattice.grid_index(1)):
        if (y >> i) & 1:
            circuit.x(q)

    for i, q in enumerate(lattice.velocity_index()):
        if (v >> i) & 1:
            circuit.x(q)

    if lattice.num_marker_qubits > 0:
        for i, q in enumerate(lattice.marker_index()):
            if (marker >> i) & 1:
                circuit.x(q)

    return circuit


def _get_ancilla_value(lattice: ABLattice, sv: Statevector, ancilla_idx: int) -> dict:
    """Extract specific obstacle ancilla probabilities from a statevector."""
    obstacle_qubit = lattice.ancillae_obstacle_index(ancilla_idx)[0]
    probs = {0: 0.0, 1: 0.0}
    for i, amp in enumerate(sv.data):
        val = (i >> obstacle_qubit) & 1
        probs[val] += abs(amp) ** 2
    return probs


def _all_ancillae_clean(lattice: ABLattice, sv: Statevector) -> bool:
    """Check that all obstacle ancillae are in the |0> state."""
    for idx in range(lattice.num_obstacle_qubits):
        probs = _get_ancilla_value(lattice, sv, idx)
        if probs[0] < 1.0 - 1e-10:
            return False
    return True


# =============================================================================
# Single geometry: full operator
# =============================================================================


class TestSpecularReflectionSingleGeometry:
    """Statevector tests for specular reflection with a single geometry.

    Tests only positions well outside any wall comparator range to
    avoid false negatives from unphysical basis states.
    """

    def test_ancillae_clean_far_from_obstacle(self):
        """All obstacle ancillae should be 0 for positions far from the obstacle."""
        lattice = _make_specular_lattice()
        op = ABReflectionOperator(lattice)

        prep = _encode_basis_state(lattice, x=0, y=0, v=0)
        prep.compose(op.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        assert _all_ancillae_clean(lattice, sv)

    def test_ancillae_clean_for_all_velocities_far_from_obstacle(self):
        """No velocity should leave dirty ancillae at positions far from the obstacle."""
        lattice = _make_specular_lattice()
        op = ABReflectionOperator(lattice)

        for v in range(9):
            prep = _encode_basis_state(lattice, x=0, y=0, v=v)
            prep.compose(op.circuit, inplace=True)
            sv = _simulate_statevector(prep)

            assert _all_ancillae_clean(lattice, sv), (
                f"Ancilla not clean for v={v} at (0,0)"
            )

    def test_operator_constructs_without_error(self):
        """The specular operator should construct without errors."""
        lattice = _make_specular_lattice()
        op = ABReflectionOperator(lattice)
        assert op.circuit is not None
        assert op.circuit.num_qubits == lattice.circuit.num_qubits


# =============================================================================
# Phase 1: set_inside_wall_ancilla_per_dim
# =============================================================================


class TestSpecularSetInsideWall:
    """Tests for the per-dimension inner wall marking primitive."""

    def test_x_ancilla_set_for_x_wall_point(self):
        """Position on the x-wall should have a_x set."""
        lattice = _make_specular_lattice()
        block = lattice.shapes["specular"][0]
        op = ABSpecularReflectionOperator(lattice, [block])

        wall_circuit = op._set_inside_wall_ancilla_per_dim(block, dim=0)

        prep = _encode_basis_state(lattice, x=2, y=3, v=0)
        prep.compose(wall_circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs_x = _get_ancilla_value(lattice, sv, 0)
        probs_y = _get_ancilla_value(lattice, sv, 1)

        assert probs_x[1] == pytest.approx(1.0, abs=1e-10), "a_x should be 1"
        assert probs_y[0] == pytest.approx(1.0, abs=1e-10), "a_y should be 0"

    def test_y_ancilla_set_for_y_wall_point(self):
        """Position on the y-wall should have a_y set."""
        lattice = _make_specular_lattice()
        block = lattice.shapes["specular"][0]
        op = ABSpecularReflectionOperator(lattice, [block])

        wall_circuit = op._set_inside_wall_ancilla_per_dim(block, dim=1)

        prep = _encode_basis_state(lattice, x=3, y=2, v=0)
        prep.compose(wall_circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs_x = _get_ancilla_value(lattice, sv, 0)
        probs_y = _get_ancilla_value(lattice, sv, 1)

        assert probs_x[0] == pytest.approx(1.0, abs=1e-10), "a_x should be 0"
        assert probs_y[1] == pytest.approx(1.0, abs=1e-10), "a_y should be 1"

    def test_both_ancillae_set_at_corner(self):
        """Inner corner points should have both a_x and a_y set after both phases."""
        lattice = _make_specular_lattice()
        block = lattice.shapes["specular"][0]
        op = ABSpecularReflectionOperator(lattice, [block])

        circuit = lattice.circuit.copy()
        circuit.compose(
            op._set_inside_wall_ancilla_per_dim(block, dim=0), inplace=True
        )
        circuit.compose(
            op._set_inside_wall_ancilla_per_dim(block, dim=1), inplace=True
        )

        # (2, 2) is an inner corner
        prep = _encode_basis_state(lattice, x=2, y=2, v=0)
        prep.compose(circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs_x = _get_ancilla_value(lattice, sv, 0)
        probs_y = _get_ancilla_value(lattice, sv, 1)

        assert probs_x[1] == pytest.approx(1.0, abs=1e-10), "a_x should be 1"
        assert probs_y[1] == pytest.approx(1.0, abs=1e-10), "a_y should be 1"

    def test_ancilla_not_set_outside(self):
        """Positions outside the obstacle should not have any ancilla set."""
        lattice = _make_specular_lattice()
        block = lattice.shapes["specular"][0]
        op = ABSpecularReflectionOperator(lattice, [block])

        for dim in range(2):
            wall_circuit = op._set_inside_wall_ancilla_per_dim(block, dim=dim)
            prep = _encode_basis_state(lattice, x=0, y=0, v=0)
            prep.compose(wall_circuit, inplace=True)
            sv = _simulate_statevector(prep)

            probs = _get_ancilla_value(lattice, sv, dim)
            assert probs[0] == pytest.approx(1.0, abs=1e-10), (
                f"ancilla[{dim}] should be 0 at (0,0)"
            )


# =============================================================================
# Multi-geometry
# =============================================================================


class TestSpecularMultiGeometry:
    """Statevector tests for specular reflection with multiple geometries."""

    def test_multi_geometry_ancillae_clean_far_from_all(self):
        """Positions far from all obstacles should have clean ancillae."""
        lattice = _make_multi_geometry_specular_lattice()
        op = ABReflectionOperator(lattice)

        for marker_val in [0, 1]:
            prep = _encode_basis_state(
                lattice, x=7, y=7, v=0, marker=marker_val
            )
            prep.compose(op.circuit, inplace=True)
            sv = _simulate_statevector(prep)

            assert _all_ancillae_clean(lattice, sv), (
                f"Failed for marker={marker_val}"
            )

    def test_multi_geometry_constructs(self):
        """Multi-geometry specular operator should construct without errors."""
        lattice = _make_multi_geometry_specular_lattice()
        op = ABReflectionOperator(lattice)
        assert op.circuit is not None
