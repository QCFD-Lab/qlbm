"""Statevector-level tests for the zone-agnostic reflection oracle and operator."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlbm.components.ab.reflection.agnosotic_reflection import (
    ABZoneAgnosticReflectionOperator,
    ABZoneAgnosticReflectionOracle,
)
from qlbm.lattice import ABLattice
from qlbm.tools.exceptions import CircuitException

# =============================================================================
# Helper utilities
# =============================================================================

_SIMULATOR = AerSimulator(method="statevector")


def _simulate_statevector(circuit: QuantumCircuit) -> Statevector:
    """Run a circuit on AerSimulator and return the final statevector."""
    qc = circuit.copy()
    qc.save_statevector()
    tqc = transpile(qc, _SIMULATOR, optimization_level=0)
    result = _SIMULATOR.run(tqc).result()
    return result.data(0)["statevector"]


def _make_single_geometry_lattice(dim_x=4, dim_y=4) -> ABLattice:
    """Create a single-geometry ABLattice with a cuboid obstacle."""
    return ABLattice(
        {
            "lattice": {"dim": {"x": dim_x, "y": dim_y}, "velocities": "d2q9"},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [1, 2],
                    "y": [1, 2],
                    "boundary": "bounceback",
                }
            ],
        }
    )


def _make_multi_geometry_lattice(dim_x=4, dim_y=4) -> ABLattice:
    """Create a multi-geometry ABLattice with two cuboid configurations."""
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": dim_x, "y": dim_y}, "velocities": "d2q9"},
        }
    )

    lattice.set_geometries(
        [
            [
                {
                    "shape": "cuboid",
                    "x": [1, 2],
                    "y": [1, 2],
                    "boundary": "bounceback",
                }
            ],
            [
                {
                    "shape": "cuboid",
                    "x": [0, 1],
                    "y": [0, 1],
                    "boundary": "bounceback",
                }
            ],
        ]
    )

    return lattice


def _encode_basis_state(lattice: ABLattice, x: int, y: int, v: int, marker: int = 0):
    r"""Encode a computational basis state |x>|y>|v>|ancillae>|marker>.

    The grid and velocity are encoded in the standard binary representation.
    Ancillae are initialized to 0. Marker is set via X gates.
    """
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


def _get_obstacle_ancilla_value(lattice: ABLattice, sv: Statevector) -> dict:
    """Extract the obstacle ancilla probabilities from a statevector.

    Returns a dict mapping obstacle ancilla value (0 or 1) to probability.
    """
    obstacle_idx = lattice.ancillae_obstacle_index()
    probs = {}
    for val in [0, 1]:
        prob = 0.0
        for i, amp in enumerate(sv.data):
            obstacle_val = (i >> obstacle_idx[0]) & 1
            if obstacle_val == val:
                prob += abs(amp) ** 2
        probs[val] = prob
    return probs


# =============================================================================
# Oracle: single geometry statevector tests
# =============================================================================


class TestOracleSingleGeometry:
    """Statevector tests for the oracle without marker control (single geometry)."""

    def test_oracle_marks_position_inside_block(self):
        """Oracle should set obstacle ancilla for positions inside the block."""
        lattice = _make_single_geometry_lattice()
        shape = lattice.shapes["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

        # Position (1, 1) is inside the block [1,2] x [1,2]
        prep = _encode_basis_state(lattice, x=1, y=1, v=0)
        prep.compose(oracle.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[1] == pytest.approx(1.0, abs=1e-10)

    def test_oracle_does_not_mark_position_outside_block(self):
        """Oracle should not set obstacle ancilla for positions outside the block."""
        lattice = _make_single_geometry_lattice()
        shape = lattice.shapes["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

        # Position (0, 0) is outside the block [1,2] x [1,2]
        prep = _encode_basis_state(lattice, x=0, y=0, v=0)
        prep.compose(oracle.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[0] == pytest.approx(1.0, abs=1e-10)

    def test_oracle_marks_corner_of_block(self):
        """Oracle should mark the upper corner of the block."""
        lattice = _make_single_geometry_lattice()
        shape = lattice.shapes["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

        # Position (2, 2) is the upper corner of the block [1,2] x [1,2]
        prep = _encode_basis_state(lattice, x=2, y=2, v=0)
        prep.compose(oracle.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[1] == pytest.approx(1.0, abs=1e-10)

    def test_oracle_is_self_inverse(self):
        """Applying the oracle twice should return to the original state."""
        lattice = _make_single_geometry_lattice()
        shape = lattice.shapes["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

        prep = _encode_basis_state(lattice, x=1, y=1, v=0)
        prep.compose(oracle.circuit, inplace=True)
        prep.compose(oracle.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[0] == pytest.approx(1.0, abs=1e-10)


# =============================================================================
# Oracle: marker-controlled statevector tests
# =============================================================================


class TestOracleWithMarkerControl:
    """Statevector tests for the oracle with marker control (parallel BCs)."""

    def test_oracle_marks_only_when_marker_matches(self):
        """Oracle controlled on marker should only set obstacle ancilla when marker is all-ones."""
        lattice = _make_multi_geometry_lattice()
        shape = lattice.geometries[0]["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(
            lattice, shape, control_on_marker_state=True
        )

        # Position (1, 1) is inside the block.
        # Marker = 1 (all ones for 1-qubit marker) -> should mark
        prep = _encode_basis_state(lattice, x=1, y=1, v=0, marker=1)
        prep.compose(oracle.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[1] == pytest.approx(1.0, abs=1e-10)

    def test_oracle_does_not_mark_when_marker_does_not_match(self):
        """Oracle controlled on marker should NOT set obstacle ancilla when marker is not all-ones."""
        lattice = _make_multi_geometry_lattice()
        shape = lattice.geometries[0]["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(
            lattice, shape, control_on_marker_state=True
        )

        # Position (1, 1) is inside the block.
        # Marker = 0 (not all ones) -> should NOT mark
        prep = _encode_basis_state(lattice, x=1, y=1, v=0, marker=0)
        prep.compose(oracle.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[0] == pytest.approx(1.0, abs=1e-10)

    def test_oracle_preserves_grid_state_regardless_of_marker(self):
        """Oracle should restore grid qubits to original state for all marker values.

        Even though the oracle temporarily modifies grid qubits (via subtraction/addition),
        the net effect on the grid should be zero, regardless of marker state.
        """
        lattice = _make_multi_geometry_lattice()
        shape = lattice.geometries[0]["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(
            lattice, shape, control_on_marker_state=True
        )

        for marker_val in [0, 1]:
            prep = _encode_basis_state(lattice, x=1, y=1, v=3, marker=marker_val)
            original_sv = _simulate_statevector(prep)

            prep.compose(oracle.circuit, inplace=True)
            after_sv = _simulate_statevector(prep)

            # Check that grid qubits are preserved by comparing marginal probabilities
            grid_qubits = lattice.grid_index()
            original_grid_probs = original_sv.probabilities(grid_qubits)
            after_grid_probs = after_sv.probabilities(grid_qubits)
            np.testing.assert_allclose(original_grid_probs, after_grid_probs, atol=1e-10)

    def test_ymonomial_raises_with_marker_control(self):
        """YMonomial oracle should raise when control_on_marker_state=True."""
        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 2, "y": 4}, "velocities": "D2Q4"},
                "geometry": [
                    {
                        "shape": "ymonomial",
                        "exponent": 2,
                        "comparator": "<=",
                        "boundary": "bounceback",
                    }
                ],
            }
        )
        lattice.set_num_marker_qubits(1)

        shape = lattice.shapes["bounceback"][0]

        with pytest.raises(CircuitException, match="Marker-controlled oracles"):
            ABZoneAgnosticReflectionOracle(
                lattice, shape, control_on_marker_state=True
            )


# =============================================================================
# Combined oracle statevector tests
# =============================================================================


class TestCombinedOracle:
    """Statevector tests for the combined oracle used in multi-geometry parallel BCs.

    The combined oracle applies each geometry's oracle controlled on the
    corresponding marker state. It verifies that for a superposition of
    marker states, the obstacle ancilla is correctly set per geometry.
    """

    def test_combined_oracle_marks_geometry_0_only(self):
        """For marker=0, only geometry 0's obstacle region should be marked."""
        lattice = _make_multi_geometry_lattice()
        operator = ABZoneAgnosticReflectionOperator(lattice)

        # Access the combined oracle through the private method
        oracle = operator._ABZoneAgnosticReflectionOperator__build_combined_oracle()

        # Position (1, 1) is inside geometry 0 ([1,2]x[1,2]) but also inside geometry 1 ([0,1]x[0,1])
        # With marker=0, only geometry 0's oracle fires
        prep = _encode_basis_state(lattice, x=1, y=1, v=0, marker=0)
        prep.compose(oracle, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[1] == pytest.approx(1.0, abs=1e-10)

    def test_combined_oracle_marks_geometry_1_only(self):
        """For marker=1, only geometry 1's obstacle region should be marked."""
        lattice = _make_multi_geometry_lattice()
        operator = ABZoneAgnosticReflectionOperator(lattice)

        oracle = operator._ABZoneAgnosticReflectionOperator__build_combined_oracle()

        # Position (0, 0) is inside geometry 1 ([0,1]x[0,1]) but NOT inside geometry 0
        # With marker=1, geometry 1's oracle fires
        prep = _encode_basis_state(lattice, x=0, y=0, v=0, marker=1)
        prep.compose(oracle, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[1] == pytest.approx(1.0, abs=1e-10)

    def test_combined_oracle_does_not_mark_wrong_geometry(self):
        """Position inside geometry 1 should NOT be marked when marker=0."""
        lattice = _make_multi_geometry_lattice()
        operator = ABZoneAgnosticReflectionOperator(lattice)

        oracle = operator._ABZoneAgnosticReflectionOperator__build_combined_oracle()

        # Position (0, 0) is inside geometry 1 but NOT geometry 0
        # With marker=0, geometry 0's oracle fires but position is outside geo 0
        prep = _encode_basis_state(lattice, x=0, y=0, v=0, marker=0)
        prep.compose(oracle, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[0] == pytest.approx(1.0, abs=1e-10)

    def test_combined_oracle_outside_all_geometries(self):
        """Position outside all geometries should never be marked."""
        lattice = _make_multi_geometry_lattice()
        operator = ABZoneAgnosticReflectionOperator(lattice)

        oracle = operator._ABZoneAgnosticReflectionOperator__build_combined_oracle()

        # Position (3, 3) is outside both geometry 0 ([1,2]x[1,2]) and geometry 1 ([0,1]x[0,1])
        for marker_val in [0, 1]:
            prep = _encode_basis_state(lattice, x=3, y=3, v=0, marker=marker_val)
            prep.compose(oracle, inplace=True)
            sv = _simulate_statevector(prep)

            probs = _get_obstacle_ancilla_value(lattice, sv)
            assert probs[0] == pytest.approx(
                1.0, abs=1e-10
            ), f"Failed for marker={marker_val}"

    def test_combined_oracle_is_self_inverse(self):
        """Applying the combined oracle twice should return to the original state."""
        lattice = _make_multi_geometry_lattice()
        operator = ABZoneAgnosticReflectionOperator(lattice)

        oracle = operator._ABZoneAgnosticReflectionOperator__build_combined_oracle()

        for marker_val in [0, 1]:
            for x, y in [(1, 1), (0, 0), (3, 3)]:
                prep = _encode_basis_state(lattice, x=x, y=y, v=0, marker=marker_val)
                prep.compose(oracle, inplace=True)
                prep.compose(oracle, inplace=True)
                sv = _simulate_statevector(prep)

                probs = _get_obstacle_ancilla_value(lattice, sv)
                assert probs[0] == pytest.approx(
                    1.0, abs=1e-10
                ), f"Not self-inverse for marker={marker_val}, pos=({x},{y})"


# =============================================================================
# Operator-level statevector tests
# =============================================================================


class TestOperatorSingleGeometry:
    """Statevector tests for the zone-agnostic reflection operator with single geometry."""

    def test_operator_obstacle_ancilla_is_clean_after_full_circuit(self):
        """After the full reflection operator, the obstacle ancilla should be |0>.

        The operator structure is O -> PermStream -> S^{-1} -> O -> S.
        After the second oracle, the obstacle ancilla should be uncomputed,
        assuming the particle's position is correctly restored.
        """
        lattice = _make_single_geometry_lattice()
        operator = ABZoneAgnosticReflectionOperator(
            lattice, shapes=lattice.shapes["bounceback"]
        )

        # Test with a position outside the obstacle
        prep = _encode_basis_state(lattice, x=0, y=0, v=0)
        prep.compose(operator.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[0] == pytest.approx(1.0, abs=1e-10)


class TestOperatorMultiGeometry:
    """Statevector tests for the zone-agnostic reflection operator with multiple geometries."""

    def test_operator_obstacle_ancilla_is_clean_outside_all_geometries(self):
        """For positions outside all geometries, obstacle ancilla should remain 0."""
        lattice = _make_multi_geometry_lattice()
        operator = ABZoneAgnosticReflectionOperator(lattice)

        for marker_val in [0, 1]:
            prep = _encode_basis_state(lattice, x=3, y=3, v=0, marker=marker_val)
            prep.compose(operator.circuit, inplace=True)
            sv = _simulate_statevector(prep)

            probs = _get_obstacle_ancilla_value(lattice, sv)
            assert probs[0] == pytest.approx(
                1.0, abs=1e-10
            ), f"Failed for marker={marker_val}"


# =============================================================================
# Backward compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests verifying that single-geometry behavior is unchanged."""

    def test_single_geometry_operator_statevector_unchanged(self):
        """The operator output for a single geometry should match regardless of code path.

        This test constructs the operator via the explicit shapes parameter
        and via None. Their outputs must match for a sample of input basis states.
        """
        lattice = _make_single_geometry_lattice()

        op_explicit = ABZoneAgnosticReflectionOperator(
            lattice, shapes=lattice.shapes["bounceback"]
        )
        op_inferred = ABZoneAgnosticReflectionOperator(lattice)

        # Sample representative positions and velocities instead of exhaustive
        for x, y in [(0, 0), (1, 1), (2, 2), (3, 0)]:
            for v in [0, 3, 5]:
                prep_a = _encode_basis_state(lattice, x=x, y=y, v=v)
                prep_a.compose(op_explicit.circuit, inplace=True)
                sv_a = _simulate_statevector(prep_a)

                prep_b = _encode_basis_state(lattice, x=x, y=y, v=v)
                prep_b.compose(op_inferred.circuit, inplace=True)
                sv_b = _simulate_statevector(prep_b)

                assert sv_a.equiv(sv_b), f"Mismatch at x={x}, y={y}, v={v}"

    def test_single_geometry_in_multi_geometry_lattice_produces_same_oracle_effect(
        self,
    ):
        """A multi-geometry lattice with one geometry should produce the same oracle marking.

        When there is only one geometry, the operator should behave identically
        to the single-geometry case (modulo the extra marker qubit).
        """
        # Single geometry lattice
        single_lattice = _make_single_geometry_lattice()
        single_oracle = ABZoneAgnosticReflectionOracle(
            single_lattice, single_lattice.shapes["bounceback"][0]
        )

        # Multi-geometry lattice with only one geometry
        multi_lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
            }
        )
        multi_lattice.set_geometries(
            [
                [
                    {
                        "shape": "cuboid",
                        "x": [1, 2],
                        "y": [1, 2],
                        "boundary": "bounceback",
                    }
                ],
            ]
        )

        # With one geometry, has_multiple_geometries() returns False
        assert not multi_lattice.has_multiple_geometries()

        # Oracle should work the same way
        multi_oracle = ABZoneAgnosticReflectionOracle(
            multi_lattice, multi_lattice.geometries[0]["bounceback"][0]
        )

        # Check that both oracles mark the same positions
        for x, y in [(1, 1), (0, 0), (2, 2), (3, 3)]:
            prep_s = _encode_basis_state(single_lattice, x=x, y=y, v=0)
            prep_s.compose(single_oracle.circuit, inplace=True)
            sv_s = _simulate_statevector(prep_s)
            probs_s = _get_obstacle_ancilla_value(single_lattice, sv_s)

            prep_m = _encode_basis_state(multi_lattice, x=x, y=y, v=0)
            prep_m.compose(multi_oracle.circuit, inplace=True)
            sv_m = _simulate_statevector(prep_m)
            probs_m = _get_obstacle_ancilla_value(multi_lattice, sv_m)

            assert probs_s[1] == pytest.approx(
                probs_m[1], abs=1e-10
            ), f"Oracle mismatch at ({x},{y})"
