"""Statevector-level tests for the standard ABReflectionOperator."""

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlbm.components.ab.reflection.standard_reflection import ABReflectionOperator
from qlbm.lattice import ABLattice
from qlbm.lattice.geometry.shapes.block import Block

_SIMULATOR = AerSimulator(method="statevector")


def _simulate_statevector(circuit: QuantumCircuit) -> Statevector:
    """Run a circuit on AerSimulator and return the final statevector."""
    qc = circuit.copy()
    qc.save_statevector()
    tqc = transpile(qc, _SIMULATOR, optimization_level=0)
    result = _SIMULATOR.run(tqc).result()
    return result.data(0)["statevector"]


def _make_single_geometry_lattice(
    dim_x=8, dim_y=8, x_bounds=(2, 5), y_bounds=(2, 5)
) -> ABLattice:
    """Create a single-geometry ABLattice with a cuboid obstacle."""
    return ABLattice(
        {
            "lattice": {"dim": {"x": dim_x, "y": dim_y}, "velocities": "d2q9"},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": list(x_bounds),
                    "y": list(y_bounds),
                    "boundary": "bounceback",
                }
            ],
        }
    )


def _make_multi_geometry_lattice() -> ABLattice:
    """Create a multi-geometry ABLattice with two cuboid configurations."""
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
                    "boundary": "bounceback",
                }
            ],
            [
                {
                    "shape": "cuboid",
                    "x": [1, 3],
                    "y": [1, 3],
                    "boundary": "bounceback",
                }
            ],
        ]
    )

    return lattice


def _encode_basis_state(lattice: ABLattice, x: int, y: int, v: int, marker: int = 0):
    """Encode a computational basis state on the lattice circuit.

    Grid positions and velocity are encoded in binary representation.
    Ancillae are initialized to 0 and the marker is set via X gates.
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
    """Extract obstacle ancilla probabilities from a statevector.

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
# Statevector: single geometry full operator
# =============================================================================


class TestStandardReflectionSingleGeometryStatevector:
    """Statevector-level verification of the standard reflection with a single geometry."""

    def test_obstacle_ancilla_clean_outside_obstacle(self):
        """Obstacle ancilla should be 0 for positions well outside the obstacle.

        For the rest velocity (v=0, stationary particles), positions far from
        the obstacle should be unaffected by the reflection operator.
        """
        lattice = _make_single_geometry_lattice()
        op = ABReflectionOperator(lattice)

        # Position (0, 0) is far from obstacle [2,5]x[2,5]
        prep = _encode_basis_state(lattice, x=0, y=0, v=0)
        prep.compose(op.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[0] == pytest.approx(1.0, abs=1e-10)

    def test_obstacle_ancilla_clean_at_corner_outside(self):
        """Obstacle ancilla should be 0 at outside corners of the obstacle.

        The gridpoints immediately adjacent to the obstacle corners
        (in the fluid domain) should have their obstacle ancilla correctly
        reset after the full reflection operator.
        """
        lattice = _make_single_geometry_lattice()
        op = ABReflectionOperator(lattice)

        # Position (1, 1) is outside the obstacle [2,5]x[2,5]
        # and is an outside corner point
        prep = _encode_basis_state(lattice, x=1, y=1, v=0)
        prep.compose(op.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[0] == pytest.approx(1.0, abs=1e-10)

    def test_operator_is_consistent_across_velocities_outside(self):
        """Obstacle ancilla should remain 0 for various velocities at positions outside.

        For positions outside the obstacle, no velocity should trigger
        the obstacle ancilla.
        """
        lattice = _make_single_geometry_lattice()
        op = ABReflectionOperator(lattice)

        for v in range(9):
            prep = _encode_basis_state(lattice, x=0, y=0, v=v)
            prep.compose(op.circuit, inplace=True)
            sv = _simulate_statevector(prep)

            probs = _get_obstacle_ancilla_value(lattice, sv)
            assert probs[0] == pytest.approx(
                1.0, abs=1e-10
            ), f"Obstacle ancilla not clean for v={v} at (0,0)"


# =============================================================================
# Statevector: set_inside_wall_ancilla_state
# =============================================================================


class TestSetInsideWallAncillaStatevector:
    """Statevector-level tests for the set_inside_wall_ancilla_state primitive.

    This primitive sets the obstacle ancilla for positions lying along the
    walls of the obstacle (excluding corners). It uses SpecularWallComparator
    to identify wall positions.
    """

    def test_wall_ancilla_set_for_interior_wall_point(self):
        """Interior wall points should have their obstacle ancilla set.

        For a [2,5]x[2,5] obstacle, position (3, 2) lies on the y=2 wall
        and is inside the obstacle. The wall ancilla state primitive should
        set the obstacle ancilla for this position.
        """
        lattice = _make_single_geometry_lattice()
        op = ABReflectionOperator(lattice)
        block: Block = lattice.shapes["bounceback"][0]  # type: ignore[assignment]

        wall_circuit = op.set_inside_wall_ancilla_state(block)

        prep = _encode_basis_state(lattice, x=3, y=2, v=0)
        prep.compose(wall_circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[1] == pytest.approx(1.0, abs=1e-10)

    def test_wall_ancilla_not_set_for_point_outside_obstacle(self):
        """Positions clearly outside the obstacle should not have ancilla set."""
        lattice = _make_single_geometry_lattice()
        op = ABReflectionOperator(lattice)
        block: Block = lattice.shapes["bounceback"][0]  # type: ignore[assignment]

        wall_circuit = op.set_inside_wall_ancilla_state(block)

        prep = _encode_basis_state(lattice, x=0, y=0, v=0)
        prep.compose(wall_circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[0] == pytest.approx(1.0, abs=1e-10)


# =============================================================================
# Statevector: multi-geometry
# =============================================================================


class TestStandardReflectionMultiGeometry:
    """Statevector tests for the standard reflection operator with multiple geometries."""

    def test_multi_geometry_obstacle_ancilla_clean_outside_all(self):
        """Positions outside all obstacles should have clean ancilla for all markers."""
        lattice = _make_multi_geometry_lattice()
        op = ABReflectionOperator(lattice)

        # (7, 7) is outside both obstacle [2,5]x[2,5] and [1,3]x[1,3]
        for marker_val in [0, 1]:
            prep = _encode_basis_state(lattice, x=7, y=7, v=0, marker=marker_val)
            prep.compose(op.circuit, inplace=True)
            sv = _simulate_statevector(prep)

            probs = _get_obstacle_ancilla_value(lattice, sv)
            assert probs[0] == pytest.approx(
                1.0, abs=1e-10
            ), f"Failed for marker={marker_val}"

    def test_multi_geometry_operator_consistent_between_explicit_and_inferred(self):
        """Operator with shapes=None should produce same statevector as explicit shapes.

        For multi-geometry, shapes are inferred from lattice.geometries.
        Passing None should give the same result as passing the grouped shapes.
        """
        lattice = _make_multi_geometry_lattice()

        op_inferred = ABReflectionOperator(lattice)

        grouped_shapes = [
            gdict["bounceback"] + gdict["specular"] for gdict in lattice.geometries
        ]
        op_explicit = ABReflectionOperator(lattice, shapes=grouped_shapes)  # type: ignore[arg-type]

        # Verify statevector equivalence at representative points
        for marker_val in [0, 1]:
            prep_a = _encode_basis_state(
                lattice, x=7, y=7, v=0, marker=marker_val
            )
            prep_a.compose(op_inferred.circuit, inplace=True)
            sv_a = _simulate_statevector(prep_a)

            prep_b = _encode_basis_state(
                lattice, x=7, y=7, v=0, marker=marker_val
            )
            prep_b.compose(op_explicit.circuit, inplace=True)
            sv_b = _simulate_statevector(prep_b)

            assert sv_a.equiv(sv_b), f"Mismatch at (7,7), marker={marker_val}"
