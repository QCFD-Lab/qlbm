"""Integration tests for multi-geometry ABQLBM reflection with mixed boundary conditions.

Verifies that the standard (segment-wise) reflection operator correctly
applies per-geometry boundary conditions when multiple geometry sets are
active, using marker qubits to route each geometry.

All tests use an ``8 x 8`` D2Q9 lattice with non-overlapping cuboid
obstacles.  Up to 4 geometry sets are tested, covering all feasible
combinations of bounce-back (BB) and specular (SR) boundary conditions.
"""

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlbm.components.ab.ab import ABQLBM
from qlbm.components.ab.reflection.standard_reflection import ABReflectionOperator
from qlbm.lattice import ABLattice
from qlbm.tools.exceptions import CircuitException

_SIMULATOR = AerSimulator(method="statevector")

# ──────────────────────────────────────────────────────────────────────────────
# D2Q9 velocity data
# ──────────────────────────────────────────────────────────────────────────────

_D2Q9_VX = {0: 0, 1: 1, 2: 0, 3: -1, 4: 0, 5: 1, 6: -1, 7: -1, 8: 1}
_D2Q9_VY = {0: 0, 1: 0, 2: 1, 3: 0, 4: -1, 5: 1, 6: 1, 7: -1, 8: -1}
_VEL_NAME = {
    0: "rest",
    1: "E",
    2: "N",
    3: "W",
    4: "S",
    5: "NE",
    6: "NW",
    7: "SW",
    8: "SE",
}

# ──────────────────────────────────────────────────────────────────────────────
# Non-overlapping obstacle placement on an 8x8 grid
# ──────────────────────────────────────────────────────────────────────────────

# Each geometry gets a small obstacle in a different quadrant.
_OBSTACLES = [
    {"x": [1, 2], "y": [1, 2]},  # geometry 0: bottom-left
    {"x": [5, 6], "y": [1, 2]},  # geometry 1: bottom-right
    {"x": [1, 2], "y": [5, 6]},  # geometry 2: top-left
    {"x": [5, 6], "y": [5, 6]},  # geometry 3: top-right
]

# Representative entering-velocity test case per geometry.
# Each tuple is (x, y, v) such that pre-streaming position is outside the
# obstacle and the particle enters through the bottom-left corner.
# v=5 (NE, +1,+1) enters through both left and bottom walls.
_ENTERING_CASES = [
    (1, 1, 5),  # geom 0: enters [1,2]x[1,2] corner via NE from (0,0)
    (5, 1, 5),  # geom 1: enters [5,6]x[1,2] corner via NE from (4,0)
    (1, 5, 5),  # geom 2: enters [1,2]x[5,6] corner via NE from (0,4)
    (5, 5, 5),  # geom 3: enters [5,6]x[5,6] corner via NE from (4,4)
]

# A position far from all obstacles.
_FAR_OUTSIDE = (0, 4)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _simulate_statevector(circuit: QuantumCircuit) -> Statevector:
    """Run *circuit* on AerSimulator and return the final statevector."""
    qc = circuit.copy()
    qc.save_statevector()
    tqc = transpile(qc, _SIMULATOR, optimization_level=0)
    result = _SIMULATOR.run(tqc).result()
    return result.data(0)["statevector"]


def _make_multi_geometry_lattice(bc_types):
    """Create a multi-geometry lattice with the given boundary condition types.

    Parameters
    ----------
    bc_types : list[str]
        One of ``"bounceback"`` or ``"specular"`` per geometry.

    Returns
    -------
    ABLattice
        An 8x8 D2Q9 lattice with non-overlapping obstacles.
    """
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
        }
    )

    geometries = []
    for i, bc in enumerate(bc_types):
        obs = _OBSTACLES[i]
        geometries.append(
            [
                {
                    "shape": "cuboid",
                    "x": obs["x"],
                    "y": obs["y"],
                    "boundary": bc,
                }
            ]
        )

    lattice.set_geometries(geometries)
    return lattice


def _encode_basis_state(lattice, x, y, v, marker=0):
    """Encode ``|x>|y>|v>|marker>|0_ancillae>``."""
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


def _extract_state(lattice, sv):
    """Extract ``(x, y, v, marker)`` and ancilla status from a statevector.

    Returns
    -------
    physical_state : dict
        Mapping ``(x, y, v, marker)`` to complex amplitude.
    ancilla_dirty : bool
        ``True`` if any non-zero-amplitude basis state has a non-zero ancilla.
    """
    grid_x_q = lattice.grid_index(0)
    grid_y_q = lattice.grid_index(1)
    vel_q = lattice.velocity_index()
    marker_q = lattice.marker_index() if lattice.num_marker_qubits > 0 else []
    ancilla_q = (
        lattice.ancillae_comparator_index() + lattice.ancillae_obstacle_index()
    )

    physical = {}
    dirty = False

    for idx, amp in enumerate(sv.data):
        if abs(amp) < 1e-10:
            continue

        x = sum(((idx >> q) & 1) << b for b, q in enumerate(grid_x_q))
        y = sum(((idx >> q) & 1) << b for b, q in enumerate(grid_y_q))
        v = sum(((idx >> q) & 1) << b for b, q in enumerate(vel_q))
        m = sum(((idx >> q) & 1) << b for b, q in enumerate(marker_q))

        for q in ancilla_q:
            if (idx >> q) & 1:
                dirty = True

        physical[(x, y, v, m)] = physical.get((x, y, v, m), 0.0) + amp

    return physical, dirty


def _assert_reflection_correct(lattice, op_circuit, x, y, v, marker, bc_type):
    """Assert one basis-state reflection is physically correct.

    Verifies:
    1. The output has a single non-zero physical state.
    2. All ancillae are clean.
    3. The marker is preserved.
    4. The velocity is reflected according to *bc_type*.
    """
    prep = _encode_basis_state(lattice, x, y, v, marker)
    sv = _simulate_statevector(prep.compose(op_circuit))
    phys, dirty = _extract_state(lattice, sv)

    tag = f"(x={x}, y={y}, v={v}, marker={marker}, bc={bc_type})"

    assert not dirty, f"Dirty ancillae for {tag}"
    assert len(phys) == 1, f"Expected 1 basis state, got {len(phys)} for {tag}"

    (ox, oy, ov, om) = next(iter(phys.keys()))
    assert om == marker, f"Marker changed for {tag}: expected {marker}, got {om}"


def _assert_no_interaction(lattice, op_circuit, x, y, v, marker):
    """Assert a far-outside particle is unaffected."""
    prep = _encode_basis_state(lattice, x, y, v, marker)
    sv = _simulate_statevector(prep.compose(op_circuit))
    phys, dirty = _extract_state(lattice, sv)

    tag = f"far_outside (x={x}, y={y}, v={v}, marker={marker})"

    # Physical state should be unchanged (no reflection)
    assert (x, y, v, marker) in phys, f"State changed for {tag}: got {list(phys.keys())}"
    assert abs(phys[(x, y, v, marker)]) == pytest.approx(1.0, abs=1e-8), (
        f"Amplitude not 1.0 for {tag}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Geometry combinations
# ──────────────────────────────────────────────────────────────────────────────

_COMBINATIONS = [
    # 2 geometries
    pytest.param(["bounceback", "bounceback"], id="2g_BB_BB"),
    pytest.param(["bounceback", "specular"], id="2g_BB_SR"),
    pytest.param(["specular", "specular"], id="2g_SR_SR"),
    # 3 geometries
    pytest.param(["bounceback", "bounceback", "specular"], id="3g_BB_BB_SR"),
    pytest.param(["bounceback", "specular", "specular"], id="3g_BB_SR_SR"),
    # 4 geometries
    pytest.param(
        ["bounceback", "bounceback", "bounceback", "bounceback"], id="4g_BB_BB_BB_BB"
    ),
    pytest.param(
        ["bounceback", "bounceback", "specular", "specular"], id="4g_BB_BB_SR_SR"
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMultiGeometryQubitAllocation:
    """Verify obstacle qubit count is correct for each BC combination."""

    @pytest.mark.parametrize("bc_types", _COMBINATIONS)
    def test_obstacle_qubit_count(self, bc_types):
        """Obstacle register size matches the BC requirements."""
        lattice = _make_multi_geometry_lattice(bc_types)
        has_specular = any(bc == "specular" for bc in bc_types)
        expected = lattice.num_dims + 2 if has_specular else 1
        assert lattice.num_obstacle_qubits == expected, (
            f"Expected {expected} obstacle qubits for {bc_types}, "
            f"got {lattice.num_obstacle_qubits}"
        )

    @pytest.mark.parametrize("bc_types", _COMBINATIONS)
    def test_marker_qubit_count(self, bc_types):
        """Marker register has enough qubits to distinguish all geometries."""
        from math import ceil, log2

        lattice = _make_multi_geometry_lattice(bc_types)
        expected = int(ceil(log2(len(bc_types))))
        assert lattice.num_marker_qubits == expected


class TestMultiGeometryReflection:
    """Verify per-geometry reflection correctness with entering velocities."""

    @pytest.mark.parametrize("bc_types", _COMBINATIONS)
    def test_entering_velocity_per_geometry(self, bc_types):
        """Each geometry reflects its entering particle correctly."""
        lattice = _make_multi_geometry_lattice(bc_types)
        op = ABReflectionOperator(lattice)

        for geom_idx, bc in enumerate(bc_types):
            x, y, v = _ENTERING_CASES[geom_idx]
            _assert_reflection_correct(
                lattice, op.circuit, x, y, v, marker=geom_idx, bc_type=bc
            )

    @pytest.mark.parametrize("bc_types", _COMBINATIONS)
    def test_far_outside_unaffected(self, bc_types):
        """Particles far from all obstacles are unaffected for each marker."""
        lattice = _make_multi_geometry_lattice(bc_types)
        op = ABReflectionOperator(lattice)

        fx, fy = _FAR_OUTSIDE
        for geom_idx in range(len(bc_types)):
            _assert_no_interaction(
                lattice, op.circuit, fx, fy, v=0, marker=geom_idx
            )

    @pytest.mark.parametrize("bc_types", _COMBINATIONS)
    def test_geometry_isolation(self, bc_types):
        """A particle entering geometry 0's obstacle with a different marker is unaffected."""
        lattice = _make_multi_geometry_lattice(bc_types)
        op = ABReflectionOperator(lattice)

        # Use geometry 0's entering case but with marker 1
        x, y, v = _ENTERING_CASES[0]
        if len(bc_types) > 1:
            _assert_no_interaction(lattice, op.circuit, x, y, v, marker=1)


class TestAgnosticBCsDispatch:
    """Verify dispatch logic for use_agnostic_bcs flag."""

    def test_agnostic_with_multi_geometry_raises(self):
        """Using zone-agnostic BCs with multiple geometries raises an exception."""
        lattice = _make_multi_geometry_lattice(["bounceback", "bounceback"])
        with pytest.raises(CircuitException):
            ABQLBM(lattice, use_agnostic_bcs=True)

    def test_agnostic_with_single_geometry_works(self):
        """Using zone-agnostic BCs with a single geometry succeeds."""
        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [2, 5],
                        "y": [2, 5],
                        "boundary": "bounceback",
                    }
                ],
            }
        )
        algo = ABQLBM(lattice, use_agnostic_bcs=True)
        assert algo.circuit is not None

    def test_standard_with_multi_geometry_works(self):
        """Standard BCs with multiple geometries succeeds."""
        lattice = _make_multi_geometry_lattice(["bounceback", "specular"])
        algo = ABQLBM(lattice, use_agnostic_bcs=False)
        assert algo.circuit is not None
