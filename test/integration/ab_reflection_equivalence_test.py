"""Integration tests verifying equivalence of segment-wise and zone-agnostic AB reflection.

For every input basis state ``|x, y, v>`` representing a physically
realizable particle (entering the obstacle from the fluid domain), both
the standard (segment-wise) and zone-agnostic reflection operators must
produce:

1. Identical output states (same particle position and velocity).
2. Identical ancilla states (both dirty or both clean).
3. Clean ancilla qubits (all comparator and obstacle ancillae reset to ``|0>``).

Only the D2Q9 discretization of ABQLBM is tested, with a single cuboid
obstacle at ``[2, 5] x [2, 5]`` on an ``8 x 8`` lattice and no marker qubits.
Test inputs are restricted to entering velocities at wall and corner
positions, since non-entering velocities at obstacle boundaries represent
physically unrealizable states.
"""

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlbm.components.ab.reflection.agnosotic_reflection import (
    ABZoneAgnosticReflectionOperator,
)
from qlbm.components.ab.reflection.standard_reflection import ABReflectionOperator
from qlbm.lattice import ABLattice

_SIMULATOR = AerSimulator(method="statevector")

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


def _make_lattice(boundary: str) -> ABLattice:
    """Create an 8x8 D2Q9 lattice with a single cuboid obstacle."""
    return ABLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [2, 5],
                    "y": [2, 5],
                    "boundary": boundary,
                }
            ],
        }
    )


def _encode_basis_state(
    lattice: ABLattice, x: int, y: int, v: int
) -> QuantumCircuit:
    """Encode the computational basis state ``|x>|y>|v>|0_ancillae>``."""
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
    return circuit


def _extract_physical_state(lattice: ABLattice, sv: Statevector):
    """Extract ``(x, y, v) -> amplitude`` and ancilla status.

    Returns
    -------
    physical_state : dict
        Mapping ``(x, y, v)`` to the sum of complex amplitudes over ancilla
        configurations (for a clean operator this is a single basis state).
    ancilla_dirty : bool
        ``True`` if any non-zero-amplitude basis state has a non-zero ancilla.
    dirty_ancilla_indices : list[int]
        Indices of ancilla qubits that were found non-zero.
    """
    grid_x_qubits = lattice.grid_index(0)
    grid_y_qubits = lattice.grid_index(1)
    vel_qubits = lattice.velocity_index()
    ancilla_qubits = (
        lattice.ancillae_comparator_index() + lattice.ancillae_obstacle_index()
    )

    physical_state: dict = {}
    ancilla_dirty = False
    dirty_ancilla_set: set = set()

    for idx, amp in enumerate(sv.data):
        if abs(amp) < 1e-10:
            continue

        x = sum(((idx >> q) & 1) << bit for bit, q in enumerate(grid_x_qubits))
        y = sum(((idx >> q) & 1) << bit for bit, q in enumerate(grid_y_qubits))
        v = sum(((idx >> q) & 1) << bit for bit, q in enumerate(vel_qubits))

        for q in ancilla_qubits:
            if (idx >> q) & 1:
                ancilla_dirty = True
                dirty_ancilla_set.add(q)

        key = (x, y, v)
        physical_state[key] = physical_state.get(key, 0.0) + amp

    return physical_state, ancilla_dirty, sorted(dirty_ancilla_set)


def _run_both(lattice, std_circuit, za_circuit, x, y, v):
    """Run both operators on the same input and return extracted states."""
    prep = _encode_basis_state(lattice, x, y, v)
    std_sv = _simulate_statevector(prep.compose(std_circuit))
    za_sv = _simulate_statevector(prep.compose(za_circuit))
    return (
        _extract_physical_state(lattice, std_sv),
        _extract_physical_state(lattice, za_sv),
    )


def _assert_equivalence(
    lattice: ABLattice,
    std_circuit: QuantumCircuit,
    za_circuit: QuantumCircuit,
    x: int,
    y: int,
    v: int,
) -> None:
    """Assert both operators produce the same output with clean ancillae.

    Checks that (a) the physical states (position + velocity) are identical,
    (b) the ancilla states agree, and (c) all ancillae are ``|0>``.
    """
    (std_phys, std_dirty, std_dirty_q), (za_phys, za_dirty, za_dirty_q) = _run_both(
        lattice, std_circuit, za_circuit, x, y, v
    )
    tag = f"(x={x}, y={y}, v={v})"

    # Physical states must match
    assert set(std_phys.keys()) == set(za_phys.keys()), (
        f"Physical states differ for {tag}: "
        f"std={sorted(std_phys.keys())}, za={sorted(za_phys.keys())}"
    )
    for key in std_phys:
        assert abs(std_phys[key]) == pytest.approx(abs(za_phys[key]), abs=1e-8), (
            f"Amplitude mismatch at {key} for {tag}: "
            f"|std|={abs(std_phys[key]):.6f}, |za|={abs(za_phys[key]):.6f}"
        )

    # Ancilla states must match
    assert std_dirty == za_dirty, (
        f"Ancilla cleanliness disagrees for {tag}: "
        f"std_dirty={std_dirty} (qubits {std_dirty_q}), "
        f"za_dirty={za_dirty} (qubits {za_dirty_q})"
    )

    # Ancillae must be clean
    assert not std_dirty, (
        f"Both operators left dirty ancillae for {tag}: qubits {std_dirty_q}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────────────────────────────────────
#
# D2Q9 velocity vectors (index -> (vx, vy)):
#   0:(0,0)  1:(+1,0)  2:(0,+1)  3:(-1,0)  4:(0,-1)
#   5:(+1,+1)  6:(-1,+1)  7:(-1,-1)  8:(+1,-1)
#
# Obstacle [2,5]x[2,5].  Entering velocities by wall:
#   Left   (x=2): +x component -> {1, 5, 8}
#   Right  (x=5): -x component -> {3, 6, 7}
#   Bottom (y=2): +y component -> {2, 5, 6}
#   Top    (y=5): -y component -> {4, 7, 8}
#
# Entering velocities by corner (union of the two adjacent walls):
#   Bottom-left  (2,2): left  | bottom = {1, 2, 5, 6, 8}
#   Bottom-right (5,2): right | bottom = {2, 3, 5, 6, 7}
#   Top-left     (2,5): left  | top    = {1, 4, 5, 7, 8}
#   Top-right    (5,5): right | top    = {3, 4, 6, 7, 8}

_ALL_V = list(range(9))

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

# D2Q9 velocity components
_D2Q9_VX = {0: 0, 1: 1, 2: 0, 3: -1, 4: 0, 5: 1, 6: -1, 7: -1, 8: 1}
_D2Q9_VY = {0: 0, 1: 0, 2: 1, 3: 0, 4: -1, 5: 1, 6: 1, 7: -1, 8: -1}

# Obstacle bounds
_OBX = (2, 5)
_OBY = (2, 5)


def _pre_streaming_outside(x: int, y: int, v: int) -> bool:
    """True if the pre-streaming position ``(x - vx, y - vy)`` is outside the obstacle.

    Only inputs whose pre-streaming position lies in the fluid domain are
    physically realizable in a well-initialised simulation.
    """
    px = x - _D2Q9_VX[v]
    py = y - _D2Q9_VY[v]
    return not (_OBX[0] <= px <= _OBX[1] and _OBY[0] <= py <= _OBY[1])


# Candidate positions covering all spatial categories
_CANDIDATE_POSITIONS = [
    # Far outside
    (0, 0),
    (7, 7),
    # Just outside each wall
    (1, 3),
    (6, 3),
    (3, 1),
    (3, 6),
    # Outside diagonal corners
    (1, 1),
    (6, 6),
    # On a wall (not corner)
    (2, 3),  # left wall
    (5, 3),  # right wall
    (3, 2),  # bottom wall
    (3, 5),  # top wall
    # On a corner
    (2, 2),  # bottom-left
    (5, 2),  # bottom-right
    (2, 5),  # top-left
    (5, 5),  # top-right
]

# Keep only velocities whose pre-streaming position is outside the obstacle.
# For fluid-domain positions this removes velocities that would require a
# particle to have originated inside the obstacle.  For wall/corner positions
# this naturally selects only the entering velocities.
_ALL_CASES = [
    (x, y, v)
    for x, y in _CANDIDATE_POSITIONS
    for v in _ALL_V
    if _pre_streaming_outside(x, y, v)
]


def _test_id(x: int, y: int, v: int) -> str:
    return f"x{x}_y{y}_v{v}_{_VEL_NAME[v]}"


_ALL_IDS = [_test_id(x, y, v) for x, y, v in _ALL_CASES]


# ──────────────────────────────────────────────────────────────────────────────
# Module-scoped fixtures (operators are expensive; build once)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bb_lattice():
    """8x8 D2Q9 lattice with a single bounceback obstacle."""
    return _make_lattice("bounceback")


@pytest.fixture(scope="module")
def bb_std_circuit(bb_lattice):
    """Standard (segment-wise) bounceback reflection circuit."""
    return ABReflectionOperator(bb_lattice).circuit


@pytest.fixture(scope="module")
def bb_za_circuit(bb_lattice):
    """Zone-agnostic bounceback reflection circuit."""
    return ABZoneAgnosticReflectionOperator(bb_lattice).circuit


@pytest.fixture(scope="module")
def sr_lattice():
    """8x8 D2Q9 lattice with a single specular obstacle."""
    return _make_lattice("specular")


@pytest.fixture(scope="module")
def sr_std_circuit(sr_lattice):
    """Standard (segment-wise) specular reflection circuit."""
    return ABReflectionOperator(sr_lattice).circuit


@pytest.fixture(scope="module")
def sr_za_circuit(sr_lattice):
    """Zone-agnostic specular reflection circuit."""
    return ABZoneAgnosticReflectionOperator(sr_lattice).circuit


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Bounce-Back
# ──────────────────────────────────────────────────────────────────────────────


class TestBounceBackEquivalence:
    """Segment-wise and zone-agnostic bounce-back must produce identical results."""

    @pytest.mark.parametrize("x,y,v", _ALL_CASES, ids=_ALL_IDS)
    def test_equivalence(self, bb_lattice, bb_std_circuit, bb_za_circuit, x, y, v):
        """Check state and ancilla equivalence for a single basis-state input."""
        _assert_equivalence(bb_lattice, bb_std_circuit, bb_za_circuit, x, y, v)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Specular
# ──────────────────────────────────────────────────────────────────────────────


class TestSpecularEquivalence:
    """Segment-wise and zone-agnostic specular must produce identical results."""

    @pytest.mark.parametrize("x,y,v", _ALL_CASES, ids=_ALL_IDS)
    def test_equivalence(self, sr_lattice, sr_std_circuit, sr_za_circuit, x, y, v):
        """Check state and ancilla equivalence for a single basis-state input."""
        _assert_equivalence(sr_lattice, sr_std_circuit, sr_za_circuit, x, y, v)
