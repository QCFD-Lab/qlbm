"""End-to-end tests for the ABQLBM algorithm.

The ABQLBM algorithm performs streaming followed by zone-agnostic reflection.
The D2Q9 velocity channels are indexed as:

    0: [ 0, 0]  (rest)
    1: [+1, 0]  (right)
    2: [ 0,+1]  (up)
    3: [-1, 0]  (left)
    4: [ 0,-1]  (down)
    5: [+1,+1]  (right-up)
    6: [-1,+1]  (left-up)
    7: [-1,-1]  (left-down)
    8: [+1,-1]  (right-down)
"""

import pytest

from qlbm.components.ab import ABQLBM
from qlbm.lattice import ABLattice

from .utils import (
    decode_state,
    get_nonzero_amplitudes,
    make_ab_qubit_layout,
    prepare_single_particle,
    run_statevector,
)

# Bounceback reverses velocity direction.
D2Q9_BOUNCEBACK = {
    0: 0,
    1: 3,
    2: 4,
    3: 1,
    4: 2,
    5: 7,
    6: 8,
    7: 5,
    8: 6,
}


# ---------------------------------------------------------------------------
# Free streaming (no geometry)
# ---------------------------------------------------------------------------


class TestABFreeStreaming:
    """ABQLBM on a 4x4 D2Q9 lattice without obstacles.

    With no geometry the reflection operator reduces to an identity,
    so the algorithm is pure streaming: a particle at position ``(x, y)``
    with velocity channel ``c`` moves to ``(x + vx, y + vy) mod 4``.
    """

    @pytest.fixture
    def lattice(self):
        """4x4 D2Q9 lattice with no obstacles (9 qubits)."""
        return ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q9"},
                "geometry": [],
            }
        )

    def test_lattice_qubit_count(self, lattice):
        """Verify expected register sizes."""
        assert lattice.num_dims == 2
        assert lattice.num_gridpoints == [3, 3]
        assert lattice.num_grid_qubits == 4
        assert lattice.num_velocity_qubits == 4
        assert lattice.num_total_qubits == 9

    def test_stream_right(self, lattice):
        """Channel 1 (+x): (0,0) -> (1,0) after 1 step."""
        alg = ABQLBM(lattice)
        circuit = prepare_single_particle(lattice, (0, 0), 1)
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ab_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 1
        assert decoded["g_y"] == 0
        assert decoded["v"] == 1
        assert decoded["a_o"] == 0

    def test_stream_up(self, lattice):
        """Channel 2 (+y): (0,0) -> (0,1) after 1 step."""
        alg = ABQLBM(lattice)
        circuit = prepare_single_particle(lattice, (0, 0), 2)
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ab_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 0
        assert decoded["g_y"] == 1
        assert decoded["v"] == 2
        assert decoded["a_o"] == 0

    def test_stream_diagonal(self, lattice):
        """Channel 5 (+x,+y): (0,0) -> (1,1) after 1 step."""
        alg = ABQLBM(lattice)
        circuit = prepare_single_particle(lattice, (0, 0), 5)
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ab_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 1
        assert decoded["g_y"] == 1
        assert decoded["v"] == 5
        assert decoded["a_o"] == 0

    def test_stream_left_wraps(self, lattice):
        """Channel 3 (-x): (0,0) -> (3,0) via periodic wrap after 1 step."""
        alg = ABQLBM(lattice)
        circuit = prepare_single_particle(lattice, (0, 0), 3)
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ab_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 3
        assert decoded["g_y"] == 0
        assert decoded["v"] == 3
        assert decoded["a_o"] == 0

    def test_rest_particle_stays(self, lattice):
        """Channel 0 (rest): (1,2) -> (1,2) after 1 step."""
        alg = ABQLBM(lattice)
        circuit = prepare_single_particle(lattice, (1, 2), 0)
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ab_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 1
        assert decoded["g_y"] == 2
        assert decoded["v"] == 0
        assert decoded["a_o"] == 0

    def test_two_steps_right(self, lattice):
        """Channel 1 (+x): (0,0) -> (2,0) after 2 steps."""
        alg = ABQLBM(lattice)
        circuit = prepare_single_particle(lattice, (0, 0), 1)
        circuit.compose(alg.circuit, inplace=True)
        circuit.compose(alg.circuit.copy(), inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ab_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 2
        assert decoded["g_y"] == 0
        assert decoded["v"] == 1
        assert decoded["a_o"] == 0


# Streaming with bounceback obstacle
class TestABBounceback:
    """ABQLBM on an 8x8 D2Q9 lattice with one bounceback cuboid.

    The zone-agnostic reflection algorithm works as follows. After
    streaming, if a particle ends up inside the obstacle, its velocity is
    reversed and the position is corrected so that it stays just outside the
    obstacle wall (i.e., the particle effectively reflects at the boundary).

    Analytically: ``|p, v> -> |p, -v>`` when ``p+v`` falls inside the
    obstacle, where ``-v`` is the bounceback-reversed velocity.
    """

    @pytest.fixture
    def lattice(self):
        """8x8 D2Q9 lattice with a bounceback wall at x in [3,5], y in [0,6]."""
        return ABLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "D2Q9"},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [3, 5],
                        "y": [0, 6],
                        "boundary": "bounceback",
                    }
                ],
            }
        )

    def test_lattice_qubit_count(self, lattice):
        """Verify register sizes with obstacle."""
        assert lattice.num_dims == 2
        assert lattice.num_gridpoints == [7, 7]
        assert lattice.num_grid_qubits == 6
        assert lattice.num_velocity_qubits == 4
        assert lattice.num_obstacle_qubits == 1
        assert lattice.num_comparator_qubits == 2
        assert lattice.num_total_qubits == 13

    def test_bounceback_right_into_wall(self, lattice):
        """Particle at (2,1) with v=1 (+x) bounces off the obstacle.

        After streaming the particle would land at (3,1) which is inside
        the obstacle [3,5]x[0,6]. Bounceback reflects it back to (2,1)
        with reversed velocity v=3 (-x).
        """
        alg = ABQLBM(lattice)
        circuit = prepare_single_particle(lattice, (2, 1), 1)
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ab_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 2
        assert decoded["g_y"] == 1
        assert decoded["v"] == D2Q9_BOUNCEBACK[1]
        assert decoded["a_o"] == 0
        assert decoded["a_c"] == 0

    def test_free_streaming_away_from_obstacle(self, lattice):
        """Particle at (1,1) with v=3 (-x) streams normally past the obstacle.

        Target position (0,1) is outside the obstacle.
        """
        alg = ABQLBM(lattice)
        circuit = prepare_single_particle(lattice, (1, 1), 3)
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ab_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 0
        assert decoded["g_y"] == 1
        assert decoded["v"] == 3
        assert decoded["a_o"] == 0
        assert decoded["a_c"] == 0

    def test_bounceback_then_free_streaming(self, lattice):
        """Two-step test: bounce off wall, then stream freely.

        Step 1: (2,1) v=1 -> bounces to (2,1) v=3
        Step 2: (2,1) v=3 -> streams to (1,1) v=3
        """
        alg = ABQLBM(lattice)
        circuit = prepare_single_particle(lattice, (2, 1), 1)
        circuit.compose(alg.circuit, inplace=True)
        circuit.compose(alg.circuit.copy(), inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ab_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 1
        assert decoded["g_y"] == 1
        assert decoded["v"] == 3
        assert decoded["a_o"] == 0
        assert decoded["a_c"] == 0
