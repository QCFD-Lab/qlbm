"""End-to-end tests for the MSQLBM algorithm.

The MSQLBM algorithm uses per-dimension velocity encoding with separate
magnitude and direction qubits.  Streaming is performed via CFL substeps.

For a 4x4 lattice with 4 discrete velocities per dimension:
  - 1 velocity magnitude qubit per dimension (values 0 or 1)
  - 1 velocity direction qubit per dimension (1 = positive, 0 = negative)
  - CFL time series ``get_time_series(4) = [[1], [1], [0, 1]]``:
    magnitude 1 streams in all 3 substeps, magnitude 0 streams once.

Therefore per algorithm step:
  - A particle with magnitude 1 moves +/-3 gridpoints.
  - A particle with magnitude 0 moves +/-1 gridpoint.
"""

import pytest

from qlbm.components.ms import MSQLBM
from qlbm.lattice import MSLattice

from .utils import (
    decode_state,
    get_nonzero_amplitudes,
    make_ms_qubit_layout,
    prepare_ms_particle,
    run_statevector,
)


# Free streaming (no geometry)
class TestMSFreeStreaming:
    """MSQLBM on a 4x4 lattice with 4 velocities/dim, no obstacles (13 qubits).

    With no geometry, the reflection operator is absent and the algorithm
    is pure streaming through CFL substeps.
    """

    @pytest.fixture
    def lattice(self):
        """4x4, 4 vel/dim, no obstacles."""
        return MSLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
                "geometry": [],
            }
        )

    def test_lattice_qubit_count(self, lattice):
        """Verify expected register sizes."""
        assert lattice.num_total_qubits == 13

    def test_stream_slow_positive(self, lattice):
        """Magnitude 0 with positive direction: (0,0) -> (1,1) after 1 step.

        Magnitude 0 streams once (last CFL substep only), moving +1 in each dim.
        """
        alg = MSQLBM(lattice)
        circuit = prepare_ms_particle(lattice, (0, 0), (0, 0), (1, 1))
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ms_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 1
        assert decoded["g_y"] == 1
        assert decoded["vd_x"] == 1
        assert decoded["vd_y"] == 1
        assert decoded["a_v"] == 0

    def test_stream_slow_negative(self, lattice):
        """Magnitude 0 with negative direction: (1,1) -> (0,0) after 1 step."""
        alg = MSQLBM(lattice)
        circuit = prepare_ms_particle(lattice, (1, 1), (0, 0), (0, 0))
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ms_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 0
        assert decoded["g_y"] == 0
        assert decoded["vd_x"] == 0
        assert decoded["vd_y"] == 0
        assert decoded["a_v"] == 0

    def test_stream_slow_negative_wraps(self, lattice):
        """Magnitude 0 with negative direction from (0,0) wraps to (3,3)."""
        alg = MSQLBM(lattice)
        circuit = prepare_ms_particle(lattice, (0, 0), (0, 0), (0, 0))
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ms_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 3
        assert decoded["g_y"] == 3

    def test_stream_fast_positive(self, lattice):
        """Magnitude 1 with positive x, magnitude 0 with positive y.

        Magnitude 1 streams 3 times in x (+3), magnitude 0 streams once in y (+1).
        (0,0) -> (3,1).
        """
        alg = MSQLBM(lattice)
        circuit = prepare_ms_particle(lattice, (0, 0), (1, 0), (1, 1))
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ms_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 3
        assert decoded["g_y"] == 1
        assert decoded["v_x"] == 1
        assert decoded["vd_x"] == 1

    def test_stream_mixed_directions(self, lattice):
        """Positive x, negative y: (1,0) -> (2,3) after 1 step.

        Magnitude 0 in both dims, streams once: x+1=2, y-1=-1=3 mod 4.
        """
        alg = MSQLBM(lattice)
        circuit = prepare_ms_particle(lattice, (1, 0), (0, 0), (1, 0))
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ms_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 2
        assert decoded["g_y"] == 3

    def test_two_steps(self, lattice):
        """Two steps with magnitude 0, positive direction: (0,0) -> (2,2)."""
        alg = MSQLBM(lattice)
        circuit = prepare_ms_particle(lattice, (0, 0), (0, 0), (1, 1))
        circuit.compose(alg.circuit, inplace=True)
        circuit.compose(alg.circuit.copy(), inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ms_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 2
        assert decoded["g_y"] == 2


# ---------------------------------------------------------------------------
# Streaming with bounceback obstacle
# ---------------------------------------------------------------------------


class TestMSBounceback:
    """MSQLBM on a 4x4 lattice with a bounceback obstacle (13 qubits).

    The obstacle spans x in [2,3], y in [0,3].  Particles streaming into
    the obstacle have their velocity direction flipped and are streamed
    back out.
    """

    @pytest.fixture
    def lattice(self):
        """4x4, 4 vel/dim, obstacle at x=[2,3], y=[0,3]."""
        return MSLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [2, 3],
                        "y": [0, 3],
                        "boundary": "bounceback",
                    }
                ],
            }
        )

    def test_bounceback_into_wall(self, lattice):
        """Particle at (1,0) with v_mag=0, vd=(+,+) bounces off obstacle.

        After streaming the particle would land at (2,1), which is inside
        the obstacle [2,3]x[0,3].  Bounceback reflects it back to (1,0)
        with flipped direction vd=(0,0) = (-,-).

        The position and velocity direction are the primary assertion.
        The obstacle ancilla may be dirty after reflection.
        """
        alg = MSQLBM(lattice)
        circuit = prepare_ms_particle(lattice, (1, 0), (0, 0), (1, 1))
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ms_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 1
        assert decoded["g_y"] == 0
        assert decoded["vd_x"] == 0
        assert decoded["vd_y"] == 0

    def test_free_streaming_away_from_obstacle(self, lattice):
        """Particle at (1,1) with vd=(-,+) streams freely away from obstacle.

        Target position (0,2) is outside the obstacle.
        """
        alg = MSQLBM(lattice)
        circuit = prepare_ms_particle(lattice, (1, 1), (0, 0), (0, 1))
        circuit.compose(alg.circuit, inplace=True)

        sv = run_statevector(circuit)
        amps = get_nonzero_amplitudes(sv)
        layout = make_ms_qubit_layout(lattice)

        assert len(amps) == 1
        decoded = decode_state(list(amps.keys())[0], layout)
        assert decoded["g_x"] == 0
        assert decoded["g_y"] == 2
        assert decoded["vd_x"] == 0
        assert decoded["vd_y"] == 1
        assert decoded["a_o"] == 0
        assert decoded["a_v"] == 0
