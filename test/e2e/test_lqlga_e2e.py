"""End-to-end tests for the LQLGA algorithm.

The LQLGA (Linear Quantum Lattice Gas Algorithm) uses one qubit per velocity
channel per gridpoint, with log-depth swap-based streaming and EQC collision.

For D1Q2 on a 4-gridpoint lattice (8 qubits):
  - Gridpoints 0, 1, 2, 3
  - Each gridpoint has 2 velocity qubits: vel_0 (+x) and vel_1 (-x)
  - Qubit layout: gp0_v0, gp0_v1, gp1_v0, gp1_v1, gp2_v0, gp2_v1, gp3_v0, gp3_v1

After 1 algorithm step (collision + streaming + reflection):
  - A particle at gridpoint ``g`` with vel_0 moves to ``g+1``
  - A particle at gridpoint ``g`` with vel_1 moves to ``g-1``
  - Periodic wrapping applies at lattice boundaries
"""

from typing import Dict

import numpy as np
import pytest

from qlbm.components.lqlga import LQLGA
from qlbm.components.lqlga.initial import LQGLAInitialConditions
from qlbm.lattice import LQLGALattice

from .utils import run_statevector


def _decode_lqlga_state(sv, num_gridpoints: int, num_velocities: int):
    """Decode LQLGA statevector into per-gridpoint velocity occupancy.

    Returns a dict ``{gridpoint: {velocity: 1}}`` for occupied channels.
    """
    data = np.array(sv)
    nonzero = np.where(np.abs(data) > 1e-8)[0]
    occupied: Dict = {}
    for idx in nonzero:
        for gp in range(num_gridpoints):
            for v in range(num_velocities):
                bit = gp * num_velocities + v
                if (idx >> bit) & 1:
                    occupied.setdefault(gp, {})[v] = 1
    return occupied


# Free streaming (no geometry)
class TestLQLGAFreeStreaming:
    """LQLGA D1Q2 on a 4-gridpoint lattice, no obstacles (8 qubits)."""

    @pytest.fixture
    def lattice(self):
        """4-gridpoint D1Q2, no obstacles."""
        return LQLGALattice(
            {
                "lattice": {"dim": {"x": 4}, "velocities": "D1Q2"},
                "geometry": [],
            }
        )

    def test_lattice_qubit_count(self, lattice):
        """Verify expected register sizes."""
        assert lattice.num_total_qubits == 8

    def test_stream_right(self, lattice):
        """Particle at gp1 with vel_0 (+x) -> gp2 after 1 step."""
        alg = LQLGA(lattice)
        ic = LQGLAInitialConditions(lattice, grid_data=[((1,), (True, False))])

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _decode_lqlga_state(sv, 4, 2)
        assert result == {2: {0: 1}}

    def test_stream_left(self, lattice):
        """Particle at gp1 with vel_1 (-x) -> gp0 after 1 step."""
        alg = LQLGA(lattice)
        ic = LQGLAInitialConditions(lattice, grid_data=[((1,), (False, True))])

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _decode_lqlga_state(sv, 4, 2)
        assert result == {0: {1: 1}}

    def test_stream_wraps(self, lattice):
        """Particle at gp0 with vel_1 (-x) wraps to gp3."""
        alg = LQLGA(lattice)
        ic = LQGLAInitialConditions(lattice, grid_data=[((0,), (False, True))])

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _decode_lqlga_state(sv, 4, 2)
        assert result == {3: {1: 1}}

    def test_both_velocities_split(self, lattice):
        """Particle at gp1 with both velocities splits to gp0 and gp2."""
        alg = LQLGA(lattice)
        ic = LQGLAInitialConditions(lattice, grid_data=[((1,), (True, True))])

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _decode_lqlga_state(sv, 4, 2)
        assert result == {0: {1: 1}, 2: {0: 1}}

    def test_two_steps_right(self, lattice):
        """Particle at gp1 with vel_0 -> gp3 after 2 steps."""
        alg = LQLGA(lattice)
        ic = LQGLAInitialConditions(lattice, grid_data=[((1,), (True, False))])

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        circuit.compose(alg.circuit.copy(), inplace=True)
        sv = run_statevector(circuit)

        result = _decode_lqlga_state(sv, 4, 2)
        assert result == {3: {0: 1}}


# Streaming with bounceback obstacle
class TestLQLGABounceback:
    """LQLGA D1Q2 with a bounceback obstacle at gridpoint 3 (8 qubits)."""

    @pytest.fixture
    def lattice(self):
        """4-gridpoint D1Q2, obstacle at gp3."""
        return LQLGALattice(
            {
                "lattice": {"dim": {"x": 4}, "velocities": "D1Q2"},
                "geometry": [
                    {"shape": "cuboid", "x": [3, 3], "boundary": "bounceback"}
                ],
            }
        )

    def test_bounceback_into_wall(self, lattice):
        """Particle at gp2 with vel_0 (+x) bounces off obstacle at gp3.

        The particle stays at gp2 with reversed velocity vel_1 (-x).
        """
        alg = LQLGA(lattice)
        ic = LQGLAInitialConditions(lattice, grid_data=[((2,), (True, False))])

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _decode_lqlga_state(sv, 4, 2)
        assert result == {2: {1: 1}}

    def test_free_streaming_away_from_obstacle(self, lattice):
        """Particle at gp2 with vel_1 (-x) streams freely to gp1."""
        alg = LQLGA(lattice)
        ic = LQGLAInitialConditions(lattice, grid_data=[((2,), (False, True))])

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _decode_lqlga_state(sv, 4, 2)
        assert result == {1: {1: 1}}

    def test_bounceback_then_free_streaming(self, lattice):
        """Two-step: bounce then stream freely.

        Step 1: gp2 vel_0 -> bounces -> gp2 vel_1
        Step 2: gp2 vel_1 -> streams -> gp1 vel_1
        """
        alg = LQLGA(lattice)
        ic = LQGLAInitialConditions(lattice, grid_data=[((2,), (True, False))])

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        circuit.compose(alg.circuit.copy(), inplace=True)
        sv = run_statevector(circuit)

        result = _decode_lqlga_state(sv, 4, 2)
        assert result == {1: {1: 1}}
