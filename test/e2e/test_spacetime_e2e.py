"""End-to-end tests for the SpaceTimeQLBM algorithm.

The SpaceTimeQLBM uses a space-time encoding where velocity information for
neighboring gridpoints is pre-loaded into the register and streaming is
performed via SWAP gates.

For D1Q2 on a 16-point 1D lattice with 1 timestep (10 qubits):
  - 4 grid qubits (positions 0-15)
  - 6 velocity qubits: 2 at origin, 2 for right neighbor, 2 for left neighbor
  - Velocity 0 = positive direction (+x)
  - Velocity 1 = negative direction (-x)

After 1 streaming step, a particle at position ``x`` with velocity 0
arrives at position ``x+1``, and with velocity 1 at ``x-1``.

Only the origin velocity qubits (v0, v1) carry meaningful post-streaming data.
The neighbor velocity qubits contain residual swap artifacts.
"""

import numpy as np
import pytest

from qlbm.components.spacetime import SpaceTimeQLBM
from qlbm.components.spacetime.initial import PointWiseSpaceTimeInitialConditions
from qlbm.lattice import SpaceTimeLattice

from .utils import run_statevector


def _origin_velocities_by_position(sv, num_grid_qubits: int, num_velocities: int):
    """Extract origin velocity values keyed by grid position.

    Returns a dict ``{position: (vel_0, vel_1, ...)}`` for positions
    where at least one origin velocity qubit is set.
    """
    data = np.array(sv)
    nonzero = np.where(np.abs(data) > 1e-8)[0]
    result = {}
    grid_mask = (1 << num_grid_qubits) - 1
    for idx in nonzero:
        g = idx & grid_mask
        vels = tuple((idx >> (num_grid_qubits + v)) & 1 for v in range(num_velocities))
        if any(vels):
            result[g] = vels
    return result


# Free streaming (no geometry)
class TestSpaceTimeFreeStreaming:
    """SpaceTimeQLBM D1Q2 on a 16-point 1D lattice, 1 timestep, no obstacles (10 qubits)."""

    @pytest.fixture
    def lattice(self):
        """16-point D1Q2, 1 timestep, no obstacles."""
        return SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 16}, "velocities": "D1Q2"},
                "geometry": [],
            },
        )

    def test_lattice_qubit_count(self, lattice):
        """Verify expected register sizes."""
        assert lattice.num_total_qubits == 10

    def test_stream_right(self, lattice):
        """Particle at x=5 with vel_0 (+x) -> x=6 after 1 step."""
        ic = PointWiseSpaceTimeInitialConditions(
            lattice, grid_data=[((5,), (True, False))]
        )
        alg = SpaceTimeQLBM(lattice)

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _origin_velocities_by_position(sv, 4, 2)
        assert result == {6: (1, 0)}

    def test_stream_left(self, lattice):
        """Particle at x=10 with vel_1 (-x) -> x=9 after 1 step."""
        ic = PointWiseSpaceTimeInitialConditions(
            lattice, grid_data=[((10,), (False, True))]
        )
        alg = SpaceTimeQLBM(lattice)

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _origin_velocities_by_position(sv, 4, 2)
        assert result == {9: (0, 1)}

    def test_two_particles(self, lattice):
        """Two particles streaming independently: x=5 right and x=10 left."""
        ic = PointWiseSpaceTimeInitialConditions(
            lattice,
            grid_data=[
                ((5,), (True, False)),
                ((10,), (False, True)),
            ],
        )
        alg = SpaceTimeQLBM(lattice)

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _origin_velocities_by_position(sv, 4, 2)
        assert result == {6: (1, 0), 9: (0, 1)}

    def test_both_velocities_at_same_point(self, lattice):
        """Particle at x=8 with both vel_0 and vel_1 splits to x=7 and x=9."""
        ic = PointWiseSpaceTimeInitialConditions(
            lattice, grid_data=[((8,), (True, True))]
        )
        alg = SpaceTimeQLBM(lattice)

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _origin_velocities_by_position(sv, 4, 2)
        assert result == {9: (1, 0), 7: (0, 1)}


# ---------------------------------------------------------------------------
# Streaming with bounceback obstacle
# ---------------------------------------------------------------------------


class TestSpaceTimeBounceback:
    """SpaceTimeQLBM D1Q2 with a bounceback obstacle at x in [7, 8] (10 qubits)."""

    @pytest.fixture
    def lattice(self):
        """16-point D1Q2, 1 timestep, obstacle at x=[7,8]."""
        return SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 16}, "velocities": "D1Q2"},
                "geometry": [
                    {"shape": "cuboid", "x": [7, 8], "boundary": "bounceback"}
                ],
            },
        )

    def test_bounceback_into_wall(self, lattice):
        """Particle at x=6 with vel_0 (+x) bounces off obstacle at [7,8].

        After streaming the particle would land at x=7 which is inside
        the obstacle.  Bounceback reflects it back to x=6 with vel_1 (-x).
        """
        ic = PointWiseSpaceTimeInitialConditions(
            lattice, grid_data=[((6,), (True, False))]
        )
        alg = SpaceTimeQLBM(lattice)

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _origin_velocities_by_position(sv, 4, 2)
        assert result == {6: (0, 1)}

    def test_free_streaming_away_from_obstacle(self, lattice):
        """Particle at x=6 with vel_1 (-x) streams freely to x=5."""
        ic = PointWiseSpaceTimeInitialConditions(
            lattice, grid_data=[((6,), (False, True))]
        )
        alg = SpaceTimeQLBM(lattice)

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _origin_velocities_by_position(sv, 4, 2)
        assert result == {5: (0, 1)}

    def test_bounceback_from_other_side(self, lattice):
        """Particle at x=9 with vel_1 (-x) bounces off obstacle at [7,8].

        Would land at x=8 (inside obstacle). Reflects back to x=9 with vel_0.
        """
        ic = PointWiseSpaceTimeInitialConditions(
            lattice, grid_data=[((9,), (False, True))]
        )
        alg = SpaceTimeQLBM(lattice)

        circuit = ic.circuit.copy()
        circuit.compose(alg.circuit, inplace=True)
        sv = run_statevector(circuit)

        result = _origin_velocities_by_position(sv, 4, 2)
        assert result == {9: (1, 0)}
