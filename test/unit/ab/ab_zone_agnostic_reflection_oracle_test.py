import logging

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlbm.components.ab.reflection.agnosotic_reflection import (
    ABZoneAgnosticReflectionOracle,
)
from qlbm.lattice import ABLattice
from qlbm.tools.exceptions import CircuitException

_SIMULATOR = AerSimulator(method="statevector")


def _simulate_statevector(circuit: QuantumCircuit) -> Statevector:
    qc = circuit.copy()
    qc.save_statevector()
    tqc = transpile(qc, _SIMULATOR, optimization_level=0)
    result = _SIMULATOR.run(tqc).result()
    return result.data(0)["statevector"]


def _encode_basis_state(lattice: ABLattice, x: int, y: int, v: int):
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


def _get_obstacle_ancilla_value(lattice: ABLattice, sv: Statevector) -> dict:
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


def test_ymonomial_oracle_raises_for_non_quadratic_exponent():
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "D2Q4"},
            "geometry": [
                {
                    "shape": "ymonomial",
                    "exponent": 1,
                    "comparator": "<=",
                    "boundary": "bounceback",
                }
            ],
        }
    )

    shape = lattice.shapes["bounceback"][0]

    with pytest.raises(CircuitException) as excinfo:
        ABZoneAgnosticReflectionOracle(lattice, shape)

    assert (
        "YMonomial oracle is a work in progress: only exponent=2 (x^2) is currently supported."
        == str(excinfo.value)
    )


def test_ymonomial_oracle_warns_for_mismatched_y_and_result_registers(caplog):
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "D2Q4"},
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

    shape = lattice.shapes["bounceback"][0]

    with caplog.at_level(logging.WARNING, logger="qlbm"):
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

    assert isinstance(oracle.circuit, QuantumCircuit)
    assert "zero-padding" in caplog.text


def test_ymonomial_oracle_builds_for_equal_register_case():
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

    shape = lattice.shapes["bounceback"][0]
    oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

    assert isinstance(oracle.circuit, QuantumCircuit)
    assert oracle.circuit is not None


class TestYMonomialOracleSquareGrid:
    """Statevector tests for the YMonomial oracle on a square grid (padded path)."""

    @staticmethod
    def _make_lattice(dim_x=8, dim_y=8):
        return ABLattice(
            {
                "lattice": {
                    "dim": {"x": dim_x, "y": dim_y},
                    "velocities": "D2Q4",
                },
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

    def test_marks_position_inside_shape(self):
        """Position (2, 3) satisfies y=3 <= x^2=4, so it is inside."""
        lattice = self._make_lattice()
        shape = lattice.shapes["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

        prep = _encode_basis_state(lattice, x=2, y=3, v=0)
        prep.compose(oracle.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[1] == pytest.approx(1.0, abs=1e-10)

    def test_does_not_mark_position_outside_shape(self):
        """Position (1, 2) does NOT satisfy y=2 <= x^2=1, so it is outside."""
        lattice = self._make_lattice()
        shape = lattice.shapes["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

        prep = _encode_basis_state(lattice, x=1, y=2, v=0)
        prep.compose(oracle.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[0] == pytest.approx(1.0, abs=1e-10)

    def test_marks_overflow_case(self):
        """Position (3, 7) has x^2=9 > 7 (overflow), so y <= x^2 is always true."""
        lattice = self._make_lattice()
        shape = lattice.shapes["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

        prep = _encode_basis_state(lattice, x=3, y=7, v=0)
        prep.compose(oracle.circuit, inplace=True)
        sv = _simulate_statevector(prep)

        probs = _get_obstacle_ancilla_value(lattice, sv)
        assert probs[1] == pytest.approx(1.0, abs=1e-10)

    def test_is_self_inverse(self):
        """Applying the oracle twice should return to the original state."""
        lattice = self._make_lattice()
        shape = lattice.shapes["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

        for x, y in [(2, 3), (1, 2), (3, 7), (0, 0)]:
            prep = _encode_basis_state(lattice, x=x, y=y, v=0)
            prep.compose(oracle.circuit, inplace=True)
            prep.compose(oracle.circuit, inplace=True)
            sv = _simulate_statevector(prep)

            probs = _get_obstacle_ancilla_value(lattice, sv)
            assert probs[0] == pytest.approx(1.0, abs=1e-10), (
                f"Not self-inverse at ({x}, {y})"
            )

    def test_preserves_grid_state(self):
        """Oracle should restore grid and copy/monomial qubits to original state."""
        lattice = self._make_lattice()
        shape = lattice.shapes["bounceback"][0]
        oracle = ABZoneAgnosticReflectionOracle(lattice, shape)

        prep = _encode_basis_state(lattice, x=2, y=3, v=1)
        original_sv = _simulate_statevector(prep)

        prep.compose(oracle.circuit, inplace=True)
        after_sv = _simulate_statevector(prep)

        grid_qubits = lattice.grid_index()
        np.testing.assert_allclose(
            original_sv.probabilities(grid_qubits),
            after_sv.probabilities(grid_qubits),
            atol=1e-10,
        )
