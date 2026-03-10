import pytest
from qiskit import QuantumCircuit

from qlbm.components.ab.reflection.agnosotic_reflection import (
    ABZoneAgnosticReflectionOracle,
)
from qlbm.lattice import ABLattice
from qlbm.tools.exceptions import CircuitException


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


def test_ymonomial_oracle_raises_for_mismatched_y_and_result_registers():
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

    with pytest.raises(CircuitException) as excinfo:
        ABZoneAgnosticReflectionOracle(lattice, shape)

    assert (
        "YMonomial oracle is a work in progress: only configurations with equal y and monomial result register sizes are currently supported."
        == str(excinfo.value)
    )


def test_ymonomial_oracle_builds_for_supported_work_in_progress_case():
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
