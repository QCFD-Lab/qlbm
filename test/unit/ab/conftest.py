import pytest

from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.oh_lattice import OHLattice


@pytest.fixture
def oh_lattice_d2q9_8x8() -> OHLattice:
    return OHLattice(
        {
            "lattice": {
                "dim": {"x": 8, "y": 8},
                "velocities": "d2q9",
            },
        },
    )


@pytest.fixture
def ab_lattice_d2q9_8x8() -> ABLattice:
    return ABLattice(
        {
            "lattice": {
                "dim": {"x": 8, "y": 8},
                "velocities": "d2q9",
            },
        },
    )
