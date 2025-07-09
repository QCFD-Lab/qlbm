import pytest

from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice


@pytest.fixture
def lattice_d2q4_256_8() -> LQLGALattice:
    return LQLGALattice(
        {
            "lattice": {
                "dim": {"x": 256, "y": 8},
                "velocities": {"x": 2, "y": 2},
            },
        },
    )


@pytest.fixture
def lattice_d1q2_256() -> LQLGALattice:
    return LQLGALattice(
        {
            "lattice": {
                "dim": {"x": 256},
                "velocities": {"x": 2},
            },
        },
    )
