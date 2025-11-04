import pytest

from qlbm.components import CQLBM
from qlbm.lattice import MSLattice


@pytest.fixture
def lattice_2d_16x16_1_object() -> MSLattice:
    return MSLattice(
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": {"x": 4, "y": 4},
            },
            "geometry": [
                {"shape": "cuboid", "x": [4, 6], "y": [3, 12], "boundary": "specular"},
            ],
        }
    )


def test_construction(lattice_2d_16x16_1_object: MSLattice):
    CQLBM(lattice=lattice_2d_16x16_1_object)
