import pytest

from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.ab_lattice import ABLattice


# 1D Lattices
@pytest.fixture
def dummy_1d_lattice() -> ABLattice:
    return ABLattice(
        0,
        {
            "lattice": {
                "dim": {"x": 256},
                "velocities": "D1Q3",
            },
        },
    )


@pytest.fixture
def lattice_1d_16_1_obstacle() -> ABLattice:
    return ABLattice(
        {
            "lattice": {
                "dim": {"x": 16},
                "velocities": "D1Q2"
            },
            "geometry": [
                {"shape": "cuboid", "x": [4, 6], "boundary": "bounceback"},
            ],
        },
    )



# 2D Lattices
@pytest.fixture
def dummy_2d_lattice() -> ABLattice:
    return ABLattice(
        0,
        {
            "lattice": {
                "dim": {"x": 32, "y": 32},
                "velocities": "D2Q4"
            },
        },
    )


@pytest.fixture
def lattice_2d_16x16_1_obstacle() -> ABLattice:
    return ABLattice(
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": "D2Q4"
            },
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [2, 6],
                    "y": [5, 10],
                    "boundary": "bounceback",
                },
            ],
        },
    )


# Shapes
@pytest.fixture
def simple_1d_block() -> Block:
    return Block([(5, 11)], [4], "bounceback")


@pytest.fixture
def simple_large_1d_block() -> Block:
    return Block([(2, 14)], [4], "bounceback")
