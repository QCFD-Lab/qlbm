import pytest

from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.utils import flatten, get_qubits_to_invert


# 1D Lattices
@pytest.fixture
def dummy_1d_lattice() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        0,
        {
            "lattice": {
                "dim": {"x": 256},
                "velocities": {"x": 2},
            },
        },
    )


@pytest.fixture
def lattice_1d_16_1_obstacle_1_timestep() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        1,
        {
            "lattice": {
                "dim": {"x": 16},
                "velocities": {"x": 2},
            },
            "geometry": [
                {"x": [4, 6], "boundary": "bounceback"},
            ],
        },
    )


@pytest.fixture
def lattice_1d_16_1_obstacle_2_timesteps() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        2,
        {
            "lattice": {
                "dim": {"x": 16},
                "velocities": {"x": 2},
            },
            "geometry": [
                {"x": [4, 6], "boundary": "bounceback"},
            ],
        },
    )


@pytest.fixture
def lattice_1d_16_1_obstacle_5_timesteps() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        5,
        {
            "lattice": {
                "dim": {"x": 16},
                "velocities": {"x": 2},
            },
        },
    )


@pytest.fixture
def volumetric_lattice_1d_16_1_obstacle_1_timestep() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        1,
        {
            "lattice": {
                "dim": {"x": 16},
                "velocities": {"x": 2},
            },
        },
        use_volumetric_ops=True,
    )


# 2D Lattices
@pytest.fixture
def dummy_lattice() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        0,
        {
            "lattice": {
                "dim": {"x": 32, "y": 32},
                "velocities": {"x": 2, "y": 2},
            },
        },
    )


@pytest.fixture
def lattice_2d_16x16_1_obstacle_1_timestep() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        1,
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": {"x": 2, "y": 2},
            },
            "geometry": [
                {"x": [2, 6], "y": [5, 10], "boundary": "bounceback"},
            ],
        },
    )


@pytest.fixture
def lattice_2d_16x16_1_obstacle_2_timesteps() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        2,
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": {"x": 2, "y": 2},
            },
            "geometry": [
                {"x": [2, 6], "y": [5, 10], "boundary": "bounceback"},
            ],
        },
    )


@pytest.fixture
def lattice_2d_16x16_1_obstacle_5_timesteps() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        5,
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": {"x": 2, "y": 2},
            },
            "geometry": [
                {"x": [2, 6], "y": [5, 10], "boundary": "bounceback"},
            ],
        },
    )

@pytest.fixture
def lattice_1d_16_1_obstacle_5_timesteps() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        5,
        {
            "lattice": {
                "dim": {"x": 16},
                "velocities": {"x": 2},
            },
            "geometry": [
                {"x": [5, 11], "boundary": "bounceback"},
            ],
        },
    )


@pytest.fixture
def lattice_1d_16_1_obstacle_5_timesteps_large_obstacle() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        5,
        {
            "lattice": {
                "dim": {"x": 16},
                "velocities": {"x": 2},
            },
            "geometry": [
                {"x": [2, 14], "boundary": "bounceback"},
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

@pytest.fixture
def simple_2d_block():
    return Block([(5, 6), (2, 10)], [4, 4], "bounceback")
