import pytest

from qlbm.lattice import SpaceTimeLattice
from qlbm.lattice.lattices.spacetime_lattice import (
    VonNeumannNeighbor,
    VonNeumannNeighborType,
)


@pytest.fixture
def dummy_lattice() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        0,
        {
            "lattice": {
                "dim": {"x": 256, "y": 256},
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
                {"x": [4, 6], "y": [3, 12], "boundary": "specular"},
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
                {"x": [4, 6], "y": [3, 12], "boundary": "specular"},
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
        },
    )


def test_lattice_num_qubis_1(
    lattice_2d_16x16_1_obstacle_1_timestep: SpaceTimeLattice,
):
    assert lattice_2d_16x16_1_obstacle_1_timestep.num_total_qubits == 28


def test_lattice_num_qubis_2(
    lattice_2d_16x16_1_obstacle_2_timesteps: SpaceTimeLattice,
):
    assert lattice_2d_16x16_1_obstacle_2_timesteps.num_total_qubits == 60


def test_lattice_num_velocities(dummy_lattice: SpaceTimeLattice):
    for num_timesteps in range(10):
        assert (
            dummy_lattice.num_required_velocity_qubits(num_timesteps)
            == 8 * (num_timesteps * num_timesteps + num_timesteps) + 4
        )


def test_get_neighbor_indices_1_timestep(
    lattice_2d_16x16_1_obstacle_1_timestep: SpaceTimeLattice,
):
    extreme_points_dict, intermediate_points_dict = (
        lattice_2d_16x16_1_obstacle_1_timestep.get_neighbor_indices()
    )

    assert len(extreme_points_dict) == 1
    assert len(intermediate_points_dict) == 0

    assert extreme_points_dict[1] == [
        VonNeumannNeighbor((1, 0), 1, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, 1), 2, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-1, 0), 3, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, -1), 4, VonNeumannNeighborType.EXTREME),
    ]


def test_get_neighbor_indices_2_timesteps(
    lattice_2d_16x16_1_obstacle_2_timesteps: SpaceTimeLattice,
):
    extreme_points_dict, intermediate_points_dict = (
        lattice_2d_16x16_1_obstacle_2_timesteps.get_neighbor_indices()
    )

    assert len(extreme_points_dict) == 2
    assert len(intermediate_points_dict) == 1

    assert extreme_points_dict[1] == [
        VonNeumannNeighbor((1, 0), 1, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, 1), 2, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-1, 0), 3, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, -1), 4, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[2] == [
        VonNeumannNeighbor((2, 0), 5, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, 2), 7, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-2, 0), 9, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, -2), 11, VonNeumannNeighborType.EXTREME),
    ]

    assert intermediate_points_dict[2] == {
        0: [VonNeumannNeighbor((1, 1), 6, VonNeumannNeighborType.INTERMEDIATE)],
        1: [VonNeumannNeighbor((-1, 1), 8, VonNeumannNeighborType.INTERMEDIATE)],
        2: [VonNeumannNeighbor((-1, -1), 10, VonNeumannNeighborType.INTERMEDIATE)],
        3: [VonNeumannNeighbor((1, -1), 12, VonNeumannNeighborType.INTERMEDIATE)],
    }


def test_get_neighbor_indices_extreme_5_timesteps(
    lattice_2d_16x16_1_obstacle_5_timesteps: SpaceTimeLattice,
):
    extreme_points_dict, _ = (
        lattice_2d_16x16_1_obstacle_5_timesteps.get_neighbor_indices()
    )

    assert len(extreme_points_dict) == 5
    # assert len(intermediate_points_dict) == 1
    assert extreme_points_dict[1] == [
        VonNeumannNeighbor((1, 0), 1, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, 1), 2, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-1, 0), 3, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, -1), 4, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[2] == [
        VonNeumannNeighbor((2, 0), 5, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, 2), 7, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-2, 0), 9, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, -2), 11, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[3] == [
        VonNeumannNeighbor((3, 0), 13, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, 3), 16, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-3, 0), 19, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, -3), 22, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[4] == [
        VonNeumannNeighbor((4, 0), 25, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, 4), 29, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-4, 0), 33, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, -4), 37, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[5] == [
        VonNeumannNeighbor((5, 0), 41, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, 5), 46, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-5, 0), 51, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((0, -5), 56, VonNeumannNeighborType.EXTREME),
    ]


def test_get_neighbor_indices_intermediate_5_timesteps(
    lattice_2d_16x16_1_obstacle_5_timesteps: SpaceTimeLattice,
):
    _, intermediate_points_dict = (
        lattice_2d_16x16_1_obstacle_5_timesteps.get_neighbor_indices()
    )
    assert intermediate_points_dict[2] == {
        0: [VonNeumannNeighbor((1, 1), 6, VonNeumannNeighborType.INTERMEDIATE)],
        1: [VonNeumannNeighbor((-1, 1), 8, VonNeumannNeighborType.INTERMEDIATE)],
        2: [VonNeumannNeighbor((-1, -1), 10, VonNeumannNeighborType.INTERMEDIATE)],
        3: [VonNeumannNeighbor((1, -1), 12, VonNeumannNeighborType.INTERMEDIATE)],
    }
    assert intermediate_points_dict[3] == {
        0: [
            VonNeumannNeighbor((2, 1), 14, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((1, 2), 15, VonNeumannNeighborType.INTERMEDIATE),
        ],
        1: [
            VonNeumannNeighbor((-1, 2), 17, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-2, 1), 18, VonNeumannNeighborType.INTERMEDIATE),
        ],
        2: [
            VonNeumannNeighbor((-2, -1), 20, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-1, -2), 21, VonNeumannNeighborType.INTERMEDIATE),
        ],
        3: [
            VonNeumannNeighbor((1, -2), 23, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((2, -1), 24, VonNeumannNeighborType.INTERMEDIATE),
        ],
    }
    assert intermediate_points_dict[4] == {
        0: [
            VonNeumannNeighbor((3, 1), 26, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((2, 2), 27, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((1, 3), 28, VonNeumannNeighborType.INTERMEDIATE),
        ],
        1: [
            VonNeumannNeighbor((-1, 3), 30, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-2, 2), 31, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-3, 1), 32, VonNeumannNeighborType.INTERMEDIATE),
        ],
        2: [
            VonNeumannNeighbor((-3, -1), 34, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-2, -2), 35, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-1, -3), 36, VonNeumannNeighborType.INTERMEDIATE),
        ],
        3: [
            VonNeumannNeighbor((1, -3), 38, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((2, -2), 39, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((3, -1), 40, VonNeumannNeighborType.INTERMEDIATE),
        ],
    }
    assert intermediate_points_dict[5] == {
        0: [
            VonNeumannNeighbor((4, 1), 42, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((3, 2), 43, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((2, 3), 44, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((1, 4), 45, VonNeumannNeighborType.INTERMEDIATE),
        ],
        1: [
            VonNeumannNeighbor((-1, 4), 47, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-2, 3), 48, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-3, 2), 49, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-4, 1), 50, VonNeumannNeighborType.INTERMEDIATE),
        ],
        2: [
            VonNeumannNeighbor((-4, -1), 52, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-3, -2), 53, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-2, -3), 54, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((-1, -4), 55, VonNeumannNeighborType.INTERMEDIATE),
        ],
        3: [
            VonNeumannNeighbor((1, -4), 57, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((2, -3), 58, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((3, -2), 59, VonNeumannNeighborType.INTERMEDIATE),
            VonNeumannNeighbor((4, -1), 60, VonNeumannNeighborType.INTERMEDIATE),
        ],
    }


def test_num_gridpoints_within_distance(dummy_lattice: SpaceTimeLattice):
    assert dummy_lattice.num_gridpoints_within_distance(1) == 5
    assert dummy_lattice.num_gridpoints_within_distance(2) == 13
    assert dummy_lattice.num_gridpoints_within_distance(3) == 25
    assert dummy_lattice.num_gridpoints_within_distance(4) == 41
    assert dummy_lattice.num_gridpoints_within_distance(5) == 61
