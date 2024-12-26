import pytest

from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import (
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


def test_lattice_num_qubits_1(
    lattice_2d_16x16_1_obstacle_1_timestep: SpaceTimeLattice,
):
    assert (
        lattice_2d_16x16_1_obstacle_1_timestep.properties.get_num_total_qubits() == 28
    )


def test_lattice_num_qubits_2(
    lattice_2d_16x16_1_obstacle_2_timesteps: SpaceTimeLattice,
):
    assert (
        lattice_2d_16x16_1_obstacle_2_timesteps.properties.get_num_total_qubits() == 60
    )


def test_lattice_num_velocities(dummy_lattice: SpaceTimeLattice):
    for num_timesteps in range(10):
        assert (
            dummy_lattice.properties.get_num_velocity_qubits(num_timesteps)
            == 8 * (num_timesteps * num_timesteps + num_timesteps) + 4
        )


def test_get_neighbor_indices_1_timestep(
    lattice_2d_16x16_1_obstacle_1_timestep: SpaceTimeLattice,
):
    (
        extreme_points_dict,
        intermediate_points_dict,
    ) = lattice_2d_16x16_1_obstacle_1_timestep.properties.get_neighbor_indices()

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
    (
        extreme_points_dict,
        intermediate_points_dict,
    ) = lattice_2d_16x16_1_obstacle_2_timesteps.properties.get_neighbor_indices()

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
    (
        extreme_points_dict,
        _,
    ) = lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_neighbor_indices()

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
    (
        _,
        intermediate_points_dict,
    ) = lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_neighbor_indices()
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
    assert dummy_lattice.properties.get_num_gridpoints_within_distance(1) == 5
    assert dummy_lattice.properties.get_num_gridpoints_within_distance(2) == 13
    assert dummy_lattice.properties.get_num_gridpoints_within_distance(3) == 25
    assert dummy_lattice.properties.get_num_gridpoints_within_distance(4) == 41
    assert dummy_lattice.properties.get_num_gridpoints_within_distance(5) == 61


def test_quadrants_edges(lattice_2d_16x16_1_obstacle_5_timesteps):
    for x in range(1, 10):
        assert (
            lattice_2d_16x16_1_obstacle_5_timesteps.properties.coordinates_to_quadrant(
                (x, 0)
            )
            == 0
        )
        assert (
            lattice_2d_16x16_1_obstacle_5_timesteps.properties.coordinates_to_quadrant(
                (0, x)
            )
            == 1
        )
        assert (
            lattice_2d_16x16_1_obstacle_5_timesteps.properties.coordinates_to_quadrant(
                (-x, 0)
            )
            == 2
        )
        assert (
            lattice_2d_16x16_1_obstacle_5_timesteps.properties.coordinates_to_quadrant(
                (0, -x)
            )
            == 3
        )


def test_quadrants_diagonals(lattice_2d_16x16_1_obstacle_5_timesteps):
    for x in range(1, 10):
        assert (
            lattice_2d_16x16_1_obstacle_5_timesteps.properties.coordinates_to_quadrant(
                (x, x)
            )
            == 0
        )
        assert (
            lattice_2d_16x16_1_obstacle_5_timesteps.properties.coordinates_to_quadrant(
                (-x, x)
            )
            == 1
        )
        assert (
            lattice_2d_16x16_1_obstacle_5_timesteps.properties.coordinates_to_quadrant(
                (-x, -x)
            )
            == 2
        )
        assert (
            lattice_2d_16x16_1_obstacle_5_timesteps.properties.coordinates_to_quadrant(
                (x, -x)
            )
            == 3
        )


def test_get_index_of_neighbor_origin(lattice_2d_16x16_1_obstacle_5_timesteps):
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((0, 0))
        == 0
    )


def test_get_index_of_neighbor_dist_1(lattice_2d_16x16_1_obstacle_5_timesteps):
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((1, 0))
        == 1
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((0, 1))
        == 2
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-1, 0)
        )
        == 3
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (0, -1)
        )
        == 4
    )


def test_get_index_of_neighbor_dist_2(lattice_2d_16x16_1_obstacle_5_timesteps):
    # Extreme points
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((2, 0))
        == 5
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((0, 2))
        == 7
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-2, 0)
        )
        == 9
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (0, -2)
        )
        == 11
    )

    # Intermediate points
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((1, 1))
        == 6
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-1, 1)
        )
        == 8
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-1, -1)
        )
        == 10
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (1, -1)
        )
        == 12
    )


def test_get_index_of_neighbor_dist_3(lattice_2d_16x16_1_obstacle_5_timesteps):
    # Extreme points
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((3, 0))
        == 13
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((0, 3))
        == 16
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-3, 0)
        )
        == 19
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (0, -3)
        )
        == 22
    )

    # Intermediate points
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((2, 1))
        == 14
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((1, 2))
        == 15
    )

    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-1, 2)
        )
        == 17
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-2, 1)
        )
        == 18
    )

    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-2, -1)
        )
        == 20
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-1, -2)
        )
        == 21
    )

    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (1, -2)
        )
        == 23
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (2, -1)
        )
        == 24
    )


def test_get_index_of_neighbor_dist_4(lattice_2d_16x16_1_obstacle_5_timesteps):
    # Extreme points
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((4, 0))
        == 25
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((0, 4))
        == 29
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-4, 0)
        )
        == 33
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (0, -4)
        )
        == 37
    )

    # Intermediate points
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((3, 1))
        == 26
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((2, 2))
        == 27
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((1, 3))
        == 28
    )

    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-1, 3)
        )
        == 30
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-2, 2)
        )
        == 31
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-3, 1)
        )
        == 32
    )

    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-3, -1)
        )
        == 34
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-2, -2)
        )
        == 35
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-1, -3)
        )
        == 36
    )

    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (1, -3)
        )
        == 38
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (2, -2)
        )
        == 39
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (3, -1)
        )
        == 40
    )


def test_get_index_of_neighbor_dist_5(lattice_2d_16x16_1_obstacle_5_timesteps):
    # Extreme points
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((5, 0))
        == 41
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((0, 5))
        == 46
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-5, 0)
        )
        == 51
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (0, -5)
        )
        == 56
    )

    # Intermediate points
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((4, 1))
        == 42
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((3, 2))
        == 43
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((2, 3))
        == 44
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((1, 4))
        == 45
    )

    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-1, 4)
        )
        == 47
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-2, 3)
        )
        == 48
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-3, 2)
        )
        == 49
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-4, 1)
        )
        == 50
    )

    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-4, -1)
        )
        == 52
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-3, -2)
        )
        == 53
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-2, -3)
        )
        == 54
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (-1, -4)
        )
        == 55
    )

    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (1, -4)
        )
        == 57
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (2, -3)
        )
        == 58
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (3, -2)
        )
        == 59
    )
    assert (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_index_of_neighbor(
            (4, -1)
        )
        == 60
    )


def test_streaming_line_dist_1(lattice_2d_16x16_1_obstacle_1_timestep):
    streaming_line_x_pos = (
        lattice_2d_16x16_1_obstacle_1_timestep.properties.get_streaming_lines(0, True)
    )
    streaming_line_x_neg = (
        lattice_2d_16x16_1_obstacle_1_timestep.properties.get_streaming_lines(0, False)
    )

    streaming_line_y_pos = (
        lattice_2d_16x16_1_obstacle_1_timestep.properties.get_streaming_lines(1, True)
    )
    streaming_line_y_neg = (
        lattice_2d_16x16_1_obstacle_1_timestep.properties.get_streaming_lines(1, False)
    )

    assert streaming_line_x_pos == [[1, 0, 3]]
    assert streaming_line_x_neg == [
        list(reversed(line)) for line in streaming_line_x_pos
    ]

    assert streaming_line_y_pos == [[2, 0, 4]]
    assert streaming_line_y_neg == [
        list(reversed(line)) for line in streaming_line_y_pos
    ]


def test_streaming_line_dist_2(lattice_2d_16x16_1_obstacle_2_timesteps):
    streaming_line_x_pos = (
        lattice_2d_16x16_1_obstacle_2_timesteps.properties.get_streaming_lines(0, True)
    )
    streaming_line_x_neg = (
        lattice_2d_16x16_1_obstacle_2_timesteps.properties.get_streaming_lines(0, False)
    )

    streaming_line_y_pos = (
        lattice_2d_16x16_1_obstacle_2_timesteps.properties.get_streaming_lines(1, True)
    )
    streaming_line_y_neg = (
        lattice_2d_16x16_1_obstacle_2_timesteps.properties.get_streaming_lines(1, False)
    )

    assert streaming_line_x_pos == [[12, 4, 10], [5, 1, 0, 3, 9], [6, 2, 8]]
    assert streaming_line_x_neg == [
        list(reversed(line)) for line in streaming_line_x_pos
    ]

    assert streaming_line_y_pos == [[8, 3, 10], [7, 2, 0, 4, 11], [6, 1, 12]]
    assert streaming_line_y_neg == [
        list(reversed(line)) for line in streaming_line_y_pos
    ]


def test_streaming_line_dist_5(lattice_2d_16x16_1_obstacle_5_timesteps):
    streaming_line_x_pos = (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_streaming_lines(0, True)
    )
    streaming_line_x_neg = (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_streaming_lines(0, False)
    )

    streaming_line_y_pos = (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_streaming_lines(1, True)
    )
    streaming_line_y_neg = (
        lattice_2d_16x16_1_obstacle_5_timesteps.properties.get_streaming_lines(1, False)
    )

    assert streaming_line_x_pos == [
        [57, 37, 55],
        [58, 38, 22, 36, 54],
        [59, 39, 23, 11, 21, 35, 53],
        [60, 40, 24, 12, 4, 10, 20, 34, 52],
        [41, 25, 13, 5, 1, 0, 3, 9, 19, 33, 51],
        [42, 26, 14, 6, 2, 8, 18, 32, 50],
        [43, 27, 15, 7, 17, 31, 49],
        [44, 28, 16, 30, 48],
        [45, 29, 47],
    ]
    assert streaming_line_x_neg == [
        list(reversed(line)) for line in streaming_line_x_pos
    ]

    assert streaming_line_y_neg == [
        [52, 33, 50],
        [53, 34, 19, 32, 49],
        [54, 35, 20, 9, 18, 31, 48],
        [55, 36, 21, 10, 3, 8, 17, 30, 47],
        [56, 37, 22, 11, 4, 0, 2, 7, 16, 29, 46],
        [57, 38, 23, 12, 1, 6, 15, 28, 45],
        [58, 39, 24, 5, 14, 27, 44],
        [59, 40, 13, 26, 43],
        [60, 25, 42],
    ]
    assert streaming_line_y_neg == [
        list(reversed(line)) for line in streaming_line_y_pos
    ]


def test_comparator_period_volume_bounds_no_overflow(
    lattice_2d_16x16_1_obstacle_1_timestep,
):
    volume_bounds = [(0, 5), (5, 15)]

    new_bound_list = (
        lattice_2d_16x16_1_obstacle_1_timestep.comparator_periodic_volume_bounds(
            volume_bounds
        )
    )

    for vb, nb in zip(volume_bounds, new_bound_list):
        assert vb == nb[0]
        assert nb[1] == (False, False)


def test_comparator_period_volume_bounds_overflow_x_1(
    lattice_2d_16x16_1_obstacle_1_timestep,
):
    volume_bounds = [(-5, 5), (5, 15)]

    new_bound_list = (
        lattice_2d_16x16_1_obstacle_1_timestep.comparator_periodic_volume_bounds(
            volume_bounds
        )
    )

    assert new_bound_list[0][0] == (11, 5)
    assert new_bound_list[0][1] == (True, False)
    assert new_bound_list[1][0] == (5, 15)
    assert new_bound_list[1][1] == (False, False)


def test_comparator_period_volume_bounds_overflow_x_2(
    lattice_2d_16x16_1_obstacle_1_timestep,
):
    volume_bounds = [(3, 18), (5, 15)]

    new_bound_list = (
        lattice_2d_16x16_1_obstacle_1_timestep.comparator_periodic_volume_bounds(
            volume_bounds
        )
    )

    assert new_bound_list[0][0] == (3, 2)
    assert new_bound_list[0][1] == (False, True)
    assert new_bound_list[1][0] == (5, 15)
    assert new_bound_list[1][1] == (False, False)


def test_comparator_period_volume_bounds_overflow_y_1(
    lattice_2d_16x16_1_obstacle_1_timestep,
):
    volume_bounds = [(3, 7), (-6, 5)]

    new_bound_list = (
        lattice_2d_16x16_1_obstacle_1_timestep.comparator_periodic_volume_bounds(
            volume_bounds
        )
    )

    assert new_bound_list[0][0] == (3, 7)
    assert new_bound_list[0][1] == (False, False)
    assert new_bound_list[1][0] == (10, 5)
    assert new_bound_list[1][1] == (True, False)


def test_comparator_period_volume_bounds_overflow_y_2(
    lattice_2d_16x16_1_obstacle_1_timestep,
):
    volume_bounds = [(3, 7), (9, 20)]

    new_bound_list = (
        lattice_2d_16x16_1_obstacle_1_timestep.comparator_periodic_volume_bounds(
            volume_bounds
        )
    )

    assert new_bound_list[0][0] == (3, 7)
    assert new_bound_list[0][1] == (False, False)
    assert new_bound_list[1][0] == (9, 4)
    assert new_bound_list[1][1] == (False, True)


def test_comparator_period_volume_bounds_overflow_xy_1(
    lattice_2d_16x16_1_obstacle_1_timestep,
):
    volume_bounds = [(-3, 7), (9, 20)]

    new_bound_list = (
        lattice_2d_16x16_1_obstacle_1_timestep.comparator_periodic_volume_bounds(
            volume_bounds
        )
    )

    assert new_bound_list[0][0] == (13, 7)
    assert new_bound_list[0][1] == (True, False)
    assert new_bound_list[1][0] == (9, 4)
    assert new_bound_list[1][1] == (False, True)


def test_comparator_period_volume_bounds_overflow_xy_2(
    lattice_2d_16x16_1_obstacle_1_timestep,
):
    volume_bounds = [(7, 17), (-4, 6)]

    new_bound_list = (
        lattice_2d_16x16_1_obstacle_1_timestep.comparator_periodic_volume_bounds(
            volume_bounds
        )
    )

    assert new_bound_list[0][0] == (7, 1)
    assert new_bound_list[0][1] == (False, True)
    assert new_bound_list[1][0] == (12, 6)
    assert new_bound_list[1][1] == (True, False)
