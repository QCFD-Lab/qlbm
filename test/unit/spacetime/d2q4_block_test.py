
import pytest

from qlbm.lattice.blocks import Block
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.utils import flatten, get_qubits_to_invert


@pytest.fixture
def simple_2d_block():
    return Block([(5, 6), (2, 10)], [4, 4], "bounceback")


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
def lattice_1d_1x6_1_obstacle_5_timesteps_large_obstacle() -> SpaceTimeLattice:
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


def test_block_surfaces_x_walls(simple_2d_block):
    surfaces = simple_2d_block.get_d2q4_surfaces()

    assert len(surfaces) == 2  # There are 2 dimensions
    assert len(surfaces[0]) == 2  # Each dimension has 2 walls
    assert len(surfaces[0][0]) == 9  # Each x-wall has 9 gridpoints, between 2 and 10
    assert len(surfaces[0][1]) == 9  # Each x-wall has 9 gridpoints, between 2 and 10

    assert surfaces[0][0] == [
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 8),
        (5, 9),
        (5, 10),
    ]

    assert surfaces[0][1] == [
        (6, 2),
        (6, 3),
        (6, 4),
        (6, 5),
        (6, 6),
        (6, 7),
        (6, 8),
        (6, 9),
        (6, 10),
    ]


def test_block_surfaces_y_walls(simple_2d_block):
    surfaces = simple_2d_block.get_d2q4_surfaces()

    assert len(surfaces) == 2  # There are 2 dimensions
    assert len(surfaces[1]) == 2  # Each dimension has 2 walls
    assert len(surfaces[1][0]) == 2  # Each y-wall has 2 gridpoints, 5 and 6
    assert len(surfaces[1][1]) == 2  # Each y-wall has 2 gridpoints, 5 and 6

    assert surfaces[1][0] == [(5, 2), (6, 2)]

    assert surfaces[1][1] == [(5, 10), (6, 10)]


def test_get_spacetime_reflection_data_1_timestep_gridpoints(
    simple_2d_block, lattice_2d_16x16_1_obstacle_1_timestep
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_1_timestep.properties, 1
    )

    assert len(stqbm_reflection_data) == 2 * 2 * (2 + 9)

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]
    gridponts_encoded_x_walls = gridpoints_encoded[:36]
    gridponts_encoded_y_walls = gridpoints_encoded[36:]

    assert gridponts_encoded_x_walls == flatten(
        [[(gp_x, gp_y) for gp_x in [4, 5]] for gp_y in range(2, 11)]  # Lower bound
    ) + flatten(
        [[(gp_x, gp_y) for gp_x in [7, 6]] for gp_y in range(2, 11)]
    )  # Upper bound

    assert gridponts_encoded_y_walls == flatten(
        [[(gp_x, gp_y) for gp_y in [1, 2]] for gp_x in range(5, 7)]  # Lower bound
    ) + flatten(
        [[(gp_x, gp_y) for gp_y in [11, 10]] for gp_x in range(5, 7)]
    )  # Upper bound


def test_get_spacetime_reflection_data_2_timestep_gridpoints_common(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_2_timesteps.properties, 2
    )

    assert len(stqbm_reflection_data) == 2 * 8 * (2 + 9)

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]

    bottom_corner_gridpoints = [
        (4, 2),  # Negative direction
        (3, 2),
        (4, 3),
        (4, 1),
        (5, 2),  # Positive direction
        (6, 2),
        (5, 3),
        (5, 1),
    ]

    upper_corner_gridpoints = [
        (6, 11),
        (6, 12),
        (7, 11),
        (5, 11),  # Positive direction
        (6, 10),
        (6, 9),
        (7, 10),
        (5, 10),  # Negative direction
    ]

    assert gridpoints_encoded[:8] == bottom_corner_gridpoints
    assert gridpoints_encoded[168:] == upper_corner_gridpoints


def test_get_spacetime_reflection_data_2_timestep_gridpoints_x_walls(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_2_timesteps.properties, 2
    )

    assert len(stqbm_reflection_data) == 2 * 8 * (2 + 9)

    num_x_wall_gridpoints = 9 * 8 * 2

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]

    ######
    # X wall bounds
    ######

    # The lower x wall bound on the x axis
    assert all(gp[0] <= 6 for gp in gridpoints_encoded[: num_x_wall_gridpoints // 2])
    assert all(gp[0] >= 3 for gp in gridpoints_encoded[: num_x_wall_gridpoints // 2])

    # The lower x wall bound on the y axis
    assert all(gp[1] <= 11 for gp in gridpoints_encoded[: num_x_wall_gridpoints // 2])
    assert all(gp[1] >= 1 for gp in gridpoints_encoded[: num_x_wall_gridpoints // 2])

    # The upper x wall bound on the x axis
    assert all(
        gp[0] <= 8
        for gp in gridpoints_encoded[num_x_wall_gridpoints // 2 : num_x_wall_gridpoints]
    )
    assert all(
        gp[0] >= 5
        for gp in gridpoints_encoded[num_x_wall_gridpoints // 2 : num_x_wall_gridpoints]
    )

    # The upper x wall bound on the y axis
    assert all(
        gp[1] <= 11
        for gp in gridpoints_encoded[num_x_wall_gridpoints // 2 : num_x_wall_gridpoints]
    )
    assert all(
        gp[1] >= 1
        for gp in gridpoints_encoded[num_x_wall_gridpoints // 2 : num_x_wall_gridpoints]
    )


def test_get_spacetime_reflection_data_2_timestep_gridpoints_y_walls(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_2_timesteps.properties, 2
    )

    assert len(stqbm_reflection_data) == 2 * 8 * (2 + 9)

    num_x_wall_gridpoints = 9 * 8 * 2
    num_y_wall_gridpoints = 2 * 8 * 2

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]
    ######
    # Y wall bounds
    ######

    # The lower y wall bound on the x axis
    assert all(
        gp[0] <= 6
        for gp in gridpoints_encoded[num_x_wall_gridpoints : num_y_wall_gridpoints // 2]
    )
    assert all(
        gp[0] >= 3
        for gp in gridpoints_encoded[num_x_wall_gridpoints : num_y_wall_gridpoints // 2]
    )

    # The lower y wall bound on the y axis
    assert all(
        gp[1] <= 11
        for gp in gridpoints_encoded[num_x_wall_gridpoints : num_y_wall_gridpoints // 2]
    )
    assert all(
        gp[1] >= 1
        for gp in gridpoints_encoded[num_x_wall_gridpoints : num_y_wall_gridpoints // 2]
    )

    # The upper y wall bound on the x axis
    assert all(
        gp[0] <= 7
        for gp in gridpoints_encoded[
            num_x_wall_gridpoints + num_y_wall_gridpoints // 2 :
        ]
    )
    assert all(
        gp[0] >= 4
        for gp in gridpoints_encoded[
            num_x_wall_gridpoints + num_y_wall_gridpoints // 2 :
        ]
    )

    # The upper y wall bound on the y axis
    assert all(
        gp[1] <= 12
        for gp in gridpoints_encoded[
            num_x_wall_gridpoints + num_y_wall_gridpoints // 2 :
        ]
    )
    assert all(
        gp[1] >= 0
        for gp in gridpoints_encoded[
            num_x_wall_gridpoints + num_y_wall_gridpoints // 2 :
        ]
    )


def test_get_spacetime_reflection_data_5_timestep_gridpoints_common_bottom_corner(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * (2 * 25) * (2 + 9)

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]

    bottom_corner_gridpoints_negative_main_line = [
        (4, 2),  # level 0
        (3, 2),  # level 0
        (2, 2),  # level 0
        (1, 2),  # level 0
        (0, 2),  # level 0
    ]

    bottom_corner_gridpoints_negative_up = [
        (4, 3),  # level +1
        (3, 3),  # level +1
        (2, 3),  # level +1
        (1, 3),  # level +1
        (4, 4),  # level +2
        (3, 4),  # level +2
        (2, 4),  # level +2
        (4, 5),  # level +3
        (3, 5),  # level +3
        (4, 6),  # level +4
    ]

    bottom_corner_gridpoints_negative_down = [
        (4, 1),  # level +1
        (3, 1),  # level +1
        (2, 1),  # level +1
        (1, 1),  # level +1
        (4, 0),  # level +2
        (3, 0),  # level +2
        (2, 0),  # level +2
        (4, 15),  # level +3, periodic BCs
        (3, 15),  # level +3, periodic BCs
        (4, 14),  # level +4, periodic BCs
    ]

    bottom_corner_gridpoints_positive_main_line = [
        (5, 2),  # level 0
        (6, 2),  # level 0
        (7, 2),  # level 0
        (8, 2),  # level 0
        (9, 2),  # level 0
    ]

    bottom_corner_gridpoints_positive_up = [
        (5, 3),  # level -1
        (6, 3),  # level -1
        (7, 3),  # level -1
        (8, 3),  # level -1
        (5, 4),  # level -2
        (6, 4),  # level -2
        (7, 4),  # level -2
        (5, 5),  # level -3
        (6, 5),  # level -3
        (5, 6),  # level -4
    ]

    bottom_corner_gridpoints_positive_down = [
        (5, 1),  # level -1
        (6, 1),  # level -1
        (7, 1),  # level -1
        (8, 1),  # level -1
        (5, 0),  # level -2
        (6, 0),  # level -2
        (7, 0),  # level -2
        (5, 15),  # level -3, periodic BCs
        (6, 15),  # level -3, periodic BCs
        (5, 14),  # level -4, periodic BCs
    ]

    assert (
        gridpoints_encoded[:50]
        == bottom_corner_gridpoints_negative_main_line
        + bottom_corner_gridpoints_negative_up
        + bottom_corner_gridpoints_negative_down
        + bottom_corner_gridpoints_positive_main_line
        + bottom_corner_gridpoints_positive_up
        + bottom_corner_gridpoints_positive_down
    )


def test_get_spacetime_reflection_data_5_timestep_gridpoints_common_upper_corner(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * (2 * 25) * (2 + 9)

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]

    bottom_corner_gridpoints_positive_main_line = [
        (6, 11),  # level 0
        (6, 12),  # level 0
        (6, 13),  # level 0
        (6, 14),  # level 0
        (6, 15),  # level 0
    ]

    bottom_corner_gridpoints_positive_up = [
        (7, 11),  # level +1
        (7, 12),  # level +1
        (7, 13),  # level +1
        (7, 14),  # level +1
        (8, 11),  # level +2
        (8, 12),  # level +2
        (8, 13),  # level +2
        (9, 11),  # level +3
        (9, 12),  # level +3
        (10, 11),  # level +4
    ]

    bottom_corner_gridpoints_positive_down = [
        (5, 11),  # level +1
        (5, 12),  # level +1
        (5, 13),  # level +1
        (5, 14),  # level +1
        (4, 11),  # level +2
        (4, 12),  # level +2
        (4, 13),  # level +2
        (3, 11),  # level +3
        (3, 12),  # level +3
        (2, 11),  # level +4
    ]

    bottom_corner_gridpoints_negative_main_line = [
        (6, 10),  # level 0
        (6, 9),  # level 0
        (6, 8),  # level 0
        (6, 7),  # level 0
        (6, 6),  # level 0
    ]

    bottom_corner_gridpoints_negative_up = [
        (7, 10),  # level -1
        (7, 9),  # level -1
        (7, 8),  # level -1
        (7, 7),  # level -1
        (8, 10),  # level -2
        (8, 9),  # level -2
        (8, 8),  # level -2
        (9, 10),  # level -3
        (9, 9),  # level -3
        (10, 10),  # level -4
    ]

    bottom_corner_gridpoints_negative_down = [
        (5, 10),  # level -1
        (5, 9),  # level -1
        (5, 8),  # level -1
        (5, 7),  # level -1
        (4, 10),  # level -2
        (4, 9),  # level -2
        (4, 8),  # level -2
        (3, 10),  # level -3
        (3, 9),  # level -3
        (2, 10),  # level -4
    ]

    assert (
        gridpoints_encoded[-50:]
        == bottom_corner_gridpoints_positive_main_line
        + bottom_corner_gridpoints_positive_up
        + bottom_corner_gridpoints_positive_down
        + bottom_corner_gridpoints_negative_main_line
        + bottom_corner_gridpoints_negative_up
        + bottom_corner_gridpoints_negative_down
    )


def test_get_spacetime_reflection_data_5_timestep_gridpoints_x_walls(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    num_x_wall_gridpoints = 9 * 25 * 2 * 2

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]

    ######
    # X wall bounds
    ######

    # The lower x wall bound on the x axis
    assert all(gp[0] <= 9 for gp in gridpoints_encoded[: num_x_wall_gridpoints // 2])
    assert all(gp[0] >= 0 for gp in gridpoints_encoded[: num_x_wall_gridpoints // 2])

    # The lower x wall bound on the y axis
    assert all(
        gp[1] <= 14 or gp[1] in [15]  # Periodicity from the lower bound
        for gp in gridpoints_encoded[: num_x_wall_gridpoints // 2]
    )
    assert all(gp[1] >= 0 for gp in gridpoints_encoded[: num_x_wall_gridpoints // 2])

    # The upper x wall bound on the x axis
    assert all(
        gp[0] <= 11
        for gp in gridpoints_encoded[num_x_wall_gridpoints // 2 : num_x_wall_gridpoints]
    )
    assert all(
        gp[0] >= 2
        for gp in gridpoints_encoded[num_x_wall_gridpoints // 2 : num_x_wall_gridpoints]
    )

    # The upper x wall bound on the y axis
    assert all(
        gp[1] <= 14 or gp[1] in [14, 15]  # Periodicity from the lower bound
        for gp in gridpoints_encoded[num_x_wall_gridpoints // 2 : num_x_wall_gridpoints]
    )
    assert all(
        gp[1] >= 0
        for gp in gridpoints_encoded[num_x_wall_gridpoints // 2 : num_x_wall_gridpoints]
    )


def test_get_spacetime_reflection_data_5_timestep_gridpoints_y_walls(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (9 + 2)

    num_x_wall_gridpoints = 9 * 25 * 2 * 2
    num_y_wall_gridpoints = 2 * 25 * 2 * 2

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]
    ######
    # Y wall bounds
    ######

    # The lower y wall bound on the x axis
    assert all(
        gp[0] <= 9
        for gp in gridpoints_encoded[num_x_wall_gridpoints : num_y_wall_gridpoints // 2]
    )
    assert all(
        gp[0] >= 0
        for gp in gridpoints_encoded[num_x_wall_gridpoints : num_y_wall_gridpoints // 2]
    )

    # The lower y wall bound on the y axis
    assert all(
        gp[1] <= 14 or gp[1] in [14, 15]
        for gp in gridpoints_encoded[num_x_wall_gridpoints : num_y_wall_gridpoints // 2]
    )
    assert all(
        gp[1] >= 0
        for gp in gridpoints_encoded[num_x_wall_gridpoints : num_y_wall_gridpoints // 2]
    )

    # The upper y wall bound on the x axis
    assert all(
        gp[0] <= 10
        for gp in gridpoints_encoded[
            num_x_wall_gridpoints + num_y_wall_gridpoints // 2 :
        ]
    )
    assert all(
        gp[0] >= 1
        for gp in gridpoints_encoded[
            num_x_wall_gridpoints + num_y_wall_gridpoints // 2 :
        ]
    )

    # The upper y wall bound on the y axis
    assert all(
        gp[1] <= 15
        for gp in gridpoints_encoded[
            num_x_wall_gridpoints + num_y_wall_gridpoints // 2 :
        ]
    )
    assert all(
        gp[1] >= 0
        for gp in gridpoints_encoded[
            num_x_wall_gridpoints + num_y_wall_gridpoints // 2 :
        ]
    )


def test_get_spacetime_reflection_data_1_timestep_qubits_to_invert(
    simple_2d_block, lattice_2d_16x16_1_obstacle_1_timestep
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_1_timestep.properties, 1
    )

    assert len(stqbm_reflection_data) == 2 * 2 * (2 + 9)

    qubits_to_invert = [rd.qubits_to_invert for rd in stqbm_reflection_data]
    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]

    assert (
        a
        == (
            get_qubits_to_invert(b[0], 4)
            + [4 + q for q in get_qubits_to_invert(b[1], 4)]
        )
        for a, b in zip(qubits_to_invert, gridpoints_encoded)
    )


def test_get_spacetime_reflection_data_2_timestep_qubits_to_invert(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_2_timesteps.properties, 2
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 4 * (2 + 9)

    qubits_to_invert = [rd.qubits_to_invert for rd in stqbm_reflection_data]
    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]

    assert (
        a
        == (
            get_qubits_to_invert(b[0], 4)
            + [4 + q for q in get_qubits_to_invert(b[1], 4)]
        )
        for a, b in zip(qubits_to_invert, gridpoints_encoded)
    )


def test_get_spacetime_reflection_data_5_timestep_qubits_to_invert(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    qubits_to_invert = [rd.qubits_to_invert for rd in stqbm_reflection_data]
    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]

    assert (
        a
        == (
            get_qubits_to_invert(b[0], 4)
            + [4 + q for q in get_qubits_to_invert(b[1], 4)]
        )
        for a, b in zip(qubits_to_invert, gridpoints_encoded)
    )


def test_get_spacetime_reflection_data_1_timestep_neighbor_velocity_pairs(
    simple_2d_block, lattice_2d_16x16_1_obstacle_1_timestep
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_1_timestep.properties, 1
    )

    assert len(stqbm_reflection_data) == 2 * 2 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### X walls
    negative_x_walls_points = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 2), (1, 0))
    ]
    assert len(negative_x_walls_points) == 2 * 9

    positive_x_walls_points = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 0), (3, 2))
    ]
    assert len(positive_x_walls_points) == 2 * 9

    ### Y walls
    negative_y_walls_points = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 1), (4, 3))
    ]
    assert len(negative_y_walls_points) == 2 * 2

    positive_y_walls_points = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 3), (2, 1))
    ]
    assert len(positive_y_walls_points) == 2 * 2


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d1(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_2_timesteps.properties, 2
    )

    assert len(stqbm_reflection_data) == 2 * 8 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### X walls, distance 1
    negative_x_walls_points_d1 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 2), (1, 0))
    ]
    assert len(negative_x_walls_points_d1) == 2 * 9

    positive_x_walls_points_d1 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 0), (3, 2))
    ]
    assert len(positive_x_walls_points_d1) == 2 * 9

    ### Y walls, distance 1
    negative_y_walls_points_d1 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 1), (4, 3))
    ]
    assert len(negative_y_walls_points_d1) == 2 * 2

    positive_y_walls_points_d1 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 3), (2, 1))
    ]
    assert len(positive_y_walls_points_d1) == 2 * 2


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d2(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_2_timesteps.properties, 2
    )

    assert len(stqbm_reflection_data) == 2 * 8 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### X walls, distance 2
    negative_x_walls_points_d2_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((1, 2), (5, 0))
    ]
    assert len(negative_x_walls_points_d2_lvl_0) == 2 * 9

    positive_x_walls_points_d2_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((3, 0), (9, 2))
    ]
    assert len(positive_x_walls_points_d2_lvl_0) == 2 * 9

    negative_x_walls_points_d2_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((2, 2), (6, 0))
    ]
    assert len(negative_x_walls_points_d2_lvl_1_pos) == 2 * 9

    positive_x_walls_points_d2_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((2, 0), (8, 2))
    ]
    assert len(positive_x_walls_points_d2_lvl_1_pos) == 2 * 9

    negative_x_walls_points_d2_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((4, 2), (12, 0))
    ]
    assert len(negative_x_walls_points_d2_lvl_1_neg) == 2 * 9

    positive_x_walls_points_d2_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((4, 0), (10, 2))
    ]
    assert len(positive_x_walls_points_d2_lvl_1_neg) == 2 * 9

    ### Y walls, distance 2
    negative_y_walls_points_d2_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((4, 1), (11, 3))
    ]
    assert len(negative_y_walls_points_d2_lvl_0) == 2 * 2

    positive_y_walls_points_d2_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((2, 3), (7, 1))
    ]
    assert len(positive_y_walls_points_d2_lvl_0) == 2 * 2

    negative_y_walls_points_d2_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((1, 1), (12, 3))
    ]
    assert len(negative_y_walls_points_d2_lvl_1_pos) == 2 * 2

    positive_y_walls_points_d2_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((1, 3), (6, 1))
    ]
    assert len(positive_y_walls_points_d2_lvl_1_pos) == 2 * 2

    negative_y_walls_points_d2_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((3, 1), (10, 3))
    ]
    assert len(negative_y_walls_points_d2_lvl_1_neg) == 2 * 2

    positive_y_walls_points_d2_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((3, 3), (8, 1))
    ]
    assert len(positive_y_walls_points_d2_lvl_1_neg) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d1(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### X walls, distance 1
    negative_x_walls_points_d1 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 2), (1, 0))
    ]
    assert len(negative_x_walls_points_d1) == 2 * 9

    positive_x_walls_points_d1 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 0), (3, 2))
    ]
    assert len(positive_x_walls_points_d1) == 2 * 9

    ### Y walls, distance 1
    negative_y_walls_points_d1 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 1), (4, 3))
    ]
    assert len(negative_y_walls_points_d1) == 2 * 2

    positive_y_walls_points_d1 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((0, 3), (2, 1))
    ]
    assert len(positive_y_walls_points_d1) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d2(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### X walls, distance 2
    negative_x_walls_points_d2_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((1, 2), (5, 0))
    ]
    assert len(negative_x_walls_points_d2_lvl_0) == 2 * 9

    positive_x_walls_points_d2_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((3, 0), (9, 2))
    ]
    assert len(positive_x_walls_points_d2_lvl_0) == 2 * 9

    negative_x_walls_points_d2_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((2, 2), (6, 0))
    ]
    assert len(negative_x_walls_points_d2_lvl_1_pos) == 2 * 9

    positive_x_walls_points_d2_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((2, 0), (8, 2))
    ]
    assert len(positive_x_walls_points_d2_lvl_1_pos) == 2 * 9

    negative_x_walls_points_d2_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((4, 2), (12, 0))
    ]
    assert len(negative_x_walls_points_d2_lvl_1_neg) == 2 * 9

    positive_x_walls_points_d2_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((4, 0), (10, 2))
    ]
    assert len(positive_x_walls_points_d2_lvl_1_neg) == 2 * 9

    ### Y walls, distance 2
    negative_y_walls_points_d2_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((4, 1), (11, 3))
    ]
    assert len(negative_y_walls_points_d2_lvl_0) == 2 * 2

    positive_y_walls_points_d2_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((2, 3), (7, 1))
    ]
    assert len(positive_y_walls_points_d2_lvl_0) == 2 * 2

    negative_y_walls_points_d2_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((1, 1), (12, 3))
    ]
    assert len(negative_y_walls_points_d2_lvl_1_pos) == 2 * 2

    positive_y_walls_points_d2_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((1, 3), (6, 1))
    ]
    assert len(positive_y_walls_points_d2_lvl_1_pos) == 2 * 2

    negative_y_walls_points_d2_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((3, 1), (10, 3))
    ]
    assert len(negative_y_walls_points_d2_lvl_1_neg) == 2 * 2

    positive_y_walls_points_d2_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((3, 3), (8, 1))
    ]
    assert len(positive_y_walls_points_d2_lvl_1_neg) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d3_lvl_0(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### X walls, distance 3
    negative_x_walls_points_d3_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((5, 2), (13, 0))
    ]
    assert len(negative_x_walls_points_d3_lvl_0) == 2 * 9

    positive_x_walls_points_d3_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((9, 0), (19, 2))
    ]
    assert len(positive_x_walls_points_d3_lvl_0) == 2 * 9

    ### Y walls, distance 3, level 0
    negative_y_walls_points_d3_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((11, 1), (22, 3))
    ]
    assert len(negative_y_walls_points_d3_lvl_0) == 2 * 2

    positive_y_walls_points_d3_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((7, 3), (16, 1))
    ]
    assert len(positive_y_walls_points_d3_lvl_0) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d3_lvl_1(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### D3, X wall, Level 1
    negative_x_walls_points_d3_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((6, 2), (14, 0))
    ]
    assert len(negative_x_walls_points_d3_lvl_1_pos) == 2 * 9

    positive_x_walls_points_d3_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((8, 0), (18, 2))
    ]
    assert len(positive_x_walls_points_d3_lvl_1_pos) == 2 * 9

    negative_x_walls_points_d3_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((12, 2), (24, 0))
    ]
    assert len(negative_x_walls_points_d3_lvl_1_neg) == 2 * 9

    positive_x_walls_points_d3_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((10, 0), (20, 2))
    ]
    assert len(positive_x_walls_points_d3_lvl_1_neg) == 2 * 9

    ### D3, Y Walls, Level 1
    negative_y_walls_points_d3_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((12, 1), (23, 3))
    ]
    assert len(negative_y_walls_points_d3_lvl_1_pos) == 2 * 2

    positive_y_walls_points_d3_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((6, 3), (15, 1))
    ]
    assert len(positive_y_walls_points_d3_lvl_1_pos) == 2 * 2

    negative_y_walls_points_d3_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((10, 1), (21, 3))
    ]
    assert len(negative_y_walls_points_d3_lvl_1_neg) == 2 * 2

    positive_y_walls_points_d3_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((8, 3), (17, 1))
    ]
    assert len(positive_y_walls_points_d3_lvl_1_neg) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d3_lvl_2(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### D3, X Walls, Level 2
    negative_x_walls_points_d3_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((7, 2), (15, 0))
    ]
    assert len(negative_x_walls_points_d3_lvl_2_pos) == 2 * 9

    positive_x_walls_points_d3_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((7, 0), (17, 2))
    ]
    assert len(positive_x_walls_points_d3_lvl_2_pos) == 2 * 9

    negative_x_walls_points_d3_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((11, 2), (23, 0))
    ]
    assert len(negative_x_walls_points_d3_lvl_2_neg) == 2 * 9

    positive_x_walls_points_d3_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((11, 0), (21, 2))
    ]
    assert len(positive_x_walls_points_d3_lvl_2_neg) == 2 * 9

    ### D3, Y Walls, Level 2
    negative_y_walls_points_d3_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((5, 1), (24, 3))
    ]
    assert len(negative_y_walls_points_d3_lvl_2_pos) == 2 * 2

    positive_y_walls_points_d3_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((5, 3), (14, 1))
    ]
    assert len(positive_y_walls_points_d3_lvl_2_pos) == 2 * 2

    negative_y_walls_points_d3_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((9, 1), (20, 3))
    ]
    assert len(negative_y_walls_points_d3_lvl_2_neg) == 2 * 2

    positive_y_walls_points_d3_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((9, 3), (18, 1))
    ]
    assert len(positive_y_walls_points_d3_lvl_2_neg) == 2 * 2


#################### D4 ####################


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d4_lvl_0(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### X walls, distance 4, level 0
    negative_x_walls_points_d4_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((13, 2), (25, 0))
    ]
    assert len(negative_x_walls_points_d4_lvl_0) == 2 * 9

    positive_x_walls_points_d4_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((19, 0), (33, 2))
    ]
    assert len(positive_x_walls_points_d4_lvl_0) == 2 * 9

    ### Y walls, distance 4, level 0
    negative_y_walls_points_d4_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((22, 1), (37, 3))
    ]
    assert len(negative_y_walls_points_d4_lvl_0) == 2 * 2

    positive_y_walls_points_d4_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((16, 3), (29, 1))
    ]
    assert len(positive_y_walls_points_d4_lvl_0) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d4_lvl_1(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### D4, X wall, Level 1
    negative_x_walls_points_d4_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((14, 2), (26, 0))
    ]
    assert len(negative_x_walls_points_d4_lvl_1_pos) == 2 * 9

    positive_x_walls_points_d4_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((18, 0), (32, 2))
    ]
    assert len(positive_x_walls_points_d4_lvl_1_pos) == 2 * 9

    negative_x_walls_points_d4_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((24, 2), (40, 0))
    ]
    assert len(negative_x_walls_points_d4_lvl_1_neg) == 2 * 9

    positive_x_walls_points_d4_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((20, 0), (34, 2))
    ]
    assert len(positive_x_walls_points_d4_lvl_1_neg) == 2 * 9

    ### D4, Y Walls, Level 1
    negative_y_walls_points_d4_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((23, 1), (38, 3))
    ]
    assert len(negative_y_walls_points_d4_lvl_1_pos) == 2 * 2

    positive_y_walls_points_d4_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((15, 3), (28, 1))
    ]
    assert len(positive_y_walls_points_d4_lvl_1_pos) == 2 * 2

    negative_y_walls_points_d4_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((21, 1), (36, 3))
    ]
    assert len(negative_y_walls_points_d4_lvl_1_neg) == 2 * 2

    positive_y_walls_points_d4_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((17, 3), (30, 1))
    ]
    assert len(positive_y_walls_points_d4_lvl_1_neg) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d4_lvl_2(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### D4, X Walls, Level 2
    negative_x_walls_points_d4_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((15, 2), (27, 0))
    ]
    assert len(negative_x_walls_points_d4_lvl_2_pos) == 2 * 9

    positive_x_walls_points_d4_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((17, 0), (31, 2))
    ]
    assert len(positive_x_walls_points_d4_lvl_2_pos) == 2 * 9

    negative_x_walls_points_d4_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((23, 2), (39, 0))
    ]
    assert len(negative_x_walls_points_d4_lvl_2_neg) == 2 * 9

    positive_x_walls_points_d4_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((21, 0), (35, 2))
    ]
    assert len(positive_x_walls_points_d4_lvl_2_neg) == 2 * 9

    ### D4, Y Walls, Level 2
    negative_y_walls_points_d4_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((24, 1), (39, 3))
    ]
    assert len(negative_y_walls_points_d4_lvl_2_pos) == 2 * 2

    positive_y_walls_points_d4_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((14, 3), (27, 1))
    ]
    assert len(positive_y_walls_points_d4_lvl_2_pos) == 2 * 2

    negative_y_walls_points_d4_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((20, 1), (35, 3))
    ]
    assert len(negative_y_walls_points_d4_lvl_2_neg) == 2 * 2

    positive_y_walls_points_d4_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((18, 3), (31, 1))
    ]
    assert len(positive_y_walls_points_d4_lvl_2_neg) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d4_lvl_3(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### D4, X Walls, Level 3
    negative_x_walls_points_d4_lvl_3_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((16, 2), (28, 0))
    ]
    assert len(negative_x_walls_points_d4_lvl_3_pos) == 2 * 9

    positive_x_walls_points_d4_lvl_3_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((16, 0), (30, 2))
    ]
    assert len(positive_x_walls_points_d4_lvl_3_pos) == 2 * 9

    negative_x_walls_points_d4_lvl_3_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((22, 2), (38, 0))
    ]
    assert len(negative_x_walls_points_d4_lvl_3_neg) == 2 * 9

    positive_x_walls_points_d4_lvl_3_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((22, 0), (36, 2))
    ]
    assert len(positive_x_walls_points_d4_lvl_3_neg) == 2 * 9

    ### D4, Y Walls, Level 3
    negative_y_walls_points_d4_lvl_3_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((13, 1), (40, 3))
    ]
    assert len(negative_y_walls_points_d4_lvl_3_pos) == 2 * 2

    positive_y_walls_points_d4_lvl_3_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((13, 3), (26, 1))
    ]
    assert len(positive_y_walls_points_d4_lvl_3_pos) == 2 * 2

    negative_y_walls_points_d4_lvl_3_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((19, 1), (34, 3))
    ]
    assert len(negative_y_walls_points_d4_lvl_3_neg) == 2 * 2

    positive_y_walls_points_d4_lvl_3_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((19, 3), (32, 1))
    ]
    assert len(positive_y_walls_points_d4_lvl_3_neg) == 2 * 2


################### D5 ####################


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d5_lvl_0(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### X walls, distance 5, level 0
    negative_x_walls_points_d5_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((25, 2), (41, 0))
    ]
    assert len(negative_x_walls_points_d5_lvl_0) == 2 * 9

    positive_x_walls_points_d5_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((33, 0), (51, 2))
    ]
    assert len(positive_x_walls_points_d5_lvl_0) == 2 * 9

    ### Y walls, distance 5, level 0
    negative_y_walls_points_d5_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((37, 1), (56, 3))
    ]
    assert len(negative_y_walls_points_d5_lvl_0) == 2 * 2

    positive_y_walls_points_d5_lvl_0 = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((29, 3), (46, 1))
    ]
    assert len(positive_y_walls_points_d5_lvl_0) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d5_lvl_1(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### D5, X wall, Level 1
    negative_x_walls_points_d5_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((26, 2), (42, 0))
    ]
    assert len(negative_x_walls_points_d5_lvl_1_pos) == 2 * 9

    positive_x_walls_points_d5_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((32, 0), (50, 2))
    ]
    assert len(positive_x_walls_points_d5_lvl_1_pos) == 2 * 9

    negative_x_walls_points_d5_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((40, 2), (60, 0))
    ]
    assert len(negative_x_walls_points_d5_lvl_1_neg) == 2 * 9

    positive_x_walls_points_d5_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((34, 0), (52, 2))
    ]
    assert len(positive_x_walls_points_d5_lvl_1_neg) == 2 * 9

    ### D5, Y Walls, Level 1
    negative_y_walls_points_d5_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((38, 1), (57, 3))
    ]
    assert len(negative_y_walls_points_d5_lvl_1_pos) == 2 * 2

    positive_y_walls_points_d5_lvl_1_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((28, 3), (45, 1))
    ]
    assert len(positive_y_walls_points_d5_lvl_1_pos) == 2 * 2

    negative_y_walls_points_d5_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((36, 1), (55, 3))
    ]
    assert len(negative_y_walls_points_d5_lvl_1_neg) == 2 * 2

    positive_y_walls_points_d5_lvl_1_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((30, 3), (47, 1))
    ]
    assert len(positive_y_walls_points_d5_lvl_1_neg) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d5_lvl_2(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### D5, X Walls, Level 2
    negative_x_walls_points_d5_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((27, 2), (43, 0))
    ]
    assert len(negative_x_walls_points_d5_lvl_2_pos) == 2 * 9

    positive_x_walls_points_d5_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((31, 0), (49, 2))
    ]
    assert len(positive_x_walls_points_d5_lvl_2_pos) == 2 * 9

    negative_x_walls_points_d5_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((39, 2), (59, 0))
    ]
    assert len(negative_x_walls_points_d5_lvl_2_neg) == 2 * 9

    positive_x_walls_points_d5_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((35, 0), (53, 2))
    ]
    assert len(positive_x_walls_points_d5_lvl_2_neg) == 2 * 9

    ### D5, Y Walls, Level 2
    negative_y_walls_points_d5_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((39, 1), (58, 3))
    ]
    assert len(negative_y_walls_points_d5_lvl_2_pos) == 2 * 2

    positive_y_walls_points_d5_lvl_2_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((27, 3), (44, 1))
    ]
    assert len(positive_y_walls_points_d5_lvl_2_pos) == 2 * 2

    negative_y_walls_points_d5_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((35, 1), (54, 3))
    ]
    assert len(negative_y_walls_points_d5_lvl_2_neg) == 2 * 2

    positive_y_walls_points_d5_lvl_2_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((31, 3), (48, 1))
    ]
    assert len(positive_y_walls_points_d5_lvl_2_neg) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d5_lvl_3(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### D5, X Walls, Level 3
    negative_x_walls_points_d5_lvl_3_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((28, 2), (44, 0))
    ]
    assert len(negative_x_walls_points_d5_lvl_3_pos) == 2 * 9

    positive_x_walls_points_d5_lvl_3_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((30, 0), (48, 2))
    ]
    assert len(positive_x_walls_points_d5_lvl_3_pos) == 2 * 9

    negative_x_walls_points_d5_lvl_3_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((38, 2), (58, 0))
    ]
    assert len(negative_x_walls_points_d5_lvl_3_neg) == 2 * 9

    positive_x_walls_points_d5_lvl_3_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((36, 0), (54, 2))
    ]
    assert len(positive_x_walls_points_d5_lvl_3_neg) == 2 * 9

    ### D5, Y Walls, Level 3
    negative_y_walls_points_d5_lvl_3_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((40, 1), (59, 3))
    ]
    assert len(negative_y_walls_points_d5_lvl_3_pos) == 2 * 2

    positive_y_walls_points_d5_lvl_3_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((26, 3), (43, 1))
    ]
    assert len(positive_y_walls_points_d5_lvl_3_pos) == 2 * 2

    negative_y_walls_points_d5_lvl_3_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((34, 1), (53, 3))
    ]
    assert len(negative_y_walls_points_d5_lvl_3_neg) == 2 * 2

    positive_y_walls_points_d5_lvl_3_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((32, 3), (49, 1))
    ]
    assert len(positive_y_walls_points_d5_lvl_3_neg) == 2 * 2


def test_get_spacetime_reflection_data_5_timesteps_neighbor_velocity_pairs_d5_lvl_4(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_2d_block.get_spacetime_reflection_data_d2q4(
        lattice_2d_16x16_1_obstacle_5_timesteps.properties, 5
    )

    assert len(stqbm_reflection_data) == 2 * 2 * 25 * (2 + 9)

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]

    ### D5, X Walls, Level 4
    negative_x_walls_points_d5_lvl_4_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((29, 2), (45, 0))
    ]
    assert len(negative_x_walls_points_d5_lvl_4_pos) == 2 * 9

    positive_x_walls_points_d5_lvl_4_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((29, 0), (47, 2))
    ]
    assert len(positive_x_walls_points_d5_lvl_4_pos) == 2 * 9

    negative_x_walls_points_d5_lvl_4_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((37, 2), (57, 0))
    ]
    assert len(negative_x_walls_points_d5_lvl_4_neg) == 2 * 9

    positive_x_walls_points_d5_lvl_4_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((37, 0), (55, 2))
    ]
    assert len(positive_x_walls_points_d5_lvl_4_neg) == 2 * 9

    ### D5, Y Walls, Level 4
    negative_y_walls_points_d5_lvl_4_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((25, 1), (60, 3))
    ]
    assert len(negative_y_walls_points_d5_lvl_4_pos) == 2 * 2

    positive_y_walls_points_d5_lvl_4_pos = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((25, 3), (42, 1))
    ]
    assert len(positive_y_walls_points_d5_lvl_4_pos) == 2 * 2

    negative_y_walls_points_d5_lvl_4_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((33, 1), (52, 3))
    ]
    assert len(negative_y_walls_points_d5_lvl_4_neg) == 2 * 2

    positive_y_walls_points_d5_lvl_4_neg = [
        nvp for nvp in neighbor_velocity_pairs if nvp == ((33, 3), (50, 1))
    ]
    assert len(positive_y_walls_points_d5_lvl_4_neg) == 2 * 2
