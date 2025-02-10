from typing import List
import pytest

from qlbm.lattice.blocks import Block, SpaceTimeVolumetricReflectionData
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


@pytest.fixture
def lattice_1d_1x6_1_obstacle_5_timesteps_large_obstacle() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        5,
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": {"x": 2, "y": 2},
            },
        },
    )


def test_volumetric_reflection_data_1_step(
    simple_2d_block, lattice_2d_16x16_1_obstacle_1_timestep
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_1_timestep.properties
        )
    )

    assert len(data) == 2 * 4  # 4 walls, 2 gridpoints relevant

    # Left wall
    assert data[0].fixed_dim == 0
    assert data[0].ranged_dims == [1]
    assert data[0].fixed_gridpoint == 4
    assert data[0].range_dimension_bounds == [simple_2d_block.bounds[1]]
    assert data[0].neighbor_velocity_pairs == ((0, 2), (1, 0))

    assert data[1].fixed_dim == 0
    assert data[1].ranged_dims == [1]
    assert data[1].fixed_gridpoint == 5
    assert data[1].range_dimension_bounds == [simple_2d_block.bounds[1]]
    assert data[1].neighbor_velocity_pairs == ((0, 0), (3, 2))

    # Right wall
    assert data[2].fixed_dim == 0
    assert data[2].ranged_dims == [1]
    assert data[2].fixed_gridpoint == 7
    assert data[2].range_dimension_bounds == [simple_2d_block.bounds[1]]
    assert data[2].neighbor_velocity_pairs == ((0, 0), (3, 2))

    assert data[3].fixed_dim == 0
    assert data[3].ranged_dims == [1]
    assert data[3].fixed_gridpoint == 6
    assert data[3].range_dimension_bounds == [simple_2d_block.bounds[1]]
    assert data[3].neighbor_velocity_pairs == ((0, 2), (1, 0))

    # Down wall
    assert data[4].fixed_dim == 1
    assert data[4].ranged_dims == [0]
    assert data[4].fixed_gridpoint == 1
    assert data[4].range_dimension_bounds == [simple_2d_block.bounds[0]]
    assert data[4].neighbor_velocity_pairs == ((0, 3), (2, 1))

    assert data[5].fixed_dim == 1
    assert data[5].ranged_dims == [0]
    assert data[5].fixed_gridpoint == 2
    assert data[5].range_dimension_bounds == [simple_2d_block.bounds[0]]
    assert data[5].neighbor_velocity_pairs == ((0, 1), (4, 3))

    # Up wall
    assert data[6].fixed_dim == 1
    assert data[6].ranged_dims == [0]
    assert data[6].fixed_gridpoint == 11
    assert data[6].range_dimension_bounds == [simple_2d_block.bounds[0]]
    assert data[6].neighbor_velocity_pairs == ((0, 1), (4, 3))

    assert data[7].fixed_dim == 1
    assert data[7].ranged_dims == [0]
    assert data[7].fixed_gridpoint == 10
    assert data[7].range_dimension_bounds == [simple_2d_block.bounds[0]]
    assert data[7].neighbor_velocity_pairs == ((0, 3), (2, 1))


def test_get_spacetime_volumetric_reflection_data_2_steps_left_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_2_timesteps.properties
        )
    )

    assert len(data) == 8 * 4  # 4 walls, 8 gridpoints relevant per wall (2t^2)

    # Left wall
    data = data[:8]

    assert all(d.fixed_dim == 0 for d in data)
    assert all(d.ranged_dims == [1] for d in data)

    assert data[0].fixed_gridpoint == 4
    assert data[0].range_dimension_bounds == [simple_2d_block.bounds[1]]
    assert data[0].neighbor_velocity_pairs == ((0, 2), (1, 0))
    assert data[0].distance_from_boundary_point == (-1, 0)

    assert data[1].fixed_gridpoint == 3
    assert data[1].range_dimension_bounds == [simple_2d_block.bounds[1]]
    assert data[1].neighbor_velocity_pairs == ((1, 2), (5, 0))
    assert data[1].distance_from_boundary_point == (-2, 0)

    assert data[2].fixed_gridpoint == 4
    assert data[2].range_dimension_bounds == [(3, 11)]
    assert data[2].neighbor_velocity_pairs == ((4, 2), (12, 0))
    assert data[2].distance_from_boundary_point == (-1, 1)

    assert data[3].fixed_gridpoint == 4
    assert data[3].range_dimension_bounds == [(1, 9)]
    assert data[3].neighbor_velocity_pairs == ((2, 2), (6, 0))
    assert data[3].distance_from_boundary_point == (-1, -1)

    assert data[4].fixed_gridpoint == 5
    assert data[4].range_dimension_bounds == [(2, 10)]
    assert data[4].neighbor_velocity_pairs == ((0, 0), (3, 2))
    assert data[4].distance_from_boundary_point == (1, 0)

    assert data[5].fixed_gridpoint == 6
    assert data[5].range_dimension_bounds == [(2, 10)]
    assert data[5].neighbor_velocity_pairs == ((3, 0), (9, 2))
    assert data[5].distance_from_boundary_point == (2, 0)

    assert data[6].fixed_gridpoint == 5
    assert data[6].range_dimension_bounds == [(3, 11)]
    assert data[6].neighbor_velocity_pairs == ((4, 0), (10, 2))
    assert data[6].distance_from_boundary_point == (1, 1)

    assert data[7].fixed_gridpoint == 5
    assert data[7].range_dimension_bounds == [(1, 9)]
    assert data[7].neighbor_velocity_pairs == ((2, 0), (8, 2))
    assert data[7].distance_from_boundary_point == (1, -1)


def test_get_spacetime_volumetric_reflection_data_2_steps_right_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_2_timesteps.properties
        )
    )

    assert len(data) == 8 * 4  # 4 walls, 8 gridpoints relevant per wall (2t^2)

    # Right wall
    data = data[8:16]
    assert all(d.fixed_dim == 0 for d in data)
    assert all(d.ranged_dims == [1] for d in data)

    assert data[0].fixed_gridpoint == 7
    assert data[0].range_dimension_bounds == [(2, 10)]
    assert data[0].distance_from_boundary_point == (1, 0)
    assert data[0].neighbor_velocity_pairs == ((0, 0), (3, 2))

    assert data[1].fixed_gridpoint == 8
    assert data[1].range_dimension_bounds == [(2, 10)]
    assert data[1].distance_from_boundary_point == (2, 0)
    assert data[1].neighbor_velocity_pairs == ((3, 0), (9, 2))

    assert data[2].fixed_gridpoint == 7
    assert data[2].range_dimension_bounds == [(3, 11)]
    assert data[2].distance_from_boundary_point == (1, 1)
    assert data[2].neighbor_velocity_pairs == ((4, 0), (10, 2))

    assert data[3].fixed_gridpoint == 7
    assert data[3].range_dimension_bounds == [(1, 9)]
    assert data[3].distance_from_boundary_point == (1, -1)
    assert data[3].neighbor_velocity_pairs == ((2, 0), (8, 2))

    assert data[4].fixed_gridpoint == 6
    assert data[4].range_dimension_bounds == [(2, 10)]
    assert data[4].distance_from_boundary_point == (-1, 0)
    assert data[4].neighbor_velocity_pairs == ((0, 2), (1, 0))

    assert data[5].fixed_gridpoint == 5
    assert data[5].range_dimension_bounds == [(2, 10)]
    assert data[5].distance_from_boundary_point == (-2, 0)
    assert data[5].neighbor_velocity_pairs == ((1, 2), (5, 0))

    assert data[6].fixed_gridpoint == 6
    assert data[6].range_dimension_bounds == [(3, 11)]
    assert data[6].distance_from_boundary_point == (-1, 1)
    assert data[6].neighbor_velocity_pairs == ((4, 2), (12, 0))

    assert data[7].fixed_gridpoint == 6
    assert data[7].range_dimension_bounds == [(1, 9)]
    assert data[7].distance_from_boundary_point == (-1, -1)
    assert data[7].neighbor_velocity_pairs == ((2, 2), (6, 0))


def test_get_spacetime_volumetric_reflection_data_2_steps_bottom_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_2_timesteps.properties
        )
    )

    assert len(data) == 8 * 4  # 4 walls, 8 gridpoints relevant per wall (2t^2)

    # Bottom wall
    data = data[16:24]

    assert all(d.fixed_dim == 1 for d in data)
    assert all(d.ranged_dims == [0] for d in data)

    assert data[0].fixed_gridpoint == 1
    assert data[0].range_dimension_bounds == [(5, 6)]
    assert data[0].neighbor_velocity_pairs == ((0, 3), (2, 1))
    assert data[0].distance_from_boundary_point == (0, -1)

    assert data[1].fixed_gridpoint == 0
    assert data[1].range_dimension_bounds == [(5, 6)]
    assert data[1].neighbor_velocity_pairs == ((2, 3), (7, 1))
    assert data[1].distance_from_boundary_point == (0, -2)

    assert data[2].fixed_gridpoint == 1
    assert data[2].range_dimension_bounds == [(6, 7)]
    assert data[2].neighbor_velocity_pairs == ((3, 3), (8, 1))
    assert data[2].distance_from_boundary_point == (1, -1)

    assert data[3].fixed_gridpoint == 1
    assert data[3].range_dimension_bounds == [(4, 5)]
    assert data[3].neighbor_velocity_pairs == ((1, 3), (6, 1))
    assert data[3].distance_from_boundary_point == (-1, -1)

    assert data[4].fixed_gridpoint == 2
    assert data[4].range_dimension_bounds == [(5, 6)]
    assert data[4].neighbor_velocity_pairs == ((0, 1), (4, 3))
    assert data[4].distance_from_boundary_point == (0, 1)

    assert data[5].fixed_gridpoint == 3
    assert data[5].range_dimension_bounds == [(5, 6)]
    assert data[5].neighbor_velocity_pairs == ((4, 1), (11, 3))
    assert data[5].distance_from_boundary_point == (0, 2)

    assert data[6].fixed_gridpoint == 2
    assert data[6].range_dimension_bounds == [(6, 7)]
    assert data[6].neighbor_velocity_pairs == ((3, 1), (10, 3))
    assert data[6].distance_from_boundary_point == (1, 1)

    assert data[7].fixed_gridpoint == 2
    assert data[7].range_dimension_bounds == [(4, 5)]
    assert data[7].neighbor_velocity_pairs == ((1, 1), (12, 3))
    assert data[7].distance_from_boundary_point == (-1, 1)


def test_get_spacetime_volumetric_reflection_data_2_steps_top_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_2_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_2_timesteps.properties
        )
    )

    assert len(data) == 8 * 4  # 4 walls, 8 gridpoints relevant per wall (2t^2)

    # Bottom wall
    data = data[24:]

    assert all(d.fixed_dim == 1 for d in data)
    assert all(d.ranged_dims == [0] for d in data)

    assert data[0].fixed_gridpoint == 11
    assert data[0].range_dimension_bounds == [(5, 6)]
    assert data[0].neighbor_velocity_pairs == ((0, 1), (4, 3))
    assert data[0].distance_from_boundary_point == (0, 1)

    assert data[1].fixed_gridpoint == 12
    assert data[1].range_dimension_bounds == [(5, 6)]
    assert data[1].neighbor_velocity_pairs == ((4, 1), (11, 3))
    assert data[1].distance_from_boundary_point == (0, 2)

    assert data[2].fixed_gridpoint == 11
    assert data[2].range_dimension_bounds == [(6, 7)]
    assert data[2].neighbor_velocity_pairs == ((3, 1), (10, 3))
    assert data[2].distance_from_boundary_point == (1, 1)

    assert data[3].fixed_gridpoint == 11
    assert data[3].range_dimension_bounds == [(4, 5)]
    assert data[3].neighbor_velocity_pairs == ((1, 1), (12, 3))
    assert data[3].distance_from_boundary_point == (-1, 1)

    assert data[4].fixed_gridpoint == 10
    assert data[4].range_dimension_bounds == [(5, 6)]
    assert data[4].neighbor_velocity_pairs == ((0, 3), (2, 1))
    assert data[4].distance_from_boundary_point == (0, -1)

    assert data[5].fixed_gridpoint == 9
    assert data[5].range_dimension_bounds == [(5, 6)]
    assert data[5].neighbor_velocity_pairs == ((2, 3), (7, 1))
    assert data[5].distance_from_boundary_point == (0, -2)

    assert data[6].fixed_gridpoint == 10
    assert data[6].range_dimension_bounds == [(6, 7)]
    assert data[6].neighbor_velocity_pairs == ((3, 3), (8, 1))
    assert data[6].distance_from_boundary_point == (1, -1)

    assert data[7].fixed_gridpoint == 10
    assert data[7].range_dimension_bounds == [(4, 5)]
    assert data[7].neighbor_velocity_pairs == ((1, 3), (6, 1))
    assert data[7].distance_from_boundary_point == (-1, -1)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d1_left_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    one_distance_data = [
        d
        for d in data[:50]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 1)
    ]
    assert len(one_distance_data) == 2

    assert one_distance_data[0].fixed_gridpoint == 4
    assert one_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert one_distance_data[0].neighbor_velocity_pairs == ((0, 2), (1, 0))
    assert one_distance_data[0].distance_from_boundary_point == (-1, 0)

    assert one_distance_data[1].fixed_gridpoint == 5
    assert one_distance_data[1].range_dimension_bounds == [(2, 10)]
    assert one_distance_data[1].neighbor_velocity_pairs == ((0, 0), (3, 2))
    assert one_distance_data[1].distance_from_boundary_point == (1, 0)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d1_right_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    one_distance_data = [
        d
        for d in data[50:100]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 1)
    ]
    assert len(one_distance_data) == 2

    assert one_distance_data[0].fixed_gridpoint == 7
    assert one_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert one_distance_data[0].neighbor_velocity_pairs == ((0, 0), (3, 2))
    assert one_distance_data[0].distance_from_boundary_point == (1, 0)

    assert one_distance_data[1].fixed_gridpoint == 6
    assert one_distance_data[1].range_dimension_bounds == [(2, 10)]
    assert one_distance_data[1].neighbor_velocity_pairs == ((0, 2), (1, 0))
    assert one_distance_data[1].distance_from_boundary_point == (-1, 0)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d1_bottom_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    one_distance_data = [
        d
        for d in data[100:150]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 1)
    ]
    assert len(one_distance_data) == 2

    assert one_distance_data[0].fixed_gridpoint == 1
    assert one_distance_data[0].range_dimension_bounds == [(5, 6)]
    assert one_distance_data[0].neighbor_velocity_pairs == ((0, 3), (2, 1))
    assert one_distance_data[0].distance_from_boundary_point == (0, -1)

    assert one_distance_data[1].fixed_gridpoint == 2
    assert one_distance_data[1].range_dimension_bounds == [(5, 6)]
    assert one_distance_data[1].neighbor_velocity_pairs == ((0, 1), (4, 3))
    assert one_distance_data[1].distance_from_boundary_point == (0, 1)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d1_top_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    one_distance_data = [
        d
        for d in data[150:]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 1)
    ]
    assert len(one_distance_data) == 2

    assert one_distance_data[0].fixed_gridpoint == 11
    assert one_distance_data[0].range_dimension_bounds == [(5, 6)]
    assert one_distance_data[0].neighbor_velocity_pairs == ((0, 1), (4, 3))
    assert one_distance_data[0].distance_from_boundary_point == (0, 1)

    assert one_distance_data[1].fixed_gridpoint == 10
    assert one_distance_data[1].range_dimension_bounds == [(5, 6)]
    assert one_distance_data[1].neighbor_velocity_pairs == ((0, 3), (2, 1))
    assert one_distance_data[1].distance_from_boundary_point == (0, -1)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d2_left_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    one_distance_data = [
        d
        for d in data[:50]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 2)
    ]
    assert len(one_distance_data) == 6

    assert all(d.fixed_dim == 0 for d in one_distance_data)
    assert all(d.ranged_dims == [1] for d in one_distance_data)

    assert one_distance_data[0].fixed_gridpoint == 3
    assert one_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert one_distance_data[0].neighbor_velocity_pairs == ((1, 2), (5, 0))
    assert one_distance_data[0].distance_from_boundary_point == (-2, 0)

    assert one_distance_data[1].fixed_gridpoint == 4
    assert one_distance_data[1].range_dimension_bounds == [(3, 11)]
    assert one_distance_data[1].neighbor_velocity_pairs == ((4, 2), (12, 0))
    assert one_distance_data[1].distance_from_boundary_point == (-1, 1)

    assert one_distance_data[2].fixed_gridpoint == 4
    assert one_distance_data[2].range_dimension_bounds == [(1, 9)]
    assert one_distance_data[2].neighbor_velocity_pairs == ((2, 2), (6, 0))
    assert one_distance_data[2].distance_from_boundary_point == (-1, -1)

    assert one_distance_data[3].fixed_gridpoint == 6
    assert one_distance_data[3].range_dimension_bounds == [(2, 10)]
    assert one_distance_data[3].neighbor_velocity_pairs == ((3, 0), (9, 2))
    assert one_distance_data[3].distance_from_boundary_point == (2, 0)

    assert one_distance_data[4].fixed_gridpoint == 5
    assert one_distance_data[4].range_dimension_bounds == [(3, 11)]
    assert one_distance_data[4].neighbor_velocity_pairs == ((4, 0), (10, 2))
    assert one_distance_data[4].distance_from_boundary_point == (1, 1)

    assert one_distance_data[5].fixed_gridpoint == 5
    assert one_distance_data[5].range_dimension_bounds == [(1, 9)]
    assert one_distance_data[5].neighbor_velocity_pairs == ((2, 0), (8, 2))
    assert one_distance_data[5].distance_from_boundary_point == (1, -1)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d2_right_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    one_distance_data = [
        d
        for d in data[50:100]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 2)
    ]
    assert len(one_distance_data) == 6

    assert all(d.fixed_dim == 0 for d in one_distance_data)
    assert all(d.ranged_dims == [1] for d in one_distance_data)

    assert one_distance_data[0].fixed_gridpoint == 8
    assert one_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert one_distance_data[0].distance_from_boundary_point == (2, 0)
    assert one_distance_data[0].neighbor_velocity_pairs == ((3, 0), (9, 2))

    assert one_distance_data[1].fixed_gridpoint == 7
    assert one_distance_data[1].range_dimension_bounds == [(3, 11)]
    assert one_distance_data[1].distance_from_boundary_point == (1, 1)
    assert one_distance_data[1].neighbor_velocity_pairs == ((4, 0), (10, 2))

    assert one_distance_data[2].fixed_gridpoint == 7
    assert one_distance_data[2].range_dimension_bounds == [(1, 9)]
    assert one_distance_data[2].distance_from_boundary_point == (1, -1)
    assert one_distance_data[2].neighbor_velocity_pairs == ((2, 0), (8, 2))

    assert one_distance_data[3].fixed_gridpoint == 5
    assert one_distance_data[3].range_dimension_bounds == [(2, 10)]
    assert one_distance_data[3].distance_from_boundary_point == (-2, 0)
    assert one_distance_data[3].neighbor_velocity_pairs == ((1, 2), (5, 0))

    assert one_distance_data[4].fixed_gridpoint == 6
    assert one_distance_data[4].range_dimension_bounds == [(3, 11)]
    assert one_distance_data[4].distance_from_boundary_point == (-1, 1)
    assert one_distance_data[4].neighbor_velocity_pairs == ((4, 2), (12, 0))

    assert one_distance_data[5].fixed_gridpoint == 6
    assert one_distance_data[5].range_dimension_bounds == [(1, 9)]
    assert one_distance_data[5].distance_from_boundary_point == (-1, -1)
    assert one_distance_data[5].neighbor_velocity_pairs == ((2, 2), (6, 0))


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d2_bottom_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    one_distance_data = [
        d
        for d in data[100:150]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 2)
    ]
    assert len(one_distance_data) == 6

    assert all(d.fixed_dim == 1 for d in one_distance_data)
    assert all(d.ranged_dims == [0] for d in one_distance_data)

    assert one_distance_data[0].fixed_gridpoint == 0
    assert one_distance_data[0].range_dimension_bounds == [(5, 6)]
    assert one_distance_data[0].neighbor_velocity_pairs == ((2, 3), (7, 1))
    assert one_distance_data[0].distance_from_boundary_point == (0, -2)

    assert one_distance_data[1].fixed_gridpoint == 1
    assert one_distance_data[1].range_dimension_bounds == [(6, 7)]
    assert one_distance_data[1].neighbor_velocity_pairs == ((3, 3), (8, 1))
    assert one_distance_data[1].distance_from_boundary_point == (1, -1)

    assert one_distance_data[2].fixed_gridpoint == 1
    assert one_distance_data[2].range_dimension_bounds == [(4, 5)]
    assert one_distance_data[2].neighbor_velocity_pairs == ((1, 3), (6, 1))
    assert one_distance_data[2].distance_from_boundary_point == (-1, -1)

    assert one_distance_data[3].fixed_gridpoint == 3
    assert one_distance_data[3].range_dimension_bounds == [(5, 6)]
    assert one_distance_data[3].neighbor_velocity_pairs == ((4, 1), (11, 3))
    assert one_distance_data[3].distance_from_boundary_point == (0, 2)

    assert one_distance_data[4].fixed_gridpoint == 2
    assert one_distance_data[4].range_dimension_bounds == [(6, 7)]
    assert one_distance_data[4].neighbor_velocity_pairs == ((3, 1), (10, 3))
    assert one_distance_data[4].distance_from_boundary_point == (1, 1)

    assert one_distance_data[5].fixed_gridpoint == 2
    assert one_distance_data[5].range_dimension_bounds == [(4, 5)]
    assert one_distance_data[5].neighbor_velocity_pairs == ((1, 1), (12, 3))
    assert one_distance_data[5].distance_from_boundary_point == (-1, 1)

def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d2_top_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    one_distance_data = [
        d
        for d in data[150:]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 2)
    ]
    assert len(one_distance_data) == 6

    assert all(d.fixed_dim == 1 for d in one_distance_data)
    assert all(d.ranged_dims == [0] for d in one_distance_data)

    assert one_distance_data[0].fixed_gridpoint == 12
    assert one_distance_data[0].range_dimension_bounds == [(5, 6)]
    assert one_distance_data[0].neighbor_velocity_pairs == ((4, 1), (11, 3))
    assert one_distance_data[0].distance_from_boundary_point == (0, 2)

    assert one_distance_data[1].fixed_gridpoint == 11
    assert one_distance_data[1].range_dimension_bounds == [(6, 7)]
    assert one_distance_data[1].neighbor_velocity_pairs == ((3, 1), (10, 3))
    assert one_distance_data[1].distance_from_boundary_point == (1, 1)

    assert one_distance_data[2].fixed_gridpoint == 11
    assert one_distance_data[2].range_dimension_bounds == [(4, 5)]
    assert one_distance_data[2].neighbor_velocity_pairs == ((1, 1), (12, 3))
    assert one_distance_data[2].distance_from_boundary_point == (-1, 1)

    assert one_distance_data[3].fixed_gridpoint == 9
    assert one_distance_data[3].range_dimension_bounds == [(5, 6)]
    assert one_distance_data[3].neighbor_velocity_pairs == ((2, 3), (7, 1))
    assert one_distance_data[3].distance_from_boundary_point == (0, -2)

    assert one_distance_data[4].fixed_gridpoint == 10
    assert one_distance_data[4].range_dimension_bounds == [(6, 7)]
    assert one_distance_data[4].neighbor_velocity_pairs == ((3, 3), (8, 1))
    assert one_distance_data[4].distance_from_boundary_point == (1, -1)

    assert one_distance_data[5].fixed_gridpoint == 10
    assert one_distance_data[5].range_dimension_bounds == [(4, 5)]
    assert one_distance_data[5].neighbor_velocity_pairs == ((1, 3), (6, 1))
    assert one_distance_data[5].distance_from_boundary_point == (-1, -1)