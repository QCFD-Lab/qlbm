from typing import List

import pytest

from qlbm.lattice.geometry.encodings.spacetime import SpaceTimeVolumetricReflectionData
from qlbm.lattice.geometry.shapes.block import Block


@pytest.fixture
def simple_2d_block():
    return Block([(5, 6), (2, 10)], [4, 4], "bounceback")


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


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d3_left_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    three_distance_data = [
        d
        for d in data[:50]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 3)
    ]
    assert len(three_distance_data) == 10

    assert all(d.fixed_dim == 0 for d in three_distance_data)
    assert all(d.ranged_dims == [1] for d in three_distance_data)

    assert three_distance_data[0].fixed_gridpoint == 2
    assert three_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert three_distance_data[0].neighbor_velocity_pairs == ((5, 2), (13, 0))
    assert three_distance_data[0].distance_from_boundary_point == (-3, 0)

    assert three_distance_data[1].fixed_gridpoint == 3
    assert three_distance_data[1].range_dimension_bounds == [(3, 11)]
    assert three_distance_data[1].neighbor_velocity_pairs == ((12, 2), (24, 0))
    assert three_distance_data[1].distance_from_boundary_point == (-2, 1)

    assert three_distance_data[2].fixed_gridpoint == 4
    assert three_distance_data[2].range_dimension_bounds == [(4, 12)]
    assert three_distance_data[2].neighbor_velocity_pairs == ((11, 2), (23, 0))
    assert three_distance_data[2].distance_from_boundary_point == (-1, 2)

    assert three_distance_data[3].fixed_gridpoint == 3
    assert three_distance_data[3].range_dimension_bounds == [(1, 9)]
    assert three_distance_data[3].neighbor_velocity_pairs == ((6, 2), (14, 0))
    assert three_distance_data[3].distance_from_boundary_point == (-2, -1)

    assert three_distance_data[4].fixed_gridpoint == 4
    assert three_distance_data[4].range_dimension_bounds == [(0, 8)]
    assert three_distance_data[4].neighbor_velocity_pairs == ((7, 2), (15, 0))
    assert three_distance_data[4].distance_from_boundary_point == (-1, -2)

    assert three_distance_data[5].fixed_gridpoint == 7
    assert three_distance_data[5].range_dimension_bounds == [(2, 10)]
    assert three_distance_data[5].neighbor_velocity_pairs == ((9, 0), (19, 2))
    assert three_distance_data[5].distance_from_boundary_point == (3, 0)

    assert three_distance_data[6].fixed_gridpoint == 6
    assert three_distance_data[6].range_dimension_bounds == [(3, 11)]
    assert three_distance_data[6].neighbor_velocity_pairs == ((10, 0), (20, 2))
    assert three_distance_data[6].distance_from_boundary_point == (2, 1)

    assert three_distance_data[7].fixed_gridpoint == 5
    assert three_distance_data[7].range_dimension_bounds == [(4, 12)]
    assert three_distance_data[7].neighbor_velocity_pairs == ((11, 0), (21, 2))
    assert three_distance_data[7].distance_from_boundary_point == (1, 2)

    assert three_distance_data[8].fixed_gridpoint == 6
    assert three_distance_data[8].range_dimension_bounds == [(1, 9)]
    assert three_distance_data[8].neighbor_velocity_pairs == ((8, 0), (18, 2))
    assert three_distance_data[8].distance_from_boundary_point == (2, -1)

    assert three_distance_data[9].fixed_gridpoint == 5
    assert three_distance_data[9].range_dimension_bounds == [(0, 8)]
    assert three_distance_data[9].neighbor_velocity_pairs == ((7, 0), (17, 2))
    assert three_distance_data[9].distance_from_boundary_point == (1, -2)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d3_right_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    three_distance_data = [
        d
        for d in data[:50]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 3)
    ]
    assert len(three_distance_data) == 10

    assert all(d.fixed_dim == 0 for d in three_distance_data)
    assert all(d.ranged_dims == [1] for d in three_distance_data)

    assert three_distance_data[0].fixed_gridpoint == 2
    assert three_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert three_distance_data[0].neighbor_velocity_pairs == ((5, 2), (13, 0))
    assert three_distance_data[0].distance_from_boundary_point == (-3, 0)

    assert three_distance_data[1].fixed_gridpoint == 3
    assert three_distance_data[1].range_dimension_bounds == [(3, 11)]
    assert three_distance_data[1].neighbor_velocity_pairs == ((12, 2), (24, 0))
    assert three_distance_data[1].distance_from_boundary_point == (-2, 1)

    assert three_distance_data[2].fixed_gridpoint == 4
    assert three_distance_data[2].range_dimension_bounds == [(4, 12)]
    assert three_distance_data[2].neighbor_velocity_pairs == ((11, 2), (23, 0))
    assert three_distance_data[2].distance_from_boundary_point == (-1, 2)

    assert three_distance_data[3].fixed_gridpoint == 3
    assert three_distance_data[3].range_dimension_bounds == [(1, 9)]
    assert three_distance_data[3].neighbor_velocity_pairs == ((6, 2), (14, 0))
    assert three_distance_data[3].distance_from_boundary_point == (-2, -1)

    assert three_distance_data[4].fixed_gridpoint == 4
    assert three_distance_data[4].range_dimension_bounds == [(0, 8)]
    assert three_distance_data[4].neighbor_velocity_pairs == ((7, 2), (15, 0))
    assert three_distance_data[4].distance_from_boundary_point == (-1, -2)

    assert three_distance_data[5].fixed_gridpoint == 7
    assert three_distance_data[5].range_dimension_bounds == [(2, 10)]
    assert three_distance_data[5].neighbor_velocity_pairs == ((9, 0), (19, 2))
    assert three_distance_data[5].distance_from_boundary_point == (3, 0)

    assert three_distance_data[6].fixed_gridpoint == 6
    assert three_distance_data[6].range_dimension_bounds == [(3, 11)]
    assert three_distance_data[6].neighbor_velocity_pairs == ((10, 0), (20, 2))
    assert three_distance_data[6].distance_from_boundary_point == (2, 1)

    assert three_distance_data[7].fixed_gridpoint == 5
    assert three_distance_data[7].range_dimension_bounds == [(4, 12)]
    assert three_distance_data[7].neighbor_velocity_pairs == ((11, 0), (21, 2))
    assert three_distance_data[7].distance_from_boundary_point == (1, 2)

    assert three_distance_data[8].fixed_gridpoint == 6
    assert three_distance_data[8].range_dimension_bounds == [(1, 9)]
    assert three_distance_data[8].neighbor_velocity_pairs == ((8, 0), (18, 2))
    assert three_distance_data[8].distance_from_boundary_point == (2, -1)

    assert three_distance_data[9].fixed_gridpoint == 5
    assert three_distance_data[9].range_dimension_bounds == [(0, 8)]
    assert three_distance_data[9].neighbor_velocity_pairs == ((7, 0), (17, 2))
    assert three_distance_data[9].distance_from_boundary_point == (1, -2)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d3_bottom_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    three_distance_data = [
        d
        for d in data[:50]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 3)
    ]
    assert len(three_distance_data) == 10

    assert all(d.fixed_dim == 0 for d in three_distance_data)
    assert all(d.ranged_dims == [1] for d in three_distance_data)

    assert three_distance_data[0].fixed_gridpoint == 2
    assert three_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert three_distance_data[0].neighbor_velocity_pairs == ((5, 2), (13, 0))
    assert three_distance_data[0].distance_from_boundary_point == (-3, 0)

    assert three_distance_data[1].fixed_gridpoint == 3
    assert three_distance_data[1].range_dimension_bounds == [(3, 11)]
    assert three_distance_data[1].neighbor_velocity_pairs == ((12, 2), (24, 0))
    assert three_distance_data[1].distance_from_boundary_point == (-2, 1)

    assert three_distance_data[2].fixed_gridpoint == 4
    assert three_distance_data[2].range_dimension_bounds == [(4, 12)]
    assert three_distance_data[2].neighbor_velocity_pairs == ((11, 2), (23, 0))
    assert three_distance_data[2].distance_from_boundary_point == (-1, 2)

    assert three_distance_data[3].fixed_gridpoint == 3
    assert three_distance_data[3].range_dimension_bounds == [(1, 9)]
    assert three_distance_data[3].neighbor_velocity_pairs == ((6, 2), (14, 0))
    assert three_distance_data[3].distance_from_boundary_point == (-2, -1)

    assert three_distance_data[4].fixed_gridpoint == 4
    assert three_distance_data[4].range_dimension_bounds == [(0, 8)]
    assert three_distance_data[4].neighbor_velocity_pairs == ((7, 2), (15, 0))
    assert three_distance_data[4].distance_from_boundary_point == (-1, -2)

    assert three_distance_data[5].fixed_gridpoint == 7
    assert three_distance_data[5].range_dimension_bounds == [(2, 10)]
    assert three_distance_data[5].neighbor_velocity_pairs == ((9, 0), (19, 2))
    assert three_distance_data[5].distance_from_boundary_point == (3, 0)

    assert three_distance_data[6].fixed_gridpoint == 6
    assert three_distance_data[6].range_dimension_bounds == [(3, 11)]
    assert three_distance_data[6].neighbor_velocity_pairs == ((10, 0), (20, 2))
    assert three_distance_data[6].distance_from_boundary_point == (2, 1)

    assert three_distance_data[7].fixed_gridpoint == 5
    assert three_distance_data[7].range_dimension_bounds == [(4, 12)]
    assert three_distance_data[7].neighbor_velocity_pairs == ((11, 0), (21, 2))
    assert three_distance_data[7].distance_from_boundary_point == (1, 2)

    assert three_distance_data[8].fixed_gridpoint == 6
    assert three_distance_data[8].range_dimension_bounds == [(1, 9)]
    assert three_distance_data[8].neighbor_velocity_pairs == ((8, 0), (18, 2))
    assert three_distance_data[8].distance_from_boundary_point == (2, -1)

    assert three_distance_data[9].fixed_gridpoint == 5
    assert three_distance_data[9].range_dimension_bounds == [(0, 8)]
    assert three_distance_data[9].neighbor_velocity_pairs == ((7, 0), (17, 2))
    assert three_distance_data[9].distance_from_boundary_point == (1, -2)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d3_top_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    three_distance_data = [
        d
        for d in data[:50]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 3)
    ]
    assert len(three_distance_data) == 10

    assert all(d.fixed_dim == 0 for d in three_distance_data)
    assert all(d.ranged_dims == [1] for d in three_distance_data)

    assert three_distance_data[0].fixed_gridpoint == 2
    assert three_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert three_distance_data[0].neighbor_velocity_pairs == ((5, 2), (13, 0))
    assert three_distance_data[0].distance_from_boundary_point == (-3, 0)

    assert three_distance_data[1].fixed_gridpoint == 3
    assert three_distance_data[1].range_dimension_bounds == [(3, 11)]
    assert three_distance_data[1].neighbor_velocity_pairs == ((12, 2), (24, 0))
    assert three_distance_data[1].distance_from_boundary_point == (-2, 1)

    assert three_distance_data[2].fixed_gridpoint == 4
    assert three_distance_data[2].range_dimension_bounds == [(4, 12)]
    assert three_distance_data[2].neighbor_velocity_pairs == ((11, 2), (23, 0))
    assert three_distance_data[2].distance_from_boundary_point == (-1, 2)

    assert three_distance_data[3].fixed_gridpoint == 3
    assert three_distance_data[3].range_dimension_bounds == [(1, 9)]
    assert three_distance_data[3].neighbor_velocity_pairs == ((6, 2), (14, 0))
    assert three_distance_data[3].distance_from_boundary_point == (-2, -1)

    assert three_distance_data[4].fixed_gridpoint == 4
    assert three_distance_data[4].range_dimension_bounds == [(0, 8)]
    assert three_distance_data[4].neighbor_velocity_pairs == ((7, 2), (15, 0))
    assert three_distance_data[4].distance_from_boundary_point == (-1, -2)

    assert three_distance_data[5].fixed_gridpoint == 7
    assert three_distance_data[5].range_dimension_bounds == [(2, 10)]
    assert three_distance_data[5].neighbor_velocity_pairs == ((9, 0), (19, 2))
    assert three_distance_data[5].distance_from_boundary_point == (3, 0)

    assert three_distance_data[6].fixed_gridpoint == 6
    assert three_distance_data[6].range_dimension_bounds == [(3, 11)]
    assert three_distance_data[6].neighbor_velocity_pairs == ((10, 0), (20, 2))
    assert three_distance_data[6].distance_from_boundary_point == (2, 1)

    assert three_distance_data[7].fixed_gridpoint == 5
    assert three_distance_data[7].range_dimension_bounds == [(4, 12)]
    assert three_distance_data[7].neighbor_velocity_pairs == ((11, 0), (21, 2))
    assert three_distance_data[7].distance_from_boundary_point == (1, 2)

    assert three_distance_data[8].fixed_gridpoint == 6
    assert three_distance_data[8].range_dimension_bounds == [(1, 9)]
    assert three_distance_data[8].neighbor_velocity_pairs == ((8, 0), (18, 2))
    assert three_distance_data[8].distance_from_boundary_point == (2, -1)

    assert three_distance_data[9].fixed_gridpoint == 5
    assert three_distance_data[9].range_dimension_bounds == [(0, 8)]
    assert three_distance_data[9].neighbor_velocity_pairs == ((7, 0), (17, 2))
    assert three_distance_data[9].distance_from_boundary_point == (1, -2)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d4_left_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    four_distance_data = [
        d
        for d in data[:50]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 4)
    ]
    assert len(four_distance_data) == 14

    assert four_distance_data[0].fixed_gridpoint == 1
    assert four_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert four_distance_data[0].neighbor_velocity_pairs == ((13, 2), (25, 0))
    assert four_distance_data[0].distance_from_boundary_point == (-4, 0)

    assert four_distance_data[1].fixed_gridpoint == 2
    assert four_distance_data[1].range_dimension_bounds == [(3, 11)]
    assert four_distance_data[1].neighbor_velocity_pairs == ((24, 2), (40, 0))
    assert four_distance_data[1].distance_from_boundary_point == (-3, 1)

    assert four_distance_data[2].fixed_gridpoint == 3
    assert four_distance_data[2].range_dimension_bounds == [(4, 12)]
    assert four_distance_data[2].neighbor_velocity_pairs == ((23, 2), (39, 0))
    assert four_distance_data[2].distance_from_boundary_point == (-2, 2)

    assert four_distance_data[3].fixed_gridpoint == 4
    assert four_distance_data[3].range_dimension_bounds == [(5, 13)]
    assert four_distance_data[3].neighbor_velocity_pairs == ((22, 2), (38, 0))
    assert four_distance_data[3].distance_from_boundary_point == (-1, 3)

    assert four_distance_data[4].fixed_gridpoint == 2
    assert four_distance_data[4].range_dimension_bounds == [(1, 9)]
    assert four_distance_data[4].neighbor_velocity_pairs == ((14, 2), (26, 0))
    assert four_distance_data[4].distance_from_boundary_point == (-3, -1)

    assert four_distance_data[5].fixed_gridpoint == 3
    assert four_distance_data[5].range_dimension_bounds == [(0, 8)]
    assert four_distance_data[5].neighbor_velocity_pairs == ((15, 2), (27, 0))
    assert four_distance_data[5].distance_from_boundary_point == (-2, -2)

    assert four_distance_data[6].fixed_gridpoint == 4
    assert four_distance_data[6].range_dimension_bounds == [(-1, 7)]
    assert four_distance_data[6].neighbor_velocity_pairs == ((16, 2), (28, 0))
    assert four_distance_data[6].distance_from_boundary_point == (-1, -3)

    assert four_distance_data[7].fixed_gridpoint == 8
    assert four_distance_data[7].range_dimension_bounds == [(2, 10)]
    assert four_distance_data[7].neighbor_velocity_pairs == ((19, 0), (33, 2))
    assert four_distance_data[7].distance_from_boundary_point == (4, 0)

    assert four_distance_data[8].fixed_gridpoint == 7
    assert four_distance_data[8].range_dimension_bounds == [(3, 11)]
    assert four_distance_data[8].neighbor_velocity_pairs == ((20, 0), (34, 2))
    assert four_distance_data[8].distance_from_boundary_point == (3, 1)

    assert four_distance_data[9].fixed_gridpoint == 6
    assert four_distance_data[9].range_dimension_bounds == [(4, 12)]
    assert four_distance_data[9].neighbor_velocity_pairs == ((21, 0), (35, 2))
    assert four_distance_data[9].distance_from_boundary_point == (2, 2)

    assert four_distance_data[10].fixed_gridpoint == 5
    assert four_distance_data[10].range_dimension_bounds == [(5, 13)]
    assert four_distance_data[10].neighbor_velocity_pairs == ((22, 0), (36, 2))
    assert four_distance_data[10].distance_from_boundary_point == (1, 3)

    assert four_distance_data[11].fixed_gridpoint == 7
    assert four_distance_data[11].range_dimension_bounds == [(1, 9)]
    assert four_distance_data[11].neighbor_velocity_pairs == ((18, 0), (32, 2))
    assert four_distance_data[11].distance_from_boundary_point == (3, -1)

    assert four_distance_data[12].fixed_gridpoint == 6
    assert four_distance_data[12].range_dimension_bounds == [(0, 8)]
    assert four_distance_data[12].neighbor_velocity_pairs == ((17, 0), (31, 2))
    assert four_distance_data[12].distance_from_boundary_point == (2, -2)

    assert four_distance_data[13].fixed_gridpoint == 5
    assert four_distance_data[13].range_dimension_bounds == [(-1, 7)]
    assert four_distance_data[13].neighbor_velocity_pairs == ((16, 0), (30, 2))
    assert four_distance_data[13].distance_from_boundary_point == (1, -3)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d4_right_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    four_distance_data = [
        d
        for d in data[50:100]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 4)
    ]
    assert len(four_distance_data) == 14

    assert four_distance_data[0].fixed_gridpoint == 10
    assert four_distance_data[0].range_dimension_bounds == [(2, 10)]
    assert four_distance_data[0].neighbor_velocity_pairs == ((19, 0), (33, 2))
    assert four_distance_data[0].distance_from_boundary_point == (4, 0)

    assert four_distance_data[1].fixed_gridpoint == 9
    assert four_distance_data[1].range_dimension_bounds == [(3, 11)]
    assert four_distance_data[1].neighbor_velocity_pairs == ((20, 0), (34, 2))
    assert four_distance_data[1].distance_from_boundary_point == (3, 1)

    assert four_distance_data[2].fixed_gridpoint == 8
    assert four_distance_data[2].range_dimension_bounds == [(4, 12)]
    assert four_distance_data[2].neighbor_velocity_pairs == ((21, 0), (35, 2))
    assert four_distance_data[2].distance_from_boundary_point == (2, 2)

    assert four_distance_data[3].fixed_gridpoint == 7
    assert four_distance_data[3].range_dimension_bounds == [(5, 13)]
    assert four_distance_data[3].neighbor_velocity_pairs == ((22, 0), (36, 2))
    assert four_distance_data[3].distance_from_boundary_point == (1, 3)

    assert four_distance_data[4].fixed_gridpoint == 9
    assert four_distance_data[4].range_dimension_bounds == [(1, 9)]
    assert four_distance_data[4].neighbor_velocity_pairs == ((18, 0), (32, 2))
    assert four_distance_data[4].distance_from_boundary_point == (3, -1)

    assert four_distance_data[5].fixed_gridpoint == 8
    assert four_distance_data[5].range_dimension_bounds == [(0, 8)]
    assert four_distance_data[5].neighbor_velocity_pairs == ((17, 0), (31, 2))
    assert four_distance_data[5].distance_from_boundary_point == (2, -2)

    assert four_distance_data[6].fixed_gridpoint == 7
    assert four_distance_data[6].range_dimension_bounds == [(-1, 7)]
    assert four_distance_data[6].neighbor_velocity_pairs == ((16, 0), (30, 2))
    assert four_distance_data[6].distance_from_boundary_point == (1, -3)

    assert four_distance_data[7].fixed_gridpoint == 3
    assert four_distance_data[7].range_dimension_bounds == [(2, 10)]
    assert four_distance_data[7].neighbor_velocity_pairs == ((13, 2), (25, 0))
    assert four_distance_data[7].distance_from_boundary_point == (-4, 0)

    assert four_distance_data[8].fixed_gridpoint == 4
    assert four_distance_data[8].range_dimension_bounds == [(3, 11)]
    assert four_distance_data[8].neighbor_velocity_pairs == ((24, 2), (40, 0))
    assert four_distance_data[8].distance_from_boundary_point == (-3, 1)

    assert four_distance_data[9].fixed_gridpoint == 5
    assert four_distance_data[9].range_dimension_bounds == [(4, 12)]
    assert four_distance_data[9].neighbor_velocity_pairs == ((23, 2), (39, 0))
    assert four_distance_data[9].distance_from_boundary_point == (-2, 2)

    assert four_distance_data[10].fixed_gridpoint == 6
    assert four_distance_data[10].range_dimension_bounds == [(5, 13)]
    assert four_distance_data[10].neighbor_velocity_pairs == ((22, 2), (38, 0))
    assert four_distance_data[10].distance_from_boundary_point == (-1, 3)

    assert four_distance_data[11].fixed_gridpoint == 4
    assert four_distance_data[11].range_dimension_bounds == [(1, 9)]
    assert four_distance_data[11].neighbor_velocity_pairs == ((14, 2), (26, 0))
    assert four_distance_data[11].distance_from_boundary_point == (-3, -1)

    assert four_distance_data[12].fixed_gridpoint == 5
    assert four_distance_data[12].range_dimension_bounds == [(0, 8)]
    assert four_distance_data[12].neighbor_velocity_pairs == ((15, 2), (27, 0))
    assert four_distance_data[12].distance_from_boundary_point == (-2, -2)

    assert four_distance_data[13].fixed_gridpoint == 6
    assert four_distance_data[13].range_dimension_bounds == [(-1, 7)]
    assert four_distance_data[13].neighbor_velocity_pairs == ((16, 2), (28, 0))
    assert four_distance_data[13].distance_from_boundary_point == (-1, -3)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d4_bottom_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    four_distance_data = [
        d
        for d in data[100:150]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 4)
    ]
    assert len(four_distance_data) == 14

    assert four_distance_data[0].fixed_gridpoint == 14
    assert four_distance_data[0].range_dimension_bounds == [(5, 6)]
    assert four_distance_data[0].neighbor_velocity_pairs == ((16, 3), (29, 1))
    assert four_distance_data[0].distance_from_boundary_point == (0, -4)

    assert four_distance_data[1].fixed_gridpoint == 15
    assert four_distance_data[1].range_dimension_bounds == [(6, 7)]
    assert four_distance_data[1].neighbor_velocity_pairs == ((17, 3), (30, 1))
    assert four_distance_data[1].distance_from_boundary_point == (1, -3)

    assert four_distance_data[2].fixed_gridpoint == 0
    assert four_distance_data[2].range_dimension_bounds == [(7, 8)]
    assert four_distance_data[2].neighbor_velocity_pairs == ((18, 3), (31, 1))
    assert four_distance_data[2].distance_from_boundary_point == (2, -2)

    assert four_distance_data[3].fixed_gridpoint == 1
    assert four_distance_data[3].range_dimension_bounds == [(8, 9)]
    assert four_distance_data[3].neighbor_velocity_pairs == ((19, 3), (32, 1))
    assert four_distance_data[3].distance_from_boundary_point == (3, -1)

    assert four_distance_data[4].fixed_gridpoint == 15
    assert four_distance_data[4].range_dimension_bounds == [(4, 5)]
    assert four_distance_data[4].neighbor_velocity_pairs == ((15, 3), (28, 1))
    assert four_distance_data[4].distance_from_boundary_point == (-1, -3)

    assert four_distance_data[5].fixed_gridpoint == 0
    assert four_distance_data[5].range_dimension_bounds == [(3, 4)]
    assert four_distance_data[5].neighbor_velocity_pairs == ((14, 3), (27, 1))
    assert four_distance_data[5].distance_from_boundary_point == (-2, -2)

    assert four_distance_data[6].fixed_gridpoint == 1
    assert four_distance_data[6].range_dimension_bounds == [(2, 3)]
    assert four_distance_data[6].neighbor_velocity_pairs == ((13, 3), (26, 1))
    assert four_distance_data[6].distance_from_boundary_point == (-3, -1)

    assert four_distance_data[7].fixed_gridpoint == 5
    assert four_distance_data[7].range_dimension_bounds == [(5, 6)]
    assert four_distance_data[7].neighbor_velocity_pairs == ((22, 1), (37, 3))
    assert four_distance_data[7].distance_from_boundary_point == (0, 4)

    assert four_distance_data[8].fixed_gridpoint == 4
    assert four_distance_data[8].range_dimension_bounds == [(6, 7)]
    assert four_distance_data[8].neighbor_velocity_pairs == ((21, 1), (36, 3))
    assert four_distance_data[8].distance_from_boundary_point == (1, 3)

    assert four_distance_data[9].fixed_gridpoint == 3
    assert four_distance_data[9].range_dimension_bounds == [(7, 8)]
    assert four_distance_data[9].neighbor_velocity_pairs == ((20, 1), (35, 3))
    assert four_distance_data[9].distance_from_boundary_point == (2, 2)

    assert four_distance_data[10].fixed_gridpoint == 2
    assert four_distance_data[10].range_dimension_bounds == [(8, 9)]
    assert four_distance_data[10].neighbor_velocity_pairs == ((19, 1), (34, 3))
    assert four_distance_data[10].distance_from_boundary_point == (3, 1)

    assert four_distance_data[11].fixed_gridpoint == 4
    assert four_distance_data[11].range_dimension_bounds == [(4, 5)]
    assert four_distance_data[11].neighbor_velocity_pairs == ((23, 1), (38, 3))
    assert four_distance_data[11].distance_from_boundary_point == (-1, 3)

    assert four_distance_data[12].fixed_gridpoint == 3
    assert four_distance_data[12].range_dimension_bounds == [(3, 4)]
    assert four_distance_data[12].neighbor_velocity_pairs == ((24, 1), (39, 3))
    assert four_distance_data[12].distance_from_boundary_point == (-2, 2)

    assert four_distance_data[13].fixed_gridpoint == 2
    assert four_distance_data[13].range_dimension_bounds == [(2, 3)]
    assert four_distance_data[13].neighbor_velocity_pairs == ((13, 1), (40, 3))
    assert four_distance_data[13].distance_from_boundary_point == (-3, 1)


def test_get_spacetime_reflection_data_2_timesteps_neighbor_velocity_pairs_d4_top_wall(
    simple_2d_block, lattice_2d_16x16_1_obstacle_5_timesteps
):
    data: List[SpaceTimeVolumetricReflectionData] = (
        simple_2d_block.get_d2q4_volumetric_reflection_data(
            lattice_2d_16x16_1_obstacle_5_timesteps.properties
        )
    )

    assert len(data) == 50 * 4  # 4 walls, 50 gridpoints relevant per wall (2t^2)
    four_distance_data = [
        d
        for d in data[150:]
        if (sum(abs(x) for x in d.distance_from_boundary_point) == 4)
    ]
    assert len(four_distance_data) == 14

    assert four_distance_data[0].fixed_gridpoint == 14
    assert four_distance_data[0].range_dimension_bounds == [(5, 6)]
    assert four_distance_data[0].neighbor_velocity_pairs == ((22, 1), (37, 3))
    assert four_distance_data[0].distance_from_boundary_point == (0, 4)

    assert four_distance_data[1].fixed_gridpoint == 13
    assert four_distance_data[1].range_dimension_bounds == [(6, 7)]
    assert four_distance_data[1].neighbor_velocity_pairs == ((21, 1), (36, 3))
    assert four_distance_data[1].distance_from_boundary_point == (1, 3)

    assert four_distance_data[2].fixed_gridpoint == 12
    assert four_distance_data[2].range_dimension_bounds == [(7, 8)]
    assert four_distance_data[2].neighbor_velocity_pairs == ((20, 1), (35, 3))
    assert four_distance_data[2].distance_from_boundary_point == (2, 2)

    assert four_distance_data[3].fixed_gridpoint == 11
    assert four_distance_data[3].range_dimension_bounds == [(8, 9)]
    assert four_distance_data[3].neighbor_velocity_pairs == ((19, 1), (34, 3))
    assert four_distance_data[3].distance_from_boundary_point == (3, 1)

    assert four_distance_data[4].fixed_gridpoint == 13
    assert four_distance_data[4].range_dimension_bounds == [(4, 5)]
    assert four_distance_data[4].neighbor_velocity_pairs == ((23, 1), (38, 3))
    assert four_distance_data[4].distance_from_boundary_point == (-1, 3)

    assert four_distance_data[5].fixed_gridpoint == 12
    assert four_distance_data[5].range_dimension_bounds == [(3, 4)]
    assert four_distance_data[5].neighbor_velocity_pairs == ((24, 1), (39, 3))
    assert four_distance_data[5].distance_from_boundary_point == (-2, 2)

    assert four_distance_data[6].fixed_gridpoint == 11
    assert four_distance_data[6].range_dimension_bounds == [(2, 3)]
    assert four_distance_data[6].neighbor_velocity_pairs == ((13, 1), (40, 3))
    assert four_distance_data[6].distance_from_boundary_point == (-3, 1)

    assert four_distance_data[7].fixed_gridpoint == 7
    assert four_distance_data[7].range_dimension_bounds == [(5, 6)]
    assert four_distance_data[7].neighbor_velocity_pairs == ((16, 3), (29, 1))
    assert four_distance_data[7].distance_from_boundary_point == (0, -4)

    assert four_distance_data[8].fixed_gridpoint == 8
    assert four_distance_data[8].range_dimension_bounds == [(6, 7)]
    assert four_distance_data[8].neighbor_velocity_pairs == ((17, 3), (30, 1))
    assert four_distance_data[8].distance_from_boundary_point == (1, -3)

    assert four_distance_data[9].fixed_gridpoint == 9
    assert four_distance_data[9].range_dimension_bounds == [(7, 8)]
    assert four_distance_data[9].neighbor_velocity_pairs == ((18, 3), (31, 1))
    assert four_distance_data[9].distance_from_boundary_point == (2, -2)

    assert four_distance_data[10].fixed_gridpoint == 10
    assert four_distance_data[10].range_dimension_bounds == [(8, 9)]
    assert four_distance_data[10].neighbor_velocity_pairs == ((19, 3), (32, 1))
    assert four_distance_data[10].distance_from_boundary_point == (3, -1)

    assert four_distance_data[11].fixed_gridpoint == 8
    assert four_distance_data[11].range_dimension_bounds == [(4, 5)]
    assert four_distance_data[11].neighbor_velocity_pairs == ((15, 3), (28, 1))
    assert four_distance_data[11].distance_from_boundary_point == (-1, -3)

    assert four_distance_data[12].fixed_gridpoint == 9
    assert four_distance_data[12].range_dimension_bounds == [(3, 4)]
    assert four_distance_data[12].neighbor_velocity_pairs == ((14, 3), (27, 1))
    assert four_distance_data[12].distance_from_boundary_point == (-2, -2)

    assert four_distance_data[13].fixed_gridpoint == 10
    assert four_distance_data[13].range_dimension_bounds == [(2, 3)]
    assert four_distance_data[13].neighbor_velocity_pairs == ((13, 3), (26, 1))
    assert four_distance_data[13].distance_from_boundary_point == (-3, -1)
