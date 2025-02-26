import pytest

from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.utils import flatten


def test_get_spacetime_reflection_data_constructor_1_timestep(
    simple_1d_block, lattice_1d_16_1_obstacle_1_timestep
):
    stqbm_reflection_data = simple_1d_block.get_spacetime_reflection_data_d1q2(
        lattice_1d_16_1_obstacle_1_timestep.properties, 1
    )

    assert len(stqbm_reflection_data) == 4

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]
    assert gridpoints_encoded == [(4, 0), (5, 0), (12, 0), (11, 0)]

    qubits_to_invert = [rd.qubits_to_invert for rd in stqbm_reflection_data]
    assert qubits_to_invert == [[0, 1, 3], [1, 3], [0, 1], [2]]

    velocities_indices_to_invert = [
        rd.velocity_indices_to_reflect for rd in stqbm_reflection_data
    ]
    assert flatten(velocities_indices_to_invert) == [0, 1, 1, 0]

    distances_from_boundary_points = [
        rd.distance_from_boundary_point for rd in stqbm_reflection_data
    ]
    assert distances_from_boundary_points == [(-1, 0), (1, 0), (1, 0), (-1, 0)]


def test_get_spacetime_reflection_data_neighbor_velocity_pairs_1_timestep(
    simple_1d_block, lattice_1d_16_1_obstacle_1_timestep
):
    stqbm_reflection_data = simple_1d_block.get_spacetime_reflection_data_d1q2(
        lattice_1d_16_1_obstacle_1_timestep.properties, 1
    )

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]
    assert flatten(neighbor_velocity_pairs) == [
        ((0, 1), (1, 0)),
        ((0, 0), (2, 1)),
        ((0, 0), (2, 1)),
        ((0, 1), (1, 0)),
    ]


def test_get_spacetime_reflection_data_constructor_2_timesteps(
    simple_1d_block, lattice_1d_16_1_obstacle_2_timesteps
):
    stqbm_reflection_data = simple_1d_block.get_spacetime_reflection_data_d1q2(
        lattice_1d_16_1_obstacle_2_timesteps.properties
    )

    assert len(stqbm_reflection_data) == 8

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]
    assert gridpoints_encoded == [
        (4, 0),
        (3, 0),
        (5, 0),
        (6, 0),
        (12, 0),
        (13, 0),
        (11, 0),
        (10, 0),
    ]

    qubits_to_invert = [rd.qubits_to_invert for rd in stqbm_reflection_data]
    assert qubits_to_invert == [
        [0, 1, 3],  # 4
        [2, 3],  # 3
        [1, 3],  # 5
        [0, 3],  # 6
        [0, 1],  # 12
        [1],  # 13
        [2],  # 11
        [0, 2],  # 10
    ]

    velocities_indices_to_invert = [
        rd.velocity_indices_to_reflect for rd in stqbm_reflection_data
    ]
    assert flatten(velocities_indices_to_invert) == [0] * 2 + [1] * 4 + [0] * 2

    distances_from_boundary_points = [
        rd.distance_from_boundary_point for rd in stqbm_reflection_data
    ]
    assert distances_from_boundary_points == [
        (-1, 0),
        (-2, 0),
        (1, 0),
        (2, 0),
        (1, 0),
        (2, 0),
        (-1, 0),
        (-2, 0),
    ]


def test_get_spacetime_reflection_data_neighbor_velocity_pairs_2_timesteps(
    simple_1d_block, lattice_1d_16_1_obstacle_2_timesteps
):
    stqbm_reflection_data = simple_1d_block.get_spacetime_reflection_data_d1q2(
        lattice_1d_16_1_obstacle_2_timesteps.properties
    )

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]
    assert flatten(neighbor_velocity_pairs) == [
        ((0, 1), (1, 0)),  # 4
        ((1, 1), (3, 0)),  # 3
        ((0, 0), (2, 1)),  # 5
        ((2, 0), (4, 1)),  # 6
        ((0, 0), (2, 1)),  # 12
        ((2, 0), (4, 1)),  # 13
        ((0, 1), (1, 0)),  # 11
        ((1, 1), (3, 0)),  # 10
    ]


def test_get_spacetime_reflection_data_constructor_4_timesteps(
    simple_1d_block, lattice_1d_16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_1d_block.get_spacetime_reflection_data_d1q2(
        lattice_1d_16_1_obstacle_5_timesteps.properties, 4
    )

    assert len(stqbm_reflection_data) == 16

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]
    assert gridpoints_encoded == [
        (4, 0),
        (3, 0),
        (2, 0),
        (1, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),
        (12, 0),
        (13, 0),
        (14, 0),
        (15, 0),
        (11, 0),
        (10, 0),
        (9, 0),
        (8, 0),
    ]

    qubits_to_invert = [rd.qubits_to_invert for rd in stqbm_reflection_data]
    assert qubits_to_invert == [
        [0, 1, 3],  # 4
        [2, 3],  # 3
        [0, 2, 3],  # 2
        [1, 2, 3],  # 1
        [1, 3],  # 5
        [0, 3],  # 6
        [3],  # 7
        [0, 1, 2],  # 8
        [0, 1],  # 12
        [1],  # 13
        [0],  # 14
        [],  # 15
        [2],  # 11
        [0, 2],  # 10
        [1, 2],  # 9
        [0, 1, 2],  # 8
    ]

    velocities_indices_to_invert = [
        rd.velocity_indices_to_reflect for rd in stqbm_reflection_data
    ]
    assert flatten(velocities_indices_to_invert) == [0] * 4 + [1] * 8 + [0] * 4

    distances_from_boundary_points = [
        rd.distance_from_boundary_point for rd in stqbm_reflection_data
    ]
    assert distances_from_boundary_points == [
        (-1, 0),
        (-2, 0),
        (-3, 0),
        (-4, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (-1, 0),
        (-2, 0),
        (-3, 0),
        (-4, 0),
    ]


def test_get_spacetime_reflection_data_neighbor_velocity_pairs_4_timesteps(
    simple_1d_block, lattice_1d_16_1_obstacle_5_timesteps
):
    stqbm_reflection_data = simple_1d_block.get_spacetime_reflection_data_d1q2(
        lattice_1d_16_1_obstacle_5_timesteps.properties, 4
    )

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]
    assert flatten(neighbor_velocity_pairs) == [
        ((0, 1), (1, 0)),  # 4
        ((1, 1), (3, 0)),  # 3
        ((3, 1), (5, 0)),  # 2
        ((5, 1), (7, 0)),  # 1
        ((0, 0), (2, 1)),  # 5
        ((2, 0), (4, 1)),  # 6
        ((4, 0), (6, 1)),  # 7
        ((6, 0), (8, 1)),  # 8
        ((0, 0), (2, 1)),  # 12
        ((2, 0), (4, 1)),  # 13
        ((4, 0), (6, 1)),  # 14
        ((6, 0), (8, 1)),  # 15
        ((0, 1), (1, 0)),  # 11
        ((1, 1), (3, 0)),  # 10
        ((3, 1), (5, 0)),  # 9
        ((5, 1), (7, 0)),  # 8
    ]


def test_get_spacetime_reflection_data_constructor_4_timesteps_outside_bounds(
    simple_large_1d_block, lattice_1d_16_1_obstacle_5_timesteps_large_obstacle
):
    stqbm_reflection_data = simple_large_1d_block.get_spacetime_reflection_data_d1q2(
        lattice_1d_16_1_obstacle_5_timesteps_large_obstacle.properties, 4
    )

    assert len(stqbm_reflection_data) == 16

    gridpoints_encoded = [rd.gridpoint_encoded for rd in stqbm_reflection_data]
    assert gridpoints_encoded == [
        (1, 0),
        (0, 0),
        (15, 0),
        (14, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (15, 0),
        (0, 0),
        (1, 0),
        (2, 0),
        (14, 0),
        (13, 0),
        (12, 0),
        (11, 0),
    ]

    qubits_to_invert = [rd.qubits_to_invert for rd in stqbm_reflection_data]
    assert qubits_to_invert == [
        [1, 2, 3],  # 1
        [0, 1, 2, 3],  # 0
        [],  # 15
        [0],  # 14
        [0, 2, 3],  # 2
        [2, 3],  # 3
        [0, 1, 3],  # 4
        [1, 3],  # 5
        [],  # 15
        [0, 1, 2, 3],  # 0
        [1, 2, 3],  # 1
        [0, 2, 3],  # 2
        [0],  # 14
        [1],  # 13
        [0, 1],  # 12
        [2],  # 11
    ]

    velocities_indices_to_invert = [
        rd.velocity_indices_to_reflect for rd in stqbm_reflection_data
    ]
    assert flatten(velocities_indices_to_invert) == [0] * 4 + [1] * 8 + [0] * 4

    distances_from_boundary_points = [
        rd.distance_from_boundary_point for rd in stqbm_reflection_data
    ]
    assert distances_from_boundary_points == [
        (-1, 0),
        (-2, 0),
        (-3, 0),
        (-4, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (-1, 0),
        (-2, 0),
        (-3, 0),
        (-4, 0),
    ]
