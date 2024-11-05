import pytest

from qlbm.lattice.blocks import Block
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


@pytest.fixture
def simple_1d_block() -> Block:
    return Block([(5, 11)], [4], "bounceback")


@pytest.fixture
def dummy_lattice() -> SpaceTimeLattice:
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
                {"x": [5, 11], "boundary": "bounceback"},
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
                {"x": [5, 11], "boundary": "bounceback"},
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


def test_get_spacetime_reflection_data_constructor_1(
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
        rd.velocity_index_to_reflect for rd in stqbm_reflection_data
    ]
    assert velocities_indices_to_invert == [0, 1, 1, 0]

    distances_from_boundary_points = [
        rd.distance_from_boundary_point for rd in stqbm_reflection_data
    ]
    assert distances_from_boundary_points == [(-1, 0), (1, 0), (1, 0), (-1, 0)]


def test_get_spacetime_reflection_data_neighbor_velocity_pairs_1(
    simple_1d_block, lattice_1d_16_1_obstacle_1_timestep
):
    stqbm_reflection_data = simple_1d_block.get_spacetime_reflection_data_d1q2(
        lattice_1d_16_1_obstacle_1_timestep.properties, 1
    )

    neighbor_velocity_pairs = [
        rd.neighbor_velocity_pairs for rd in stqbm_reflection_data
    ]
    assert neighbor_velocity_pairs == [
        ((0, 1), (1, 0)),
        ((0, 0), (2, 1)),
        ((0, 0), (2, 1)),
        ((0, 1), (1, 0)),
    ]
