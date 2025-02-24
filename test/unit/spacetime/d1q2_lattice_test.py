import pytest

from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import (
    VonNeumannNeighbor,
    VonNeumannNeighborType,
)
from qlbm.tools.exceptions import LatticeException


def test_lattice_num_qubits_1(
    lattice_1d_16_1_obstacle_1_timestep: SpaceTimeLattice,
):
    assert lattice_1d_16_1_obstacle_1_timestep.properties.get_num_total_qubits() == 10


def test_lattice_num_qubits_2(
    lattice_1d_16_1_obstacle_2_timesteps: SpaceTimeLattice,
):
    assert lattice_1d_16_1_obstacle_2_timesteps.properties.get_num_total_qubits() == 14


def test_lattice_num_velocities(dummy_1d_lattice: SpaceTimeLattice):
    for num_timesteps in range(10):
        assert (
            dummy_1d_lattice.properties.get_num_velocity_qubits(num_timesteps)
            == 4 * num_timesteps + 2
        )


def test_get_neighbor_indices_1_timestep(
    lattice_1d_16_1_obstacle_1_timestep: SpaceTimeLattice,
):
    (
        extreme_points_dict,
        intermediate_points_dict,
    ) = lattice_1d_16_1_obstacle_1_timestep.properties.get_neighbor_indices()

    assert len(extreme_points_dict) == 1
    assert len(intermediate_points_dict) == 0

    assert extreme_points_dict[1] == [
        VonNeumannNeighbor((1, 0), 1, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-1, 0), 2, VonNeumannNeighborType.EXTREME),
    ]


def test_lattice_1d_16_1_obstacle_2_timesteps(
    lattice_1d_16_1_obstacle_2_timesteps: SpaceTimeLattice,
):
    (
        extreme_points_dict,
        intermediate_points_dict,
    ) = lattice_1d_16_1_obstacle_2_timesteps.properties.get_neighbor_indices()

    assert len(extreme_points_dict) == 2
    assert len(intermediate_points_dict) == 0

    assert extreme_points_dict[1] == [
        VonNeumannNeighbor((1, 0), 1, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-1, 0), 2, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[2] == [
        VonNeumannNeighbor((2, 0), 3, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-2, 0), 4, VonNeumannNeighborType.EXTREME),
    ]


def test_lattice_1d_16_1_obstacle_5_timesteps(
    lattice_1d_16_1_obstacle_5_timesteps: SpaceTimeLattice,
):
    (
        extreme_points_dict,
        intermediate_points_dict,
    ) = lattice_1d_16_1_obstacle_5_timesteps.properties.get_neighbor_indices()

    assert len(extreme_points_dict) == 5
    assert len(intermediate_points_dict) == 0

    assert extreme_points_dict[1] == [
        VonNeumannNeighbor((1, 0), 1, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-1, 0), 2, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[2] == [
        VonNeumannNeighbor((2, 0), 3, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-2, 0), 4, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[3] == [
        VonNeumannNeighbor((3, 0), 5, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-3, 0), 6, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[4] == [
        VonNeumannNeighbor((4, 0), 7, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-4, 0), 8, VonNeumannNeighborType.EXTREME),
    ]
    assert extreme_points_dict[5] == [
        VonNeumannNeighbor((5, 0), 9, VonNeumannNeighborType.EXTREME),
        VonNeumannNeighbor((-5, 0), 10, VonNeumannNeighborType.EXTREME),
    ]


def test_num_gridpoints_within_distance(dummy_1d_lattice: SpaceTimeLattice):
    assert dummy_1d_lattice.properties.get_num_gridpoints_within_distance(1) == 3
    assert dummy_1d_lattice.properties.get_num_gridpoints_within_distance(2) == 5
    assert dummy_1d_lattice.properties.get_num_gridpoints_within_distance(3) == 7
    assert dummy_1d_lattice.properties.get_num_gridpoints_within_distance(4) == 9
    assert dummy_1d_lattice.properties.get_num_gridpoints_within_distance(5) == 11


def test_get_index_of_neighbor_origin(lattice_1d_16_1_obstacle_5_timesteps):
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((0, 0))
        == 0
    )


def test_get_index_of_neighbor_dist_1(lattice_1d_16_1_obstacle_5_timesteps):
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((1, 0))
        == 1
    )
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((-1, 0))
        == 2
    )


def test_get_index_of_neighbor_dist_2(lattice_1d_16_1_obstacle_5_timesteps):
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((2, 0))
        == 3
    )
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((-2, 0))
        == 4
    )


def test_get_index_of_neighbor_dist_3_4_5(lattice_1d_16_1_obstacle_5_timesteps):
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((2, 0))
        == 3
    )
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((-2, 0))
        == 4
    )
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((3, 0))
        == 5
    )
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((-3, 0))
        == 6
    )
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((4, 0))
        == 7
    )
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((-4, 0))
        == 8
    )
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((5, 0))
        == 9
    )
    assert (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_index_of_neighbor((-5, 0))
        == 10
    )


def test_streaming_line_dist_1(lattice_1d_16_1_obstacle_1_timestep):
    streaming_line_x_pos = (
        lattice_1d_16_1_obstacle_1_timestep.properties.get_streaming_lines(0, True)
    )
    streaming_line_x_neg = (
        lattice_1d_16_1_obstacle_1_timestep.properties.get_streaming_lines(0, False)
    )

    assert streaming_line_x_pos == [[1, 0, 2]]
    assert streaming_line_x_neg == [
        list(reversed(line)) for line in streaming_line_x_pos
    ]


def test_streaming_line_dist_2(lattice_1d_16_1_obstacle_2_timesteps):
    streaming_line_x_pos = (
        lattice_1d_16_1_obstacle_2_timesteps.properties.get_streaming_lines(0, True)
    )
    streaming_line_x_neg = (
        lattice_1d_16_1_obstacle_2_timesteps.properties.get_streaming_lines(0, False)
    )

    assert streaming_line_x_pos == [[3, 1, 0, 2, 4]]
    assert streaming_line_x_neg == [
        list(reversed(line)) for line in streaming_line_x_pos
    ]


def test_streaming_line_dist_5(lattice_1d_16_1_obstacle_5_timesteps):
    streaming_line_x_pos = (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_streaming_lines(0, True)
    )
    streaming_line_x_neg = (
        lattice_1d_16_1_obstacle_5_timesteps.properties.get_streaming_lines(0, False)
    )

    assert streaming_line_x_pos == [[9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 10]]
    assert streaming_line_x_neg == [
        list(reversed(line)) for line in streaming_line_x_pos
    ]


def test_streaming_line_bad_dimension(lattice_1d_16_1_obstacle_5_timesteps):
    for dim in [1, 2, 3]:
        with pytest.raises(LatticeException) as excinfo:
            lattice_1d_16_1_obstacle_5_timesteps.properties.get_streaming_lines(
                dim, True if dim % 2 else False, dim
            )

        assert (
            f"Dimension {dim} unsupported, D1Q2 lattices only support dimension 0."
            == str(excinfo.value)
        )


def test_bad_lattice_specification_velocities():
    with pytest.raises(LatticeException) as excinfo_measure:
        SpaceTimeLattice(
            2,
            {
                "lattice": {
                    "dim": {"x": 16},
                    "velocities": {"x": 4},
                },
            },
        )

    assert (
        "Unsupported number of velocities for 1D: 4. Only D1Q2 is supported at the moment."
        == str(excinfo_measure.value)
    )


def test_volumetric_ancilla_qubit_combinations_no_overflow(
    volumetric_lattice_1d_16_1_obstacle_1_timestep,
):
    assert (
        volumetric_lattice_1d_16_1_obstacle_1_timestep.volumetric_ancilla_qubit_combinations(
            [False]
        )
        == [[10, 11]]
    )


def test_volumetric_ancilla_qubit_combinations_overflow(
    volumetric_lattice_1d_16_1_obstacle_1_timestep,
):
    assert (
        volumetric_lattice_1d_16_1_obstacle_1_timestep.volumetric_ancilla_qubit_combinations(
            [True]
        )
        == [[10], [11]]
    )


def test_volumetric_ancilla_qubit_combinations_exception_overflow(
    lattice_1d_16_1_obstacle_1_timestep,
):
    with pytest.raises(LatticeException) as excinfo_measure:
        lattice_1d_16_1_obstacle_1_timestep.volumetric_ancilla_qubit_combinations(
            [True]
        )

    assert (
        "Lattice contains no comparator ancilla qubits. To enable comparator (volumetric) operations, construct the Lattice with use_volumetric_ops=True."
        == str(excinfo_measure.value)
    )
