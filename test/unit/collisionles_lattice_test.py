import pytest

from qlbm.lattice import CollisionlessLattice
from qlbm.tools.exceptions import LatticeException


@pytest.fixture
def lattice_2d_16x16_1_obstacle() -> CollisionlessLattice:
    return CollisionlessLattice(
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": {"x": 4, "y": 4},
            },
            "geometry": [
                {"x": [4, 6], "y": [3, 12], "boundary": "specular"},
            ],
        }
    )


@pytest.fixture
def lattice_2d_16x16_1_obstacle_asymmetric() -> CollisionlessLattice:
    return CollisionlessLattice(
        {
            "lattice": {
                "dim": {"x": 16, "y": 64},
                "velocities": {"x": 16, "y": 4},
            },
            "geometry": [
                {"x": [4, 6], "y": [3, 12], "boundary": "bounceback"},
            ],
        }
    )


def test_lattice_exception_empty_dict():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice({})

    assert 'Input configuration missing "lattice" properties.' == str(excinfo.value)


def test_lattice_exception_no_dims():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice({"lattice": {}})

    assert 'Lattice configuration missing "dim" properties.' == str(excinfo.value)


def test_lattice_exception_no_velocities():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice({"lattice": {"dim": {}}})

    assert 'Lattice configuration missing "velocities" properties.' == str(
        excinfo.value
    )


def test_lattice_exception_mismatched_velocities_and_dims():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice({"lattice": {"dim": {"x": 64}, "velocities": [4, 4]}})

    assert "Lattice configuration dimensionality is inconsistent." == str(excinfo.value)


def test_lattice_exception_mismatched_too_many_dimensions():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64, "z": 128, "w": 64},
                    "velocities": [4, 4, 4, 4],
                }
            }
        )

    assert (
        "Only 1, 2, and 3-dimensional lattices are supported. Provided lattice has 4 dimensions."
        == str(excinfo.value)
    )


def test_lattice_exception_mismatched_bad_dimensions():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 127},
                    "velocities": {"x": 4, "y": 4},
                }
            }
        )

    assert (
        "Lattice y-dimension has a number of grid points that is not divisible by 2."
        == str(excinfo.value)
    )


def test_lattice_exception_mismatched_bad_velocities():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64},
                    "velocities": {"x": 4, "y": 5},
                }
            }
        )

    assert (
        "Lattice y-dimension has a number of velocities that is not divisible by 2."
        == str(excinfo.value)
    )


def test_lattice_exception_mismatched_bad_object_dimensions():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64},
                    "velocities": {"x": 4, "y": 4},
                },
                "geometry": [
                    {"x": [5, 6], "y": [1, 2], "z": [1, 2], "boundary": "specular"},
                ],
            }
        )

    assert "Obstacle 1 has 3 dimensions whereas the lattice has 2." == str(
        excinfo.value
    )


def test_lattice_exception_mismatched_bad_object_bound_specification():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64},
                    "velocities": {"x": 4, "y": 4},
                },
                "geometry": [
                    {"x": [5, 6, 7], "y": [1, 2], "boundary": "bounceback"},
                ],
            }
        )

    assert "Obstacle 1 is ill-formed in dimension x." == str(excinfo.value)


def test_lattice_exception_mismatched_bad_object_bound_decreasing():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64},
                    "velocities": {"x": 4, "y": 4},
                },
                "geometry": [
                    {"x": [5, 6], "y": [2, 1], "boundary": "specular"},
                ],
            }
        )

    assert "Obstacle 1 y-dimension bounds are not increasing." == str(excinfo.value)


def test_lattice_exception_mismatched_bad_object_out_of_bounds():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64},
                    "velocities": {"x": 4, "y": 4},
                },
                "geometry": [
                    {"x": [-1, 6], "y": [1, 2], "boundary": "bounceback"},
                ],
            }
        )

    assert "Obstacle 1 is out of bounds in the x-dimension." == str(excinfo.value)

    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64},
                    "velocities": {"x": 4, "y": 4},
                },
                "geometry": [
                    {"x": [4, 6], "y": [1, 128], "boundary": "specular"},
                ],
            }
        )

    assert "Obstacle 1 is out of bounds in the y-dimension." == str(excinfo.value)


def test_lattice_exception_bad_object_boundary_conditions():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64},
                    "velocities": {"x": 4, "y": 4},
                },
                "geometry": [
                    {
                        "x": [5, 6],
                        "y": [1, 5],
                        "boundary": "something that doesn't exist",
                    },
                ],
            }
        )

    assert (
        "Obstacle 1 boundary conditions ('something that doesn't exist') are not supported. Supported boundary conditions are ['specular', 'bounceback']."
        == str(excinfo.value)
    )


def test_lattice_exception_no_object_boundary_conditions():
    with pytest.raises(LatticeException) as excinfo:
        CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64},
                    "velocities": {"x": 4, "y": 4},
                },
                "geometry": [
                    {
                        "x": [5, 6],
                        "y": [1, 5],
                    },
                ],
            }
        )

    assert "Obstacle 1 specification includes no boundary conditions." == str(
        excinfo.value
    )


def test_2d_lattice_basic_properties(lattice_2d_16x16_1_obstacle: CollisionlessLattice):
    assert lattice_2d_16x16_1_obstacle.num_dimensions == 2
    assert lattice_2d_16x16_1_obstacle.num_gridpoints == [15, 15]
    assert lattice_2d_16x16_1_obstacle.num_velocities == [3, 3]
    assert lattice_2d_16x16_1_obstacle.num_ancilla_qubits == 6
    assert lattice_2d_16x16_1_obstacle.num_grid_qubits == 8
    assert lattice_2d_16x16_1_obstacle.num_velocity_qubits == 4
    assert lattice_2d_16x16_1_obstacle.num_total_qubits == 18


def test_2d_lattice_basic_qubit_register_size(
    lattice_2d_16x16_1_obstacle: CollisionlessLattice,
):
    # Ancilla velocity (1r2q)
    # ancilla obstacle (1r2q)
    # Ancilla comparator (1r2q)
    # Grid (2r8q)
    # Velocity (2r2q)
    # Direction (2r2q)
    assert len(lattice_2d_16x16_1_obstacle.registers) == 9  # type: ignore
    assert sum([qr.size for qr in lattice_2d_16x16_1_obstacle.registers]) == 18  # type: ignore


def test_2d_lattice_ancilla_velocity_register(
    lattice_2d_16x16_1_obstacle: CollisionlessLattice,
):
    # Ancilla velocity (1r2q)
    # ancilla obstacle (1r2q)
    # Ancilla reflection (1r2q)
    # Ancilla comparator (1r2q)
    # Grid (2r8q)
    # Velocity (2r2q)
    # Direction (2r2q)
    assert lattice_2d_16x16_1_obstacle.ancillae_velocity_index(0) == [0]
    assert lattice_2d_16x16_1_obstacle.ancillae_velocity_index(1) == [1]
    assert lattice_2d_16x16_1_obstacle.ancillae_velocity_index() == [0, 1]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.ancillae_velocity_index(17)
    assert (
        "Cannot index ancilla velocity register for dimension 17 in 2-dimensional lattice."
        == str(excinfo.value)
    )


def test_2d_lattice_ancilla_obstacle_register(
    lattice_2d_16x16_1_obstacle: CollisionlessLattice,
):
    # Ancilla velocity (1r2q)
    # ancilla obstacle (1r2q)
    # Ancilla reflection (1r2q)
    # Ancilla comparator (1r2q)
    # Grid (2r8q)
    # Velocity (2r2q)
    # Direction (2r2q)
    assert lattice_2d_16x16_1_obstacle.ancillae_obstacle_index(0) == [2]
    assert lattice_2d_16x16_1_obstacle.ancillae_obstacle_index(1) == [3]
    assert lattice_2d_16x16_1_obstacle.ancillae_obstacle_index() == [2, 3]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.ancillae_obstacle_index(2)
    assert (
        "Cannot index ancilla obstacle register for dimension 2 in 2-dimensional lattice."
        == str(excinfo.value)
    )


def test_2d_lattice_ancilla_comparator_register(
    lattice_2d_16x16_1_obstacle: CollisionlessLattice,
):
    assert lattice_2d_16x16_1_obstacle.ancillae_comparator_index(0) == [4, 5]
    assert lattice_2d_16x16_1_obstacle.ancillae_comparator_index() == [4, 5]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.ancillae_comparator_index(1)
    assert (
        "Cannot index ancilla comparator register for index 1 in 2-dimensional lattice. Maximum is 0."
        == str(excinfo.value)
    )


def test_2d_lattice_grid_register(lattice_2d_16x16_1_obstacle: CollisionlessLattice):
    assert lattice_2d_16x16_1_obstacle.grid_index(0) == [6, 7, 8, 9]
    assert lattice_2d_16x16_1_obstacle.grid_index(1) == [10, 11, 12, 13]
    assert lattice_2d_16x16_1_obstacle.grid_index() == list(range(6, 14))

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.grid_index(2)
    assert (
        "Cannot index grid register for dimension 2 in 2-dimensional lattice."
        == str(excinfo.value)
    )


def test_2d_lattice_velocity_register(
    lattice_2d_16x16_1_obstacle: CollisionlessLattice,
):
    assert lattice_2d_16x16_1_obstacle.velocity_index(0) == [14]
    assert lattice_2d_16x16_1_obstacle.velocity_index(1) == [15]
    assert lattice_2d_16x16_1_obstacle.velocity_index() == [14, 15]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.velocity_index(2)
    assert (
        "Cannot index velocity register for dimension 2 in 2-dimensional lattice."
        == str(excinfo.value)
    )


def test_2d_lattice_velocity_dir_register(
    lattice_2d_16x16_1_obstacle: CollisionlessLattice,
):
    assert lattice_2d_16x16_1_obstacle.velocity_dir_index(0) == [16]
    assert lattice_2d_16x16_1_obstacle.velocity_dir_index(1) == [17]
    assert lattice_2d_16x16_1_obstacle.velocity_dir_index() == [16, 17]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.velocity_dir_index(2)
    assert (
        "Cannot index velocity direction register for dimension 2 in 2-dimensional lattice."
        == str(excinfo.value)
    )


def test_2d_asymmetric_lattice_ancialle_registers(
    lattice_2d_16x16_1_obstacle: CollisionlessLattice,
    lattice_2d_16x16_1_obstacle_asymmetric: CollisionlessLattice,
):
    assert lattice_2d_16x16_1_obstacle_asymmetric.num_gridpoints == [15, 63]
    assert lattice_2d_16x16_1_obstacle_asymmetric.num_velocities == [15, 3]
    assert lattice_2d_16x16_1_obstacle_asymmetric.num_ancilla_qubits == 6
    assert lattice_2d_16x16_1_obstacle_asymmetric.num_grid_qubits == 10
    assert lattice_2d_16x16_1_obstacle_asymmetric.num_velocity_qubits == 6
    assert lattice_2d_16x16_1_obstacle_asymmetric.num_total_qubits == 22

    assert len(lattice_2d_16x16_1_obstacle_asymmetric.registers) == len(
        lattice_2d_16x16_1_obstacle.registers
    )  # type: ignore
    assert (
        sum([qr.size for qr in lattice_2d_16x16_1_obstacle_asymmetric.registers]) == 22  # type: ignore
    )

    dim_range = [None, 0, 1]
    for dim in dim_range:
        assert lattice_2d_16x16_1_obstacle.ancillae_velocity_index(
            dim
        ) == lattice_2d_16x16_1_obstacle_asymmetric.ancillae_velocity_index(dim)

        assert lattice_2d_16x16_1_obstacle.ancillae_obstacle_index(
            dim
        ) == lattice_2d_16x16_1_obstacle_asymmetric.ancillae_obstacle_index(dim)

        if dim != 1:
            assert lattice_2d_16x16_1_obstacle.ancillae_comparator_index(
                dim
            ) == lattice_2d_16x16_1_obstacle_asymmetric.ancillae_comparator_index(dim)


def test_2d_asymmetric_lattice_grid_registers(
    lattice_2d_16x16_1_obstacle_asymmetric: CollisionlessLattice,
):
    assert lattice_2d_16x16_1_obstacle_asymmetric.grid_index(0) == [6, 7, 8, 9]
    assert lattice_2d_16x16_1_obstacle_asymmetric.grid_index(1) == list(range(10, 16))
    assert lattice_2d_16x16_1_obstacle_asymmetric.grid_index() == list(range(6, 16))

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle_asymmetric.grid_index(2)
    assert (
        "Cannot index grid register for dimension 2 in 2-dimensional lattice."
        == str(excinfo.value)
    )


def test_2d_asymmetric_lattice_velocity_register(
    lattice_2d_16x16_1_obstacle_asymmetric: CollisionlessLattice,
):
    assert lattice_2d_16x16_1_obstacle_asymmetric.velocity_index(0) == [16, 17, 18]
    assert lattice_2d_16x16_1_obstacle_asymmetric.velocity_index(1) == [19]
    assert lattice_2d_16x16_1_obstacle_asymmetric.velocity_index() == list(
        range(16, 20)
    )

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle_asymmetric.velocity_index(2)
    assert (
        "Cannot index velocity register for dimension 2 in 2-dimensional lattice."
        == str(excinfo.value)
    )


def test_2d_asymmetric_lattice_velocity_dir_register(
    lattice_2d_16x16_1_obstacle_asymmetric: CollisionlessLattice,
):
    assert lattice_2d_16x16_1_obstacle_asymmetric.velocity_dir_index(0) == [20]
    assert lattice_2d_16x16_1_obstacle_asymmetric.velocity_dir_index(1) == [21]
    assert lattice_2d_16x16_1_obstacle_asymmetric.velocity_dir_index() == [20, 21]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle_asymmetric.velocity_dir_index(2)
    assert (
        "Cannot index velocity direction register for dimension 2 in 2-dimensional lattice."
        == str(excinfo.value)
    )