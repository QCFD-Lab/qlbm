import pytest

from qlbm.lattice import ABLattice
from qlbm.tools.exceptions import LatticeException


def test_2d_abe_lattice_basic_properties(lattice_2d_16x16_1_obstacle: ABLattice):
    assert lattice_2d_16x16_1_obstacle.num_dims == 2
    assert lattice_2d_16x16_1_obstacle.num_gridpoints == [15, 15]
    assert lattice_2d_16x16_1_obstacle.num_ancilla_qubits == 3
    assert lattice_2d_16x16_1_obstacle.num_grid_qubits == 8
    assert lattice_2d_16x16_1_obstacle.num_velocity_qubits == 2
    assert lattice_2d_16x16_1_obstacle.num_total_qubits == 13


def test_2d_lattice_grid_register(lattice_2d_16x16_1_obstacle: ABLattice):
    assert lattice_2d_16x16_1_obstacle.grid_index(0) == list(range(4))
    assert lattice_2d_16x16_1_obstacle.grid_index(1) == list(range(4, 8))
    assert lattice_2d_16x16_1_obstacle.grid_index() == list(range(8))

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.grid_index(2)
    assert (
        "Cannot index grid register for dimension 2 in 2-dimensional lattice."
        == str(excinfo.value)
    )


def test_2d_lattice_velocity_register(
    lattice_2d_16x16_1_obstacle: ABLattice,
):
    assert lattice_2d_16x16_1_obstacle.velocity_index() == [8, 9]

def test_2d_lattice_ancilla_comparator_register(
    lattice_2d_16x16_1_obstacle: ABLattice,
):
    assert lattice_2d_16x16_1_obstacle.ancillae_comparator_index(0) == [10, 11]
    assert lattice_2d_16x16_1_obstacle.ancillae_comparator_index() == [10, 11]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.ancillae_comparator_index(1)
    assert (
        "Cannot index ancilla comparator register for index 1 in 2-dimensional lattice. Maximum is 0."
        == str(excinfo.value)
    )


def test_2d_lattice_ancilla_obstacle_register(
    lattice_2d_16x16_1_obstacle: ABLattice,
):
    assert lattice_2d_16x16_1_obstacle.ancillae_obstacle_index() == [8 + 2 + 2]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.ancillae_obstacle_index(2)
    assert (
        "Cannot index ancilla obstacle register for index 2. Maximum index for this lattice is 0."
        == str(excinfo.value)
    )


def test_2d_lattice_no_cuboid_has_no_comparator_register():
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
            "geometry": [
                {
                    "shape": "ymonomial",
                    "exponent": 2,
                    "comparator": "<=",
                    "boundary": "bounceback",
                }
            ],
        }
    )

    assert lattice.num_comparator_qubits == 0
    assert lattice.ancillae_comparator_index() == []
    assert len(lattice.ancilla_comparator_register) == 0
    with pytest.raises(LatticeException) as excinfo:
        lattice.ancillae_comparator_index(0)
    assert (
        "Cannot index ancilla comparator register because this lattice has no comparator qubits."
        == str(excinfo.value)
    )
    assert lattice.ancillae_obstacle_index() == [10]


def test_set_geometries_updates_comparator_register_allocation():
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [2, 6],
                    "y": [5, 10],
                    "boundary": "bounceback",
                }
            ],
        }
    )

    assert lattice.num_comparator_qubits == 2
    assert lattice.ancillae_comparator_index() == [10, 11]

    lattice.set_geometries(
        [
            [
                {
                    "shape": "ymonomial",
                    "exponent": 3,
                    "comparator": ">",
                    "boundary": "bounceback",
                }
            ]
        ]
    )

    assert lattice.num_comparator_qubits == 0
    assert lattice.ancillae_comparator_index() == []
    assert len(lattice.ancilla_comparator_register) == 0
    with pytest.raises(LatticeException) as excinfo:
        lattice.ancillae_comparator_index(0)
    assert (
        "Cannot index ancilla comparator register because this lattice has no comparator qubits."
        == str(excinfo.value)
    )
    assert lattice.ancillae_obstacle_index() == [10]


def test_2d_lattice_no_objects_has_no_comparator_register():
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
        }
    )

    assert lattice.num_comparator_qubits == 0
    assert lattice.ancillae_comparator_index() == []
    assert len(lattice.ancilla_comparator_register) == 0

    with pytest.raises(LatticeException) as excinfo:
        lattice.ancillae_comparator_index(0)
    assert (
        "Cannot index ancilla comparator register because this lattice has no comparator qubits."
        == str(excinfo.value)
    )


@pytest.mark.parametrize(
    "geometry, expected_comparator_qubits, expected_obstacle_qubits, expected_copy_qubits, expected_monomial_qubits",
    [
        (
            [
                {
                    "shape": "cuboid",
                    "x": [2, 6],
                    "y": [5, 10],
                    "boundary": "bounceback",
                }
            ],
            2,
            1,
            0,
            0,
        ),
        (
            [
                {
                    "shape": "ymonomial",
                    "exponent": 2,
                    "comparator": "<=",
                    "boundary": "bounceback",
                }
            ],
            0,
            1,
            4,
            8,
        ),
        (
            [
                {
                    "shape": "cuboid",
                    "x": [1, 3],
                    "y": [7, 11],
                    "boundary": "specular",
                },
                {
                    "shape": "ymonomial",
                    "exponent": 1,
                    "comparator": ">=",
                    "boundary": "bounceback",
                },
            ],
            2,
            2,
            4,
            4,
        ),
        (
            [
                {
                    "shape": "ymonomial",
                    "exponent": 1,
                    "comparator": "<",
                    "boundary": "bounceback",
                },
                {
                    "shape": "ymonomial",
                    "exponent": 4,
                    "comparator": ">",
                    "boundary": "bounceback",
                },
            ],
            0,
            1,
            4,
            16,
        ),
    ],
)
def test_2d_ab_lattice_cuboid_ymonomial_combinations(
    geometry,
    expected_comparator_qubits,
    expected_obstacle_qubits,
    expected_copy_qubits,
    expected_monomial_qubits,
):
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
            "geometry": geometry,
        }
    )

    assert lattice.num_comparator_qubits == expected_comparator_qubits
    assert lattice.num_obstacle_qubits == expected_obstacle_qubits
    assert lattice.num_copy_qubits == expected_copy_qubits
    assert lattice.num_monomial_qubits == expected_monomial_qubits
    assert lattice.num_ancilla_qubits == (
        expected_comparator_qubits + expected_obstacle_qubits
    )

    if expected_comparator_qubits == 2:
        assert lattice.ancillae_comparator_index() == [10, 11]
        assert lattice.ancillae_comparator_index(0) == [10, 11]
        assert len(lattice.ancilla_comparator_register) == 1
    else:
        assert lattice.ancillae_comparator_index() == []
        assert len(lattice.ancilla_comparator_register) == 0
        with pytest.raises(LatticeException) as excinfo:
            lattice.ancillae_comparator_index(0)
        assert (
            "Cannot index ancilla comparator register because this lattice has no comparator qubits."
            == str(excinfo.value)
        )

    expected_obstacle_start = 10 + expected_comparator_qubits
    assert lattice.ancillae_obstacle_index()[0] == expected_obstacle_start


@pytest.mark.parametrize(
    "new_geometries, expected_comparator_qubits, expected_obstacle_qubits, expected_copy_qubits, expected_monomial_qubits, expected_marker_qubits",
    [
        (
            [
                [
                    {
                        "shape": "cuboid",
                        "x": [2, 6],
                        "y": [5, 10],
                        "boundary": "bounceback",
                    }
                ],
                [
                    {
                        "shape": "ymonomial",
                        "exponent": 2,
                        "comparator": "<=",
                        "boundary": "bounceback",
                    }
                ],
            ],
            2,
            1,
            4,
            8,
            1,
        ),
        (
            [
                [
                    {
                        "shape": "ymonomial",
                        "exponent": 1,
                        "comparator": "<",
                        "boundary": "specular",
                    }
                ],
                [
                    {
                        "shape": "ymonomial",
                        "exponent": 3,
                        "comparator": ">",
                        "boundary": "bounceback",
                    }
                ],
            ],
            0,
            2,
            4,
            12,
            1,
        ),
        (
            [
                [
                    {
                        "shape": "cuboid",
                        "x": [1, 4],
                        "y": [1, 4],
                        "boundary": "specular",
                    }
                ],
                [
                    {
                        "shape": "cuboid",
                        "x": [8, 10],
                        "y": [8, 10],
                        "boundary": "bounceback",
                    },
                    {
                        "shape": "ymonomial",
                        "exponent": 0,
                        "comparator": ">=",
                        "boundary": "bounceback",
                    },
                ],
                [
                    {
                        "shape": "ymonomial",
                        "exponent": 5,
                        "comparator": "<=",
                        "boundary": "bounceback",
                    }
                ],
            ],
            2,
            2,
            4,
            20,
            2,
        ),
    ],
)
def test_set_geometries_updates_registers_for_ymonomial_cuboid_combinations(
    new_geometries,
    expected_comparator_qubits,
    expected_obstacle_qubits,
    expected_copy_qubits,
    expected_monomial_qubits,
    expected_marker_qubits,
):
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [2, 6],
                    "y": [5, 10],
                    "boundary": "bounceback",
                }
            ],
        }
    )

    lattice.set_geometries(new_geometries)

    assert lattice.num_comparator_qubits == expected_comparator_qubits
    assert lattice.num_obstacle_qubits == expected_obstacle_qubits
    assert lattice.num_copy_qubits == expected_copy_qubits
    assert lattice.num_monomial_qubits == expected_monomial_qubits
    assert lattice.num_marker_qubits == expected_marker_qubits
    assert lattice.has_multiple_geometries()

    assert lattice.num_ancilla_qubits == (
        expected_comparator_qubits + expected_obstacle_qubits
    )
    assert len(lattice.ancilla_comparator_register) == (
        1 if expected_comparator_qubits > 0 else 0
    )

    if expected_copy_qubits > 0:
        assert len(lattice.copy_register) == 1
    else:
        assert len(lattice.copy_register) == 0

    if expected_monomial_qubits > 0:
        assert len(lattice.monomial_register) == 1
    else:
        assert len(lattice.monomial_register) == 0

    assert len(lattice.marker_index()) == expected_marker_qubits


def test_2d_ymonomial_register_sizes_and_indices():
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
            "geometry": [
                {
                    "shape": "ymonomial",
                    "exponent": 3,
                    "comparator": "<=",
                    "boundary": "bounceback",
                }
            ],
        }
    )

    assert lattice.num_copy_qubits == 4
    assert lattice.num_monomial_qubits == 12
    assert len(lattice.ancillae_copy_index()) == 4
    assert len(lattice.ancillae_monomial_index()) == 12
    assert set(lattice.ancillae_copy_index()).isdisjoint(set(lattice.ancillae_monomial_index()))


def test_3d_cuboid_comparator_qubits_equal_num_dims():
    lattice = ABLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8, "z": 8}, "velocities": "D3Q6"},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [1, 3],
                    "y": [2, 4],
                    "z": [0, 2],
                    "boundary": "bounceback",
                }
            ],
        }
    )

    assert lattice.num_dims == 3
    assert lattice.num_comparator_qubits == 3
    assert len(lattice.ancillae_comparator_index()) == 3
    assert lattice.ancillae_comparator_index(0) == lattice.ancillae_comparator_index()
