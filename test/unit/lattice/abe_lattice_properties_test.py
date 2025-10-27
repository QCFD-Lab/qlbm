import pytest

from qlbm.lattice import ABELattice
from qlbm.tools.exceptions import LatticeException


def test_2d_abe_lattice_basic_properties(lattice_2d_16x16_1_obstacle: ABELattice):
    assert lattice_2d_16x16_1_obstacle.num_dims == 2
    assert lattice_2d_16x16_1_obstacle.num_gridpoints == [15, 15]
    assert lattice_2d_16x16_1_obstacle.num_ancilla_qubits == 3
    assert lattice_2d_16x16_1_obstacle.num_grid_qubits == 8
    assert lattice_2d_16x16_1_obstacle.num_velocity_qubits == 2
    assert lattice_2d_16x16_1_obstacle.num_total_qubits == 13


def test_2d_lattice_grid_register(lattice_2d_16x16_1_obstacle: ABELattice):
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
    lattice_2d_16x16_1_obstacle: ABELattice,
):
    assert lattice_2d_16x16_1_obstacle.velocity_index() == [8, 9]

def test_2d_lattice_ancilla_comparator_register(
    lattice_2d_16x16_1_obstacle: ABELattice,
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
    lattice_2d_16x16_1_obstacle: ABELattice,
):
    assert lattice_2d_16x16_1_obstacle.ancillae_obstacle_index() == [8 + 2 + 2]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle.ancillae_obstacle_index(2)
    assert (
        "Cannot index ancilla obstacle register for index 2. Maximum index for this lattice is 0."
        == str(excinfo.value)
    )
