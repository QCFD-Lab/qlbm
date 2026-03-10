import pytest

from qlbm.lattice import OHLattice
from qlbm.tools.exceptions import LatticeException


def test_2d_lattice_basic_properties(lattice_2d_16x16_1_obstacle_oh: OHLattice):
    assert lattice_2d_16x16_1_obstacle_oh.num_dims == 2
    assert lattice_2d_16x16_1_obstacle_oh.num_gridpoints == [15, 15]
    assert lattice_2d_16x16_1_obstacle_oh.num_ancilla_qubits == 3
    assert lattice_2d_16x16_1_obstacle_oh.num_grid_qubits == 8
    assert lattice_2d_16x16_1_obstacle_oh.num_velocity_qubits == 9
    assert lattice_2d_16x16_1_obstacle_oh.num_total_qubits == 20


def test_2d_lattice_grid_register(lattice_2d_16x16_1_obstacle_oh: OHLattice):
    assert lattice_2d_16x16_1_obstacle_oh.grid_index(0) == list(range(4))
    assert lattice_2d_16x16_1_obstacle_oh.grid_index(1) == list(range(4, 8))
    assert lattice_2d_16x16_1_obstacle_oh.grid_index() == list(range(8))

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle_oh.grid_index(2)
    assert (
        "Cannot index grid register for dimension 2 in 2-dimensional lattice."
        == str(excinfo.value)
    )


def test_2d_lattice_velocity_register(
    lattice_2d_16x16_1_obstacle_oh: OHLattice,
):
    assert lattice_2d_16x16_1_obstacle_oh.velocity_index() == list(range(8, 17))


def test_2d_lattice_ancilla_comparator_register(
    lattice_2d_16x16_1_obstacle_oh: OHLattice,
):
    assert lattice_2d_16x16_1_obstacle_oh.ancillae_comparator_index(0) == [17, 18]
    assert lattice_2d_16x16_1_obstacle_oh.ancillae_comparator_index() == [17, 18]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle_oh.ancillae_comparator_index(1)
    assert (
        "Cannot index ancilla comparator register for index 1 in 2-dimensional lattice. Maximum is 0."
        == str(excinfo.value)
    )


def test_2d_lattice_ancilla_obstacle_register(
    lattice_2d_16x16_1_obstacle_oh: OHLattice,
):
    assert lattice_2d_16x16_1_obstacle_oh.ancillae_obstacle_index() == [19]

    with pytest.raises(LatticeException) as excinfo:
        lattice_2d_16x16_1_obstacle_oh.ancillae_obstacle_index(2)
    assert (
        "Cannot index ancilla obstacle register for index 2. Maximum index for this lattice is 0."
        == str(excinfo.value)
    )
