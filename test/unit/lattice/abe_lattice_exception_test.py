import pytest

from qlbm.lattice import ABLattice
from qlbm.tools.exceptions import LatticeException


def test_lattice_exception_empty_dict():
    with pytest.raises(LatticeException) as excinfo:
        ABLattice({})

    assert 'Input configuration missing "lattice" properties.' == str(excinfo.value)


def test_lattice_exception_no_dims():
    with pytest.raises(LatticeException) as excinfo:
        ABLattice({"lattice": {}})

    assert 'Lattice configuration missing "dim" properties.' == str(excinfo.value)


def test_lattice_exception_no_velocities():
    with pytest.raises(LatticeException) as excinfo:
        ABLattice({"lattice": {"dim": {}}})

    assert 'Lattice configuration missing "velocities" properties.' == str(
        excinfo.value
    )

def test_lattice_exception_mismatched_velocities_and_dims():
    with pytest.raises(LatticeException) as excinfo:
        ABLattice({"lattice": {"dim": {"x": 64}, "velocities": "D2Q4"}})
    
    assert "Velocity specification dimensions (2) do not match lattice dimensions (1)." == str(excinfo.value)


def test_lattice_exception_unsupported_discretization():
    with pytest.raises(LatticeException) as excinfo:
        ABLattice({"lattice": {"dim": {"x": 64}, "velocities": {"x": 4}}})

    assert 'Discretization LatticeDiscretization.CFLDISCRETIZATION is not supported.' == str(
        excinfo.value
    )

def test_lattice_exception_mismatched_bad_dimensions():
    with pytest.raises(LatticeException) as excinfo:
        ABLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 127},
                    "velocities": "D2Q4",
                }
            }
        )

    assert (
        "Lattice has a number of grid points that is not divisible by 2 in dimension y."
        == str(excinfo.value)
    )

def test_lattice_exception_mismatched_bad_object_dimensions():
    with pytest.raises(LatticeException) as excinfo:
        ABLattice(
            {
                "lattice": {
                    "dim": {"x": 64, "y": 64},
                    "velocities": "D2Q4",
                },
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [5, 6],
                        "y": [1, 2],
                        "z": [1, 2],
                        "boundary": "specular",
                    },
                ],
            }
        )

    assert "Obstacle 1 has 3 dimensions whereas the lattice has 2." == str(
        excinfo.value
    )
        