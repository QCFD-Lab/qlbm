import pytest

from qlbm.components.ms.streaming import ControlledIncrementer
from qlbm.lattice import MSLattice


@pytest.fixture
def lattice_asymmetric_medium_3d():
    return MSLattice("test/resources/asymmetric_3d_no_obstacles.json")


@pytest.fixture
def lattice_symmetric_small_2d():
    return MSLattice("test/resources/symmetric_2d_no_obstacles.json")


def test_3d_incrementer_size(lattice_asymmetric_medium_3d):
    inc: ControlledIncrementer = ControlledIncrementer(lattice_asymmetric_medium_3d)

    assert inc.circuit.size() == 78


def test_2d_incrementer_size(lattice_symmetric_small_2d):
    inc: ControlledIncrementer = ControlledIncrementer(lattice_symmetric_small_2d)

    assert inc.circuit.size() == 68
