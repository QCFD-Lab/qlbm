from typing import Set, Tuple

import pytest

from qlbm.lattice.eqc.eqc import EquivalenceClass
from qlbm.lattice.eqc.eqc_generator import (
    EquivalenceClassGenerator,
)
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException


def test_equivalence_class_bad_number_of_velocity_configurations():
    velocity_profile: Set[Tuple[bool, ...]] = {(True, True, True, True)}
    with pytest.raises(LatticeException) as excinfo:
        EquivalenceClass(LatticeDiscretization.D2Q4, velocity_profile)
    assert (
        f"Equivalence class must have at least two configurations. Provided velocity profiles are {velocity_profile}."
        == str(excinfo.value)
    )


def test_equivalence_class_bad_number_velocity_configurations():
    velocity_profile: Set[Tuple[bool, ...]] = {(True, True, True), (True, True, True, False)}
    with pytest.raises(LatticeException) as excinfo:
        EquivalenceClass(LatticeDiscretization.D2Q4, velocity_profile)
    assert (
        f"All velocity configurations must have length 4. Provided configurations are {velocity_profile}."
        == str(excinfo.value)
    )


def test_equivalence_class_bad_mass():
    velocity_profile: Set[Tuple[bool, ...]] = {(True, False, True, False), (True, True, True, False)}
    with pytest.raises(LatticeException) as excinfo:
        EquivalenceClass(LatticeDiscretization.D2Q4, velocity_profile)
    assert "Velocity configurations have different masses." == str(excinfo.value)


def test_equivalence_class_bad_momentum():
    velocity_profile: Set[Tuple[bool, ...]] = {(True, False, True, False), (True, True, False, False)}
    with pytest.raises(LatticeException) as excinfo:
        EquivalenceClass(LatticeDiscretization.D2Q4, velocity_profile)
    assert "Velocity configurations have different momenta." == str(excinfo.value)


def test_equivalence_class_ok():
    velocity_profile: Set[Tuple[bool, ...]] = {(True, False, True, False), (False, True, False, True)}
    eqc = EquivalenceClass(LatticeDiscretization.D2Q4, velocity_profile)
    assert eqc.mass == 2
    assert eqc.momentum.tolist() == [0, 0]
