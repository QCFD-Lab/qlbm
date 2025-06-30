import numpy as np

from qlbm.components.spacetime.collision.eqc_discretizations import (
    EquivalenceClass,
    EquivalenceClassGenerator,
)
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization


def test_eqc_generator_d1q2():
    generator = EquivalenceClassGenerator(LatticeDiscretization.D1Q2)
    eqcs = generator.generate_equivalence_classes()
    assert len(eqcs) == 0


def test_eqc_generator_d2q4():
    generator = EquivalenceClassGenerator(LatticeDiscretization.D2Q4)
    eqcs = generator.generate_equivalence_classes()
    eqcs_expected = set(
        [
            EquivalenceClass(
                LatticeDiscretization.D2Q4,
                {(True, False, True, False), (False, True, False, True)},
            )
        ]
    )
    assert len(eqcs) == 1
    assert eqcs_expected == eqcs


def test_eqc_generator_d3q6():
    generator = EquivalenceClassGenerator(LatticeDiscretization.D3Q6)
    eqcs = generator.generate_equivalence_classes()
    eqcs_expected = set(
        [
            EquivalenceClass(
                LatticeDiscretization.D3Q6,
                {
                    (True, False, False, True, False, False),  # 100100
                    (False, True, False, False, True, False),  # 010010
                    (False, False, True, False, False, True),  # 001001
                },
            ),
            EquivalenceClass(
                LatticeDiscretization.D3Q6,
                {
                    (True, True, False, True, True, False),  # 110110
                    (True, False, True, True, False, True),  # 101101
                    (False, True, True, False, True, True),  # 011011
                },
            ),
            EquivalenceClass(
                LatticeDiscretization.D3Q6,  # 3
                {
                    (True, True, False, False, True, False),  # 110010
                    (True, False, True, False, False, True),  # 101001
                },
            ),
            EquivalenceClass(
                LatticeDiscretization.D3Q6,  # 4
                {
                    (False, True, False, True, True, False),  # 010110
                    (False, False, True, True, False, True),  # 001101
                },
            ),
            EquivalenceClass(
                LatticeDiscretization.D3Q6,  # 5
                {
                    (True, True, False, True, False, False),  # 110100
                    (False, True, True, False, False, True),  # 011001
                },
            ),
            EquivalenceClass(
                LatticeDiscretization.D3Q6,  # 6
                {
                    (True, False, False, True, True, False),  # 100110
                    (False, False, True, False, True, True),  # 001011
                },
            ),
            EquivalenceClass(
                LatticeDiscretization.D3Q6,  # 7
                {
                    (True, False, True, True, False, False),  # 101100
                    (False, True, True, False, True, False),  # 011010
                },
            ),
            EquivalenceClass(
                LatticeDiscretization.D3Q6,  # 8
                {
                    (False, True, False, False, True, True),  # 010011
                    (True, False, False, True, False, True),  # 100101
                },
            ),
        ]
    )
    assert len(eqcs) == 8
    assert all(eqc in eqcs_expected for eqc in eqcs)
