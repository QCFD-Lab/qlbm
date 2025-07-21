"""Generator class for equivalence classes in LGA-based algorithms."""

from itertools import product
from typing import Dict, Set

import numpy as np

from qlbm.lattice.eqc.eqc import EquivalenceClass
from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)


class EquivalenceClassGenerator:
    """
    A class that generates equivalence classes for a given lattice discretization.

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`discretization`
          - The :class:`.LatticeDiscretization` that the equivalence class belongs to.

    Example usage:

    .. code-block:: python
        :linenos:

        from qlbm.lattice import LatticeDiscretization
        from qlbm.lattice.eqc import EquivalenceClassGenerator

        # Generate some equivalence classes
        eqcs = EquivalenceClassGenerator(
            LatticeDiscretization.D3Q6
        ).generate_equivalence_classes()

        print(eqcs.pop().get_bitstrings())
    """

    discretization: LatticeDiscretization

    def __init__(self, discretization):
        self.discretization = discretization

    def generate_equivalence_classes(self) -> Set[EquivalenceClass]:
        """
        Generates equivalence classes for the given lattice discretization.

        Returns
        -------
        Set[EquivalenceClass]
            All equivalence classes of the discretization.
        """
        equivalence_classes: Dict = {}
        for state in product(
            [0, 1],
            repeat=LatticeDiscretizationProperties.get_num_velocities(
                self.discretization
            ),
        ):
            velocity_vectors = LatticeDiscretizationProperties.get_velocity_vectors(
                self.discretization
            )
            state = np.array(state)  # type: ignore
            mass = state @ LatticeDiscretizationProperties.get_channel_masses(
                self.discretization
            )
            momentum = np.sum(state[:, None] * velocity_vectors, axis=0)  # type: ignore

            key = (mass, tuple(momentum))
            if key not in equivalence_classes:
                equivalence_classes[key] = []
            equivalence_classes[key].append(state)

        return {
            EquivalenceClass(self.discretization, set(tuple(cfg.tolist()) for cfg in v))
            for _, v in equivalence_classes.items()
            if len(v) > 1
        }
