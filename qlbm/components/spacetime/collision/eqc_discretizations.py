"""Equivalence class utility functions. For a discussion of equivalence classes, we refer the reader to Section 4 of :cite:`spacetime2`."""

from itertools import product
from typing import Dict, Set, Tuple, override

import numpy as np

from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)
from qlbm.tools.exceptions import LatticeException


class EquivalenceClass:
    """
    Class representing LGA equivalence classes.

    In ``qlbm``, an equivalence class is a set of velocity configurations that share the same mass and momentum.
    For a more in depth explanation, consult Section 4 of :cite:p:`spacetime2`.

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`discretization`
          - The :class:`.LatticeDiscretization` that the equivalence class belongs to.
        * - :attr:`velocity_configurations`
          - The :class:`Set[Tuple[bool, ...]]` that contains the velocity configurations of the equivalence class. Configurations are stored as ``q``-tuples where an entry is ``True`` if the velocity channel is occupied and ``False`` otherwise.

    .. list-table:: Class Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`mass`
          - The total mass of the equivalence class, which is the sum of all occupied velocity channels.
        * - :attr:`momentum`
          - The total momentum of the equivalence class, which is the vector sum of all occupied velocity channels multiplid by their :class:`.LatticeDiscretizationProperties` velocity contribution.
    """

    discretization: LatticeDiscretization
    velocity_configurations: Set[Tuple[bool, ...]]
    mass: int
    momentum: np.typing.NDArray

    def __init__(
        self,
        discretization: LatticeDiscretization,
        velocity_configurations: Set[Tuple[bool, ...]],
    ):
        if len(velocity_configurations) < 2:
            raise LatticeException(
                f"Equivalence class must have at least two configurations. Provided velocity profiles are {velocity_configurations}."
            )
        num_velocities = LatticeDiscretizationProperties.get_num_velocities(
            discretization
        )
        if not all(len(v) == num_velocities for v in velocity_configurations):
            raise LatticeException(
                f"All velocity configurations must have length {num_velocities}. Provided configurations are {velocity_configurations}."
            )

        self.discretization = discretization
        self.velocity_configurations = velocity_configurations
        self.mass = sum(list(velocity_configurations)[0])

        velocity_vectors = LatticeDiscretizationProperties.get_velocity_vectors(
            discretization
        )
        if not all(
            (sum(velocity_cfg) == self.mass) for velocity_cfg in velocity_configurations
        ):
            raise LatticeException("Velocity configurations have different masses.")
        self.momentum = sum(
            [
                velocity_vectors[i] * list(velocity_configurations)[0][i]
                for i in range(len(list(velocity_configurations)[0]))
            ]
        )
        if not all(
            (
                np.array_equal(
                    self.momentum,
                    np.sum(
                        [
                            velocity_vectors[i] * v[i]
                            for i in range(len(velocity_vectors))
                        ],
                        axis=0,
                    ),
                )
            )
            for v in velocity_configurations
        ):
            raise LatticeException("Velocity configurations have different momenta.")

    def size(self) -> int:
        """
        The number of velocity configurations in the equivalence class.

        Returns
        -------
        int
            The number of velocity configurations in the equivalence class.
        """
        return len(self.velocity_configurations)

    def id(self) -> Tuple[int, np.typing.NDArray]:
        """
        The identifier of the equivalence class.

        For a given discretization, an equivalence class can be uniquely identified by the common mass and momentum of its velocity configurations.

        Returns
        -------
        Tuple[int, np.typing.NDArray]
            The mass and momentum of the equivalence class.
        """
        return (self.mass, self.momentum)

    @override
    def __eq__(self, value):
        if not isinstance(value, EquivalenceClass):
            return False
        return (
            self.discretization == value.discretization
            and self.velocity_configurations == value.velocity_configurations
            and self.mass == value.mass
            and np.array_equal(self.momentum, value.momentum)
        )

    @override
    def __hash__(self):
        return hash((self.discretization, tuple(self.velocity_configurations)))


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
            mass = np.sum(state)
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
