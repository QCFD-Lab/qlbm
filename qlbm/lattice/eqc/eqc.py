from typing import List, Set, Tuple, override

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

    def id(self) -> Tuple[int, List[float]]:
        """
        The identifier of the equivalence class.

        For a given discretization, an equivalence class can be uniquely identified by the common mass and momentum of its velocity configurations.

        Returns
        -------
        Tuple[int, List[float]]
            The mass and momentum of the equivalence class.
        """
        return (self.mass, self.momentum.tolist())

    def get_bitstrings(self) -> List[str]:
        """
        Returns the velocity configurations as bitstrings.

        Returns
        -------
        List[str]
            The velocity configurations of the equivalence class as bitstrings.
        """
        return [''.join(['1' if x else '0' for x in cfg]) for cfg in self.velocity_configurations]

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