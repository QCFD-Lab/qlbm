from itertools import product
from typing import Dict, List, Set, Tuple

import numpy as np

from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)
from qlbm.tools.exceptions import LatticeException


class EquivalenceClass:
    discretization: LatticeDiscretization
    velocity_configurations: Set[Tuple[bool, ...]]
    mass: int
    momentum: np.array

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
        self.velocity_configuration = velocity_configurations
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
        return len(self.velocity_configuration)

    def id(self) -> Tuple[int, np.array]:
        return (self.mass, self.momentum)


class EquivalenceClassGenerator:
    discretization: LatticeDiscretization

    def __init__(self, discretization):
        self.discretization = discretization

    def generate_equivalence_classes(self) -> Set[EquivalenceClass]:
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
            EquivalenceClass(self.discretization, v)
            for _, v in equivalence_classes.items()
            if len(v) > 1
        }
