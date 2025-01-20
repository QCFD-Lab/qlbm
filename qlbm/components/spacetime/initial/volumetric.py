"""Prepares the initial state for the :class:`.SpaceTimeQLBM` for a volumetric region."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple, cast

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, XGate
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.components.collisionless.primitives import Comparator, ComparatorMode
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.utils import flatten


class VolumetricSpaceTimeInitialConditions(LBMPrimitive):
    """
    Prepares the initial state for the :class:`.SpaceTimeQLBM` for a volumetric region.

    Work in progress.
    """

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        cuboid_bounds: List[Tuple[int, int]],
        velocity_profile: Tuple[int, ...],
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)

        self.lattice = lattice
        self.cuboid_bounds = cuboid_bounds
        self.velocity_profile = velocity_profile

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        circuit.h(self.lattice.grid_index())

        for manhattan_distance in range(self.lattice.num_timesteps + 1):
            for neighbor in (
                self.lattice.extreme_point_indices[manhattan_distance]
                + (
                    flatten(
                        list(
                            self.lattice.intermediate_point_indices[
                                manhattan_distance
                            ].values()
                        )
                    )
                    if manhattan_distance in self.lattice.intermediate_point_indices
                    else []
                )
                if manhattan_distance > 0
                else [self.lattice.properties.origin]
            ):
                periodic_volume_bounds: List[
                    Tuple[Tuple[int, int], Tuple[bool, bool]]
                ] = self.lattice.comparator_periodic_volume_bounds(
                    cast(
                        List[Tuple[int, int]],
                        [
                            tuple(
                                dim_bound[i]
                                + neighbor.coordinates_relative[
                                    dim
                                ]  # Add the offset to the bound in each dimension
                                for i in range(len(dim_bound))
                            )
                            for dim, dim_bound in enumerate(self.cuboid_bounds)
                        ],
                    )
                )

                comparators = [
                    [
                        Comparator(
                            self.lattice.properties.get_num_grid_qubits() + 1,
                            pvb[0][bound],
                            self.__adjusted_comparator_mode(bound),
                            logger=self.logger,
                        )
                        for bound in [False, True]
                    ]
                    for pvb in periodic_volume_bounds
                ]

                for dim in range(self.lattice.num_dims):
                    for bound in [False, True]:
                        circuit.compose(
                            comparators[dim][bound].circuit,
                            qubits=self.lattice.grid_index()
                            + [self.lattice.ancilla_comparator_index(dim)[bound]],
                            inplace=True,
                        )

                for (
                    control_qubit_sequence
                ) in self.lattice.volumetric_ancilla_qubit_combinations(
                    [any(pvb[1]) for pvb in periodic_volume_bounds]
                ):
                    circuit.compose(
                        MCMT(
                            XGate(),
                            num_ctrl_qubits=len(control_qubit_sequence),
                            num_target_qubits=sum(
                                self.velocity_profile
                            ),  # The sum is equal to the number of velocities set to true
                        ),
                        qubits=control_qubit_sequence
                        + flatten(
                            [
                                self.lattice.velocity_index(neighbor.neighbor_index, c)
                                for c, is_velocity_enabled in enumerate(
                                    self.velocity_profile
                                )
                                if is_velocity_enabled
                            ]
                        ),
                        inplace=True,
                    )

                for dim in range(self.lattice.num_dims):
                    for bound in [False, True]:
                        circuit.compose(
                            comparators[dim][bound].circuit,
                            qubits=self.lattice.grid_index()
                            + [self.lattice.ancilla_comparator_index(dim)[bound]],
                            inplace=True,
                        )

        return circuit

    def __adjusted_comparator_mode(self, bound: bool) -> ComparatorMode:
        return ComparatorMode.LE if (bound) else ComparatorMode.GE

    @override
    def __str__(self) -> str:
        return f"[Primitive VolumetricSpaceTimeInitialConditions with range={self.cuboid_bounds}, profile = {self.velocity_profile}, and lattice={self.lattice}]"
