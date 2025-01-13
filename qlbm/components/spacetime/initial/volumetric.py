from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple, cast

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, XGate

from qlbm.components.base import LBMPrimitive
from qlbm.components.collisionless.primitives import Comparator, ComparatorMode
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException
from qlbm.tools.utils import flatten


class VolumetricSpaceTimeInitialConditions(LBMPrimitive):
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

    def create_circuit(self) -> QuantumCircuit:
        discretization = self.lattice.properties.get_discretization()
        if discretization == LatticeDiscretization.D1Q2:
            return self.__create_circuit_d1q2()
        if discretization == LatticeDiscretization.D2Q4:
            return self.__create_circuit_d2q4()

        raise CircuitException(f"Reflection Operator unsupported for {discretization}.")

    def __adjusted_comparator_mode(self, bound: bool) -> ComparatorMode:
        return ComparatorMode.LE if (bound) else ComparatorMode.GE

    def __create_circuit_d1q2(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        circuit.h(self.lattice.grid_index())

        for manhattan_distance in range(self.lattice.num_timesteps + 1):
            for neighbor in (
                self.lattice.extreme_point_indices[manhattan_distance]
                if manhattan_distance > 0
                else [self.lattice.properties.origin]
            ):
                periodic_volume_bounds: Tuple[Tuple[int, int], Tuple[bool, bool]] = (
                    self.lattice.comparator_periodic_volume_bounds(
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
                    )[0]
                )

                comparators = [
                    Comparator(
                        self.lattice.properties.get_num_grid_qubits() + 1,
                        periodic_volume_bounds[0][bound],
                        self.__adjusted_comparator_mode(bound),
                        logger=self.logger,
                    ).circuit
                    for bound in [True, False]
                ]

                for comparator_bound, comp in enumerate(comparators):
                    circuit.compose(
                        comp,
                        qubits=self.lattice.grid_index()
                        + [self.lattice.ancilla_comparator_index(0)[comparator_bound]],
                        inplace=True,
                    )

                # If overflow occurs, perform settings sequentially
                if any(periodic_volume_bounds[1]):
                    for bound in [False, True]:
                        circuit.compose(
                            MCMT(
                                XGate(),
                                num_ctrl_qubits=1,
                                num_target_qubits=sum(
                                    self.velocity_profile
                                ),  # The sum is equal to the number of velocities set to true
                            ),
                            qubits=list(
                                [self.lattice.ancilla_comparator_index(0)[0]]
                                + flatten(
                                    [
                                        self.lattice.velocity_index(
                                            neighbor.neighbor_index, c
                                        )
                                        for c, is_velocity_enabled in enumerate(
                                            self.velocity_profile
                                        )
                                        if is_velocity_enabled
                                    ]
                                ),
                            ),
                            inplace=True,
                        )
                else:
                    circuit.compose(
                        MCMT(
                            XGate(),
                            num_ctrl_qubits=2,
                            num_target_qubits=sum(
                                self.velocity_profile
                            ),  # The sum is equal to the number of velocities set to true
                        ),
                        qubits=list(
                            self.lattice.ancilla_comparator_index()
                            + flatten(
                                [
                                    self.lattice.velocity_index(
                                        neighbor.neighbor_index, c
                                    )
                                    for c, is_velocity_enabled in enumerate(
                                        self.velocity_profile
                                    )
                                    if is_velocity_enabled
                                ]
                            ),
                        ),
                        inplace=True,
                    )

                for comparator_bound, comp in enumerate(comparators):
                    circuit.compose(
                        comp,
                        qubits=self.lattice.grid_index()
                        + [self.lattice.ancilla_comparator_index(0)[comparator_bound]],
                        inplace=True,
                    )

        return circuit

    def __create_circuit_d2q4(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        circuit.h(self.lattice.grid_index())

        for manhattan_distance in range(self.lattice.num_timesteps + 1):
            for neighbor_offset in (
                self.lattice.extreme_point_indices[manhattan_distance]
                + flatten(
                    list(
                        self.lattice.intermediate_point_indices[
                            manhattan_distance
                        ].values()
                    )
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
                                + neighbor_offset.coordinates_relative[
                                    dim
                                ]  # Add the offset to the bound in each dimension
                                for i in range(len(dim_bound))
                            )
                            for dim, dim_bound in enumerate(self.cuboid_bounds)
                        ],
                    )
                )

                num_overflows = any(periodic_volume_bounds[0][1]) + any(
                    periodic_volume_bounds[1][1]
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

                for dim in [0, 1]:
                    for bound in [False, True]:
                        circuit.compose(
                            comparators[dim][bound].circuit,
                            qubits=self.lattice.grid_index()
                            + [self.lattice.ancilla_comparator_index(dim)[bound]],
                            inplace=True,
                        )

                if num_overflows == 0:
                    circuit.compose(
                        MCMT(
                            XGate(),
                            num_ctrl_qubits=4,
                            num_target_qubits=sum(
                                self.velocity_profile
                            ),  # The sum is equal to the number of velocities set to true
                        ),
                        qubits=self.lattice.ancilla_comparator_index()
                        + flatten(
                            [
                                self.lattice.velocity_index(0, c)
                                for c, is_velocity_enabled in enumerate(
                                    self.velocity_profile
                                )
                                if is_velocity_enabled
                            ]
                        ),
                        inplace=True,
                    )

                elif num_overflows == 1:
                    dim_overflow = 1 - periodic_volume_bounds[0][1]

                for dim in [0, 1]:
                    for bound in [False, True]:
                        circuit.compose(
                            comparators[dim][bound].circuit,
                            qubits=self.lattice.grid_index()
                            + [self.lattice.ancilla_comparator_index(dim)[bound]],
                            inplace=True,
                        )
        return circuit

    def __str__(self) -> str:
        return f"[Primitive VolumetricSpaceTimeInitialConditions with range={self.cuboid_bounds}, profile = {self.velocity_profile}, and lattice={self.lattice}]"
