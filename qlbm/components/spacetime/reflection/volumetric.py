"""Reflection operator for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime` that swaps particles for simultaneously for fixed volumes."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple, cast

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import SpaceTimeOperator
from qlbm.components.collisionless.primitives import Comparator, ComparatorMode
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.exceptions import CircuitException
from qlbm.tools.utils import flatten


class VolumetricSpaceTimeReflectionOperator(SpaceTimeOperator):
    """
    Prepares the initial state for the :class:`.SpaceTimeQLBM` for a volumetric region.

    Work in progress.
    """

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        timestep: int,
        blocks: List[Block],
        filter_inside_blocks: bool = True,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice, logger)
        self.timestep = timestep

        if timestep < 1 or timestep > lattice.num_timesteps:
            raise CircuitException(
                f"Invalid time step {timestep}, select a value between 1 and {lattice.num_timesteps}"
            )

        self.blocks = blocks
        self.filter_inside_blocks = filter_inside_blocks

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()

        self.all_cuboid_bounds: List[List[Tuple[int, int]]] = [
            block.bounds for block in blocks
        ]
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        circuit.h(self.lattice.grid_index())

        # For all blocks
        for block in self.blocks:
            cuboid_bounds = block.bounds
            # For all "layers" of gridpoints
            for manhattan_distance in range(self.lattice.num_timesteps + 1):
                # For all relative position of the point in the stencil, at this layer
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
                                for dim, dim_bound in enumerate(cuboid_bounds)
                            ],
                        )
                    )

                    # Assemble the comparators only once
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

                    # Place the comparators on the appropriate ancillae
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
                        # circuit.compose(
                        #     MCSwap(
                        #         self.lattice,
                        #         control_qubit_sequence,
                        #         cast(
                        #             Tuple[int, int],
                        #             tuple(
                        #                 [
                        #                     self.lattice.velocity_index(
                        #                         neighbor_velocity_pair[0],
                        #                         neighbor_velocity_pair[1],
                        #                     )[0]
                        #                     for neighbor_velocity_pair in reflection_data.neighbor_velocity_pairs
                        #                 ]
                        #             ),
                        #         ),
                        #         self.logger,
                        #     ).circuit,
                        #     inplace=True,
                        # )
                        pass

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
