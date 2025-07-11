"""Streaming operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple

from typing_extensions import override

from qlbm.components.base import LQLGAOperator
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice


class LQLGAStreamingOperator(LQLGAOperator):
    def __init__(
        self,
        lattice: LQLGALattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        circuit = self.lattice.circuit.copy()
        # ! TODO Generalize in 2 and 3D
        num_gps = self.lattice.num_gridpoints[0] + 1

        for direction, velocity_qubit_of_line in enumerate(
            self.lattice.get_velocity_qubits_of_line(0)
        ):
            gridpoints_to_swap = self.logarithmic_depth_streaming_line_swaps(
                num_gps, negative_direction=bool(direction)
            )
            for layer in gridpoints_to_swap:
                for i, j in layer:
                    circuit.swap(
                        self.lattice.velocity_index_flat(i, velocity_qubit_of_line),
                        self.lattice.velocity_index_flat(j, velocity_qubit_of_line),
                    )

        return circuit

    def logarithmic_depth_streaming_line_swaps(
        self, num_gridpoints: int, negative_direction: bool
    ) -> List[List[Tuple[int, int]]]:
        if num_gridpoints < 2:
            return []

        layers: List[List[Tuple[int, int]]] = []
        stride = 1

        while stride < num_gridpoints:
            layer: List[Tuple[int, int]] = []
            for i in range(0, num_gridpoints, 2 * stride):
                if i + stride < num_gridpoints:
                    layer.append(
                        (i, i + stride)
                        if not negative_direction
                        else (num_gridpoints - 1 - i, num_gridpoints - 1 - i - stride)
                    )
            layers.append(layer)
            stride *= 2

        return layers

    @override
    def __str__(self):
        return f"[LQLGAStreamingOperator on lattice={self.lattice}]"
