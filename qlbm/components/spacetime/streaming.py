from logging import Logger, getLogger
from time import perf_counter_ns

from qlbm.components.base import SpaceTimeOperator
from qlbm.lattice import SpaceTimeLattice
from qlbm.tools.exceptions import CircuitException


class SpaceTimeStreamingOperator(SpaceTimeOperator):
    def __init__(
        self,
        lattice: SpaceTimeLattice,
        timestep: int,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice
        self.timestep = timestep

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        if self.timestep == 1:
            # !!! TODO: Generalize
            for point_class, extreme_point in enumerate(
                self.lattice.extreme_point_indices[self.timestep]
            ):
                velocity_to_swap = extreme_point.velocity_index_to_swap(point_class, 1)
                circuit.swap(
                    self.lattice.velocity_index(
                        extreme_point.neighbor_index,
                        velocity_to_swap,
                    ),
                    self.lattice.velocity_index(0, velocity_to_swap),
                )
        else:
            raise CircuitException("Not implemented.")

        return circuit

    def __str__(self) -> str:
        # TODO: Implement
        return "Space Time Streaming Operator"
