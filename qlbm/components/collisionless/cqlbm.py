from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit

from qlbm.components.base import LBMAlgorithm
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import get_time_series

from .bounceback_reflection import BounceBackReflectionOperator
from .primitives import StreamingAncillaPreparation
from .specular_reflection import SpecularReflectionOperator
from .streaming import CollisionlessStreamingOperator


class CQLBM(LBMAlgorithm):
    def __init__(
        self,
        lattice: CollisionlessLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice: CollisionlessLattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self):
        # Assumes equal velocities in all dimensions
        # ! TODO adapt to DnQm discretization
        time_series = get_time_series(2 ** self.lattice.num_velocities[0].bit_length())
        circuit = QuantumCircuit(
            *self.lattice.registers,
        )

        for velocities_to_increment in time_series:
            circuit.compose(
                CollisionlessStreamingOperator(
                    self.lattice,
                    velocities_to_increment,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )

            circuit.compose(
                SpecularReflectionOperator(
                    self.lattice,
                    self.lattice.blocks["specular"],
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )

            circuit.compose(
                BounceBackReflectionOperator(
                    self.lattice,
                    self.lattice.blocks["bounceback"],
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )

            for dim in range(self.lattice.num_dimensions):
                circuit.compose(
                    StreamingAncillaPreparation(
                        self.lattice,
                        velocities_to_increment,
                        dim,
                        logger=self.logger,
                    ).circuit,
                    inplace=True,
                )
        return circuit

    def __str__(self) -> str:
        return f"[Algorithm CQLBM with lattice {self.lattice}]"
