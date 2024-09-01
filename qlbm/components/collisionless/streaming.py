from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import QuantumCircuit

from qlbm.components.base import CQLBMOperator
from qlbm.lattice import CollisionlessLattice

from .primitives import ControlledIncrementer, StreamingAncillaPreparation


class CollisionlessStreamingOperator(CQLBMOperator):
    circuit: QuantumCircuit

    def __init__(
        self,
        lattice: CollisionlessLattice,
        velocities: List[int],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.velocities_to_stream = velocities

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        for dim in range(self.lattice.num_dimensions):
            circuit.compose(
                StreamingAncillaPreparation(
                    self.lattice,
                    self.velocities_to_stream,
                    dim,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )

        circuit.compose(
            ControlledIncrementer(
                self.lattice,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )

        return circuit

    def __str__(self) -> str:
        return (
            f"[Operator StreamingOperator for velocities {self.velocities_to_stream}]"
        )
