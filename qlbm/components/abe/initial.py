from enum import Enum
from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.synthesis import synth_qft_full as QFT
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.components.collisionless.streaming import SpeedSensitivePhaseShift
from qlbm.components.common.primitives import TruncatedQFT
from qlbm.lattice import CollisionlessLattice
from qlbm.lattice.geometry.encodings.collisionless import ReflectionResetEdge
from qlbm.lattice.lattices.abe_lattice import ABELattice
from qlbm.tools import flatten


class ABEInitialConditions(LBMPrimitive):
    """TODO."""

    def __init__(
        self,
        lattice: ABELattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        circuit.compose(
            TruncatedQFT(
                self.lattice.num_velocity_qubits,
                self.lattice.num_velocities_per_point,
                self.logger,
            ).circuit,
            qubits=self.lattice.velocity_index(),
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABEInitialConditions with lattice {self.lattice}]"
