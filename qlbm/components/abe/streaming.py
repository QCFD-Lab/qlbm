from logging import Logger, getLogger
from math import pi
from time import perf_counter_ns
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit.synthesis import synth_qft_full as QFT
from typing_extensions import override

from qlbm.components.base import CQLBMOperator, LBMOperator, LBMPrimitive
from qlbm.components.collisionless.streaming import PhaseShift
from qlbm.lattice import CollisionlessLattice
from qlbm.lattice.lattices.abe_lattice import ABELattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools import CircuitException, bit_value
from qlbm.tools.exceptions import LatticeException


class ABEStreamingOperator(LBMOperator):
    """TODO."""

    lattice: ABELattice

    def __init__(
        self,
        lattice: ABELattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        if self.lattice.discretization == LatticeDiscretization.D1Q3:
            return self.__create_circuit_d1q3()

        raise LatticeException("ABE only currently supported in D1Q3")

    def __create_circuit_d1q3(self):
        circuit = self.lattice.circuit.copy()

        circuit.compose(
            QFT(self.lattice.num_grid_qubits),
            qubits=self.lattice.grid_index(),
            inplace=True,
        )

        # 01 streaming in the positive direction
        circuit.x(self.lattice.velocity_index()[0])

        # Controlled Phase Gates for the positive direction
        circuit.compose(
            PhaseShift(
                num_qubits=len(self.lattice.grid_index()),
                positive=True,
                logger=self.logger,
            )
            .circuit.control(2)
            .decompose(),
            qubits=self.lattice.velocity_index() + self.lattice.grid_index(),
            inplace=True,
        )

        # 10 Streaming in the negative direction (and resetting the previous state prep)
        circuit.x(self.lattice.velocity_index())

        circuit.compose(
            PhaseShift(
                num_qubits=len(self.lattice.grid_index()),
                positive=False,  # Negative this time
                logger=self.logger,
            )
            .circuit.control(2)
            .decompose(),
            qubits=self.lattice.velocity_index() + self.lattice.grid_index(),
            inplace=True,
        )

        # Undo the second state prep
        circuit.x(self.lattice.velocity_index()[1])

        # Inverse QFT to return the grid to the computational basis
        circuit.compose(
            QFT(self.lattice.num_grid_qubits, inverse=True),
            qubits=self.lattice.grid_index(),
            inplace=True,
        )

        return circuit
    
    @override
    def __str__(self) -> str:
        return f"[Operator ABEStreaming with lattice {self.lattice}]"
