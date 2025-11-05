from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import ClassicalRegister, QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.ab_lattice import ABLattice


class ABGridMeasurement(LBMPrimitive):
    """TODO."""

    def __init__(
        self,
        lattice: ABLattice,
        measure_velocity_qubits: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
        self.measure_velocity_qubits = measure_velocity_qubits

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        circuit.add_register(
            ClassicalRegister(
                self.lattice.num_grid_qubits
                + (
                    self.lattice.num_velocity_qubits
                    if self.measure_velocity_qubits
                    else 0
                )
            )
        )

        circuit.measure(
            self.lattice.grid_index()
            + (self.lattice.velocity_index() if self.measure_velocity_qubits else []),
            list(
                range(
                    self.lattice.num_grid_qubits
                    + (
                        self.lattice.num_velocity_qubits
                        if self.measure_velocity_qubits
                        else 0
                    )
                )
            ),
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABEGridMeasurement with lattice {self.lattice}]"
