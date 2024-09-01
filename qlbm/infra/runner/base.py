from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, cast

from qiskit import QuantumCircuit as QiskitQC
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlbm.infra.reinitialize import (
    CollisionlessReinitializer,
    Reinitializer,
    SpaceTimeReinitializer,
)
from qlbm.infra.result import CollisionlessResult, QBMResult, SpaceTimeResult
from qlbm.lattice import CollisionlessLattice, Lattice, SpaceTimeLattice
from qlbm.tools.exceptions import CircuitException, ResultsException

from .simulation_config import SimulationConfig


class CircuitRunner(ABC):
    available_devices: List[str] = AerSimulator().available_devices()  # type: ignore

    def __init__(
        self,
        config: SimulationConfig,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
        device: str = "CPU",  # ! TODO reimplement
    ) -> None:
        super().__init__()

        if device not in self.available_devices:
            raise CircuitException(
                f"Unsupported Qiskit StatevectorSimulator device: {device}. Supported devices are: {self.available_devices}"
            )
        self.config = config
        self.lattice = lattice
        self.logger = logger
        self.device = device
        self.reinitializer = self.new_reinitializer()

    @abstractmethod
    def run(
        self,
        num_steps: int,
        num_shots: int,
        output_directory: str,
        output_file_name: str = "step",
        statevector_snapshots: bool = False,
    ) -> QBMResult:
        pass

    def new_result(self, output_directory: str, output_file_name: str) -> QBMResult:
        if isinstance(self.lattice, CollisionlessLattice):
            return CollisionlessResult(
                cast(CollisionlessLattice, self.lattice),
                output_directory,
                output_file_name,
            )
        elif isinstance(self.lattice, SpaceTimeLattice):
            return SpaceTimeResult(
                cast(SpaceTimeLattice, self.lattice), output_directory, output_file_name
            )
        else:
            raise ResultsException(f"Unsupported lattice: {self.lattice}.")

    def new_reinitializer(self) -> Reinitializer:
        if isinstance(self.lattice, CollisionlessLattice):
            return CollisionlessReinitializer(
                cast(CollisionlessLattice, self.lattice),
                self.config.get_execution_compiler(),
                self.logger,
            )
        elif isinstance(self.lattice, SpaceTimeLattice):
            return SpaceTimeReinitializer(
                cast(SpaceTimeLattice, self.lattice),
                self.config.get_execution_compiler(),
            )
        else:
            raise ResultsException(f"Unsupported lattice: {self.lattice}.")

    def statevector_to_circuit(self, statevector: Statevector) -> QiskitQC:
        circuit = self.lattice.circuit.copy()
        circuit.append(Initialize(statevector), circuit.qubits)
        return circuit
