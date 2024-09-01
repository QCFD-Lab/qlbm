from logging import Logger, getLogger
from typing import Any, List

from qiskit import QuantumCircuit as QiskitQC
from qiskit.quantum_info import Statevector
from qiskit_aer.backends.aerbackend import AerBackend
from qulacs import QuantumCircuit as QulacsQC
from qulacs import QuantumState

from qlbm.components.base import QuantumComponent
from qlbm.infra.compiler import CircuitCompiler
from qlbm.tools.exceptions import ExecutionException


class SimulationConfig:
    initial_conditions: (
        Statevector | QiskitQC | QuantumState | QulacsQC | QuantumComponent
    )
    algorithm: QiskitQC | QulacsQC | QuantumComponent
    postprocessing: QiskitQC | QulacsQC | QuantumComponent
    measurement: QiskitQC | QuantumComponent
    execution_backend: AerBackend | None
    sampling_backend: AerBackend
    logger: Logger

    QISKIT = "QISKIT"
    QULACS = "QULACS"

    initial_conditions_types = {
        QISKIT: [Statevector, QiskitQC, QuantumComponent],
        QULACS: [QuantumState, QulacsQC, QuantumComponent],
    }

    algorithm_types = {
        QISKIT: [QiskitQC, QuantumComponent],
        QULACS: [QulacsQC, QuantumComponent],
    }

    measurement_types = {
        QISKIT: [QiskitQC, QuantumComponent],
        QULACS: [QiskitQC, QuantumComponent],
    }

    execution_backend_types = {QISKIT: [AerBackend], QULACS: [type(None)]}

    sampling_backend_types = {
        QISKIT: [AerBackend],
        QULACS: [AerBackend],
    }

    def __init__(
        self,
        initial_conditions: Statevector
        | QiskitQC
        | QuantumState
        | QulacsQC
        | QuantumComponent,
        algorithm: QiskitQC | QulacsQC | QuantumComponent,
        postprocessing: QiskitQC | QulacsQC | QuantumComponent,
        measurement: QiskitQC | QuantumComponent,
        target_platform: str,
        compiler_platform: str,
        optimization_level: int,
        statevector_sampling: bool,
        execution_backend: AerBackend | None,
        sampling_backend: AerBackend | None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        # Circuits
        self.initial_conditions = initial_conditions
        self.algorithm = algorithm
        self.postprocessing = postprocessing
        self.measurement = measurement

        # Simulation details
        self.target_platform = target_platform
        self.compiler_platform = compiler_platform
        self.optimization_level = optimization_level
        self.statevector_sampling = statevector_sampling

        # Backends
        self.execution_backend = execution_backend
        self.sampling_backend = sampling_backend
        self.logger = logger

    def validate(
        self,
    ) -> None:
        self.__is_appropriate_target_platform(self.target_platform)
        self.__is_compatible_type(
            self.initial_conditions,
            self.initial_conditions_types[self.target_platform],
            "Initial conditions",
            self.target_platform,
        )

        self.__is_compatible_type(
            self.algorithm,
            self.algorithm_types[self.target_platform],
            "Algorithm",
            self.target_platform,
        )

        self.__is_compatible_type(
            self.postprocessing,
            self.algorithm_types[self.target_platform],
            "Postprocessing",
            self.target_platform,
        )

        self.__is_compatible_type(
            self.measurement,
            self.measurement_types[self.target_platform],
            "Measurement",
            self.target_platform,
        )

        if self.statevector_sampling:
            self.__is_compatible_type(
                self.sampling_backend,
                self.sampling_backend_types[self.target_platform],
                "Sampling Backend",
                self.target_platform,
            )

        self.__is_compatible_type(
            self.execution_backend,
            self.execution_backend_types[self.target_platform],
            "Execution Backend",
            self.target_platform,
        )

    def __is_compatible_type(
        self,
        object_to_validate: Any,
        accepted_types: List[Any],
        object_name: str,
        platform: str,
    ) -> None:
        if not any(isinstance(object_to_validate, t) for t in accepted_types):
            raise ExecutionException(
                f"{object_name} object of type {type(object_to_validate)} is not in supported types {accepted_types} for platform {platform}",
            )

    def __is_appropriate_target_platform(self, platform: str) -> None:
        if platform not in [self.QISKIT, self.QULACS]:
            raise ExecutionException(
                f"Platform {platform} unsupported. Supported platforms are: {[self.QISKIT, self.QULACS]}"
            )

    def prepare_for_simulation(
        self,
    ) -> None:
        self.__perpare_circuits(
            self.get_execution_compiler(),
            self.get_sampling_compiler(),
        )

    def get_execution_compiler(self) -> CircuitCompiler:
        self.__is_appropriate_target_platform(self.target_platform)

        return CircuitCompiler(
            self.compiler_platform, self.target_platform, self.logger
        )

    def get_sampling_compiler(self) -> CircuitCompiler:
        return (
            CircuitCompiler(self.compiler_platform, "QISKIT", self.logger)
            if self.statevector_sampling
            else self.get_execution_compiler()
        )

    def __perpare_circuits(
        self, execution_compiler: CircuitCompiler, sampling_compiler: CircuitCompiler
    ) -> None:
        if not isinstance(self.initial_conditions, Statevector):
            self.initial_conditions = execution_compiler.compile(
                self.initial_conditions, self.execution_backend, self.optimization_level
            )

        self.algorithm = execution_compiler.compile(
            self.algorithm, self.execution_backend, self.optimization_level
        )
        self.postprocessing = sampling_compiler.compile(
            self.postprocessing,
            self.sampling_backend
            if self.statevector_sampling
            else self.execution_backend,
            self.optimization_level,
        )
        self.measurement = sampling_compiler.compile(
            self.measurement,
            self.sampling_backend
            if self.statevector_sampling
            else self.execution_backend,
            self.optimization_level,
        )
