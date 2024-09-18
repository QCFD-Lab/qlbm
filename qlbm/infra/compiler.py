from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from pytket.backends import Backend as TketBackend
from pytket.circuit import Circuit as PytketQC
from pytket.extensions.qiskit import AerBackend as TketQiskitBackend
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.extensions.qulacs import QulacsBackend, tk_to_qulacs
from qiskit import QuantumCircuit as QiskitQC
from qiskit.compiler import transpile
from qiskit_aer.backends.aerbackend import AerBackend
from qiskit_qulacs import QulacsProvider
from qulacs import QuantumCircuit as QulacsQC

from qlbm.components.base import QuantumComponent
from qlbm.tools.exceptions import CompilerException
from qlbm.tools.utils import get_circuit_properties


class CircuitCompiler:
    compiler_type: str
    compiler_target: str
    supported_compiler_types: List[str] = ["QISKIT", "TKET"]
    supported_compiler_targets: List[str] = ["QISKIT", "QULACS"]

    def __init__(
        self,
        compiler_type: str,
        compiler_target: str,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        """Multi-target compiler of Qiskit circuits into the Qulacs gate set.

        Args:
            compiler_type (CircuitCompilerType): The compiler to use (either qiskit or pytket).
            compiler_target (CircuitCompilerTarget): The target to compile to (either qiskit or qulacs).
            optimization_level (int): The level of optimization to apply to the circuit.
        """
        super().__init__()

        if compiler_type not in self.supported_compiler_types:
            raise CompilerException(
                f"Unsupported compiler type: {compiler_type}. Supported compiler types are: {self.supported_compiler_types}"
            )

        if compiler_target not in self.supported_compiler_targets:
            raise CompilerException(
                f"Unsupported compiler target: {compiler_type}. Supported compiler targets are: {self.supported_compiler_targets}"
            )

        self.compiler_type = compiler_type
        self.compiler_target = compiler_target
        self.logger = logger

        logger.info(str(self))

    def compile(
        self,
        compile_object: QiskitQC | QuantumComponent,
        backend: AerBackend | None,
        optimization_level: int = 0,
    ) -> QulacsQC | QiskitQC:
        if optimization_level not in [0, 1, 2]:
            raise CompilerException(
                f"Unsupported optimization level {optimization_level}. Supported optimization levels are 0, 1, and 2."
            )

        if isinstance(compile_object, QiskitQC):
            qiskit_circuit = compile_object
            self.logger.info(
                f"{str(self)}: Compiling Qiskit circuit with properties {get_circuit_properties(qiskit_circuit)} on {str(backend)} with opt={optimization_level}"
            )
        else:
            qiskit_circuit = compile_object.circuit
            self.logger.info(
                f"{str(self)}: Compiling {str(compile_object)} circuit with properties {get_circuit_properties(qiskit_circuit)} on {str(backend)} with opt={optimization_level}"
            )

        compiler_start_time = perf_counter_ns()

        if self.compiler_target == "QULACS":
            if backend is not None:
                self.logger.warn(
                    "Provided backend is ignored. The qiskit-qulacs backend is always used when compiling to Qulacs from Qiskit."
                )
            backend = QulacsProvider().get_backend("qulacs_simulator")

        if self.compiler_target == "QISKIT":
            if not isinstance(backend, AerBackend):
                raise CompilerException(
                    f"Cannot use compiler with backend {backend}. Only Qiskit AerBackend objects are supported. When Using Tket, the Qiskit backend is automatically converted."
                )

        compiled_circuit = None

        if self.compiler_type == "QISKIT":
            # Compile the circuit by transpiling in qiskit
            compiled_circuit = self.__compile_qiskit(
                qiskit_circuit, backend, optimization_level
            )

            if self.compiler_target == "QULACS":
                compiled_circuit = tk_to_qulacs(qiskit_to_tk(compiled_circuit))

        elif self.compiler_type == "TKET":
            # Decompose the Qiskit circuit and convert it to tket
            # Decomposition is necessary because tket cannot parse
            # Some custom gates
            tk_circuit = qiskit_to_tk(qiskit_circuit.decompose())

            tket_backend = (
                TketQiskitBackend(simulation_method=backend.options.method)  # type: ignore
                if self.compiler_target == "QISKIT"
                else QulacsBackend()
            )

            # Compile the tk circuit using the Qulacs backend
            compiled_circuit = self.__compile_pytket(
                tk_circuit, tket_backend, optimization_level
            )

            compiled_circuit = (
                tk_to_qiskit(compiled_circuit)
                if self.compiler_target == "QISKIT"
                else tk_to_qulacs(compiled_circuit)
            )
        else:
            raise CompilerException(f"Unsupported compiler type: {self.compiler_type}.")

        self.logger.info(
            f"Compilation took {perf_counter_ns() - compiler_start_time} (ns)"
        )
        self.logger.info(
            f"Compiled circuit has properties {get_circuit_properties(compiled_circuit)}"
        )

        return compiled_circuit

    def __str__(self) -> str:
        return (
            f"[Compiler using {self.compiler_type} with target {self.compiler_target}]"
        )

    def __compile_pytket(
        self, tk_circuit: PytketQC, backend: TketBackend, optimization_level: int
    ) -> PytketQC:
        # Compile using pytket
        return backend.get_compiled_circuit(
            tk_circuit, optimisation_level=optimization_level
        )

    def __compile_qiskit(
        self, qiskit_circuit: QiskitQC, backend: AerBackend, optimization_level: int
    ) -> QiskitQC:
        # Transpile the Qiskit circuit to the target backend
        return transpile(
            qiskit_circuit,
            backend=backend,
            optimization_level=optimization_level,
        )