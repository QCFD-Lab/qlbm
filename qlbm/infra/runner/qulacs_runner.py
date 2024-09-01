from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import QuantumCircuit as QiskitQC
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector
from qiskit.result import Counts
from qulacs import QuantumCircuit as QulacsQC
from qulacs import QuantumCircuitSimulator, QuantumState

from qlbm.infra.result import QBMResult
from qlbm.lattice import Lattice
from qlbm.tools.exceptions import ExecutionException
from qlbm.tools.utils import get_circuit_properties

from .base import CircuitRunner
from .simulation_config import SimulationConfig


class QulacsRunner(CircuitRunner):
    def __init__(
        self,
        config: SimulationConfig,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
        device: str = "CPU",  # ! TODO reimplement
    ) -> None:
        super().__init__(config, lattice, logger, device)

        self.execution_backend = self.config.execution_backend
        self.sampling_backend = self.config.sampling_backend

    def run(
        self,
        num_steps: int,
        num_shots: int,
        output_directory: str,
        output_file_name: str = "step",
        statevector_snapshots: bool = False,
    ) -> QBMResult:
        if (self.sampling_backend is None) and self.config.statevector_sampling:
            raise ExecutionException(
                "Cannot perform statevector sampling without a dedicated backend."
            )

        simulation_result = self.new_result(output_directory, output_file_name)
        simulation_result.visualize_geometry()

        self.logger.info(
            f"Simulation start: QULACS with config {self.config} with num_steps={num_steps}, num_shots={num_shots}, snapshots={statevector_snapshots}"  # type: ignore
        )

        runner_start_time = perf_counter_ns()

        if isinstance(self.config.initial_conditions, QulacsQC):
            initial_state_circuit = self.config.initial_conditions.copy()
            initial_condition_statevector = QuantumState(
                self.config.algorithm.get_qubit_count()  # type: ignore
            )
            initial_condition_statevector.set_zero_state()
            initial_condition_backend = QuantumCircuitSimulator(
                initial_state_circuit, initial_condition_statevector
            )
            initial_condition_backend.simulate()

        simulation_result = (
            self._run_snapshot_time_loop(
                num_steps,
                num_shots,
                initial_condition_statevector,
                simulation_result,
            )
            if statevector_snapshots
            else self._run_time_loop(
                num_steps,
                num_shots,
                initial_condition_statevector,
                simulation_result,
            )
        )

        self.logger.info(
            f"Entire simulation took {perf_counter_ns() - runner_start_time} (ns)"
        )

        return simulation_result

    def _run_snapshot_time_loop(
        self,
        num_steps: int,
        num_shots: int,
        initial_conditions: QuantumState,
        simulation_result: QBMResult,
    ) -> QBMResult:
        # circuit = QulacsQC(self.config.algorithm.get_qubit_count())  # type: ignore
        circuit = self.config.algorithm.copy() # type: ignore

        for step in range(num_steps + 1):
            step_start_time = perf_counter_ns()

            self.logger.info(
                f"Main circuit for step {step} has properties {get_circuit_properties(circuit)}"
            )

            backend = QuantumCircuitSimulator(circuit, initial_conditions)
            backend.simulate()

            if self.config.statevector_sampling:
                measurement_qc = QiskitQC(
                    *(self.config.measurement.qregs + self.config.measurement.cregs)  # type: ignore
                )
                measurement_qc.append(
                    Initialize(Statevector(initial_conditions.get_vector())),
                    list(range(measurement_qc.num_qubits)),
                )
                measurement_qc.compose(self.config.postprocessing.copy(), inplace=True)  # type: ignore
                measurement_qc.compose(self.config.measurement.copy(), inplace=True)  # type: ignore

                measurement_result = self.sampling_backend.run(
                    measurement_qc, shots=num_shots
                ).result()

                simulation_result.save_timestep_counts(
                    measurement_result.get_counts(), step
                )
                self.logger.info(
                    f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
                )
            else:
                self.logger.error(
                    "Qulacs simulation does not currently support statevector_sampling=False due to performance reasons."
                )
                raise ExecutionException("Combination unsupported.")

            if self.reinitializer.requires_statevector():
                # ! TODO
                # We assume the Qulacs quantum state object in initial conditions
                # Is exactly the one required to continue the simulation
                pass
            else:
                # Reset the quantum state
                # The result produced by the reinitializer is
                # A new quantum circuit that starts in the |0> state
                initial_conditions.set_zero_state()

                circuit = QulacsQC(self.config.algorithm.get_qubit_count())  # type: ignore

                reinitialization_circuit = self.reinitializer.reinitialize(
                    Statevector([0]),
                    measurement_result.get_counts(),
                    self.config.get_execution_compiler(),
                    self.config.optimization_level,
                )

                circuit.merge_circuit(reinitialization_circuit)
                circuit.merge_circuit(self.config.algorithm)  # type: ignore
            self.logger.info(
                f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
            )

        return simulation_result

    def _run_time_loop(
        self,
        num_steps: int,
        num_shots: int,
        initial_conditions: Statevector,
        simulation_result: QBMResult,
    ) -> QBMResult:
        for step in range(num_steps + 1):
            step_start_time = perf_counter_ns()
            circuit = QulacsQC(self.config.algorithm.get_qubit_count())  # type: ignore

            for _ in range(step):
                circuit.merge_circuit(self.config.algorithm.copy())  # type: ignore

            circuit.merge_circuit(self.config.postprocessing.copy())  # type: ignore

            self.logger.info(
                f"Main circuit has properties {get_circuit_properties(circuit)}"
            )

            backend = QuantumCircuitSimulator(circuit, initial_conditions)
            backend.simulate()
            if self.config.statevector_sampling:
                measurement_qc = QiskitQC(
                    *(self.config.measurement.qregs + self.config.measurement.cregs)  # type: ignore
                )
                measurement_qc.append(
                    Initialize(Statevector(initial_conditions.get_vector())),
                    list(range(measurement_qc.num_qubits)),
                )
                measurement_qc.compose(self.config.measurement.copy(), inplace=True)  # type: ignore

                counts = (
                    self.sampling_backend.run(measurement_qc, shots=num_shots)
                    .result()
                    .get_counts()
                )
                simulation_result.save_timestep_counts(counts, step)
                self.logger.info(
                    f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
                )
            else:
                self.logger.error(
                    "Qulacs simulation does not currently support statevector_sampling=False due to performance reasons."
                )
                raise ExecutionException("Combination unsupported.")
        return simulation_result

    def get_counts(self, qulacs_samples: List[int], num_bits: int) -> Counts:
        return {
            format(measurement, f"0{num_bits}b"): qulacs_samples.count(measurement)
            / len(qulacs_samples)
            for measurement in sorted(set(qulacs_samples))
        }

    def __str__(self) -> str:
        return f"[QulacsRunner on device {self.device}] and sampling backend {self.sampling_backend}"
