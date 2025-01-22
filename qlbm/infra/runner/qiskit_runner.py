"""Qiskit-specific implementation of the :class:`CircuitRunner`."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit as QiskitQC
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector
from typing_extensions import override

from qlbm.infra.result import QBMResult
from qlbm.lattice import Lattice
from qlbm.tools.exceptions import ExecutionException
from qlbm.tools.utils import get_circuit_properties

from .base import CircuitRunner
from .simulation_config import SimulationConfig


class QiskitRunner(CircuitRunner):
    """
    Qiskit-specific implementation of the :class:`CircuitRunner`.

    A provided simulation configuration is compatible with this runner if the following conditions are met:

    #. The ``initial_conditions`` is either a ``qlbm`` :class:`.QuantumComponent`, a Qiskit ``Statevector`` or a Qiskit``QuantumCircuit``.
    #. The ``execution_backend`` is a Qiskit ``AerBackend``.
    #. If enabled, the ``sampling_backend`` is a Qiskit ``AerBackend``.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`config`              The :class:`.SimulationConfig` containing the simulation information.
    :attr:`lattice`             The :class:`.Lattice` of the simulated system.
    :attr:`reinitializer`       The :class:`.Reinitializer` that performs the transition between time steps.
    :attr:`device`              Currently ignored.
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``.
    =========================== ======================================================================

    Any simulator with a Qiskit ``AerBackend`` interface
    can be used as either ``execution_backend`` or ``sampling_backend``
    in the config interface.
    """

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

    @override
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
            f"Simulation start: QISKIT with config {self.config} with num_steps={num_steps}, num_shots={num_shots}, snapshots={statevector_snapshots}"  # type: ignore
        )

        runner_start_time = perf_counter_ns()

        initial_conditions = (
            self.statevector_to_circuit(self.config.initial_conditions)
            if isinstance(self.config.initial_conditions, Statevector)
            else self.config.initial_conditions
        )

        simulation_result = (
            self._run_snapshot_time_loop(
                num_steps,
                num_shots,
                initial_conditions,
                simulation_result,
            )
            if statevector_snapshots
            else self._run_time_loop(
                num_steps,
                num_shots,
                initial_conditions,
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
        initial_conditions: QiskitQC,
        simulation_result: QBMResult,
    ) -> QBMResult:
        for step in range(num_steps + 1):
            step_start_time = perf_counter_ns()
            circuit = QiskitQC(
                *(self.config.measurement.qregs + self.config.measurement.cregs)  # type: ignore
            )
            circuit.compose(
                initial_conditions,
                inplace=True,
                qubits=range(circuit.num_qubits),
            )
            # The first step consists of just initial conditions
            if step > 0:
                circuit.compose(self.config.algorithm, inplace=True)

            if (
                self.reinitializer.requires_statevector()
                or self.config.statevector_sampling
            ):
                circuit.save_statevector(label="step")

            self.logger.info(
                f"Main circuit for step {step} has properties {get_circuit_properties(circuit)}"
            )

            if self.config.statevector_sampling:
                qiskit_execution_result = self.execution_backend.run(  # type: ignore
                    circuit,
                ).result()
                measurement_qc = QiskitQC(
                    *(self.config.measurement.qregs + self.config.measurement.cregs)  # type: ignore
                )
                measurement_qc.compose(
                    self.statevector_to_circuit(
                        qiskit_execution_result.data(0)["step"]
                    ),
                    list(range(self.config.measurement.num_qubits)),  # type: ignore
                    inplace=True,
                )
                measurement_qc.compose(self.config.postprocessing.copy(), inplace=True)  # type: ignore
                measurement_qc.compose(self.config.measurement.copy(), inplace=True)  # type: ignore

                qiskit_measurement_result = self.sampling_backend.run(
                    measurement_qc, shots=num_shots
                ).result()

                simulation_result.save_timestep_counts(
                    qiskit_measurement_result.get_counts(), step
                )
            else:
                circuit.compose(self.config.postprocessing.copy(), inplace=True)  # type: ignore
                circuit.compose(self.config.measurement.copy(), inplace=True)  # type: ignore
                qiskit_execution_result = self.execution_backend.run(  # type: ignore
                    circuit, shots=num_shots
                ).result()

                simulation_result.save_timestep_counts(
                    qiskit_execution_result.get_counts(), step
                )

            # Update the initial conditions for the next step
            initial_conditions = self.reinitializer.reinitialize(
                statevector=qiskit_execution_result.data(0)["step"]
                if self.reinitializer.requires_statevector()
                else Statevector([0]),
                counts=qiskit_measurement_result.get_counts()
                if self.config.statevector_sampling
                else qiskit_execution_result.get_counts(),
                backend=self.config.execution_backend,
                optimization_level=self.config.optimization_level,
            )
            self.logger.info(
                f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
            )

        return simulation_result

    def _run_time_loop(
        self,
        num_steps: int,
        num_shots: int,
        initial_conditions: QiskitQC,
        simulation_result: QBMResult,
    ) -> QBMResult:
        for step in range(num_steps + 1):
            step_start_time = perf_counter_ns()
            self.logger.info(f"Simulating {step} steps")
            circuit = QiskitQC(self.config.algorithm.num_qubits)  # type: ignore
            circuit.compose(
                initial_conditions,
                inplace=True,
                qubits=range(circuit.num_qubits),
            )
            for _ in range(step):
                circuit.compose(self.config.algorithm.copy(), inplace=True)  # type: ignore

            circuit.compose(self.config.postprocessing.copy(), inplace=True)  # type: ignore

            self.logger.info(
                f"Main circuit for step {step} has properties {get_circuit_properties(circuit)}"
            )

            if self.config.statevector_sampling:
                result = self.execution_backend.run(circuit, shots=num_shots).result()  # type: ignore
                statevector = result.get_statevector()
                measurement_qc = QiskitQC(
                    *(self.config.measurement.qregs + self.config.measurement.cregs)  # type: ignore
                )
                measurement_qc.compose(
                    Initialize(statevector),
                    list(range(measurement_qc.num_qubits)),
                    inplace=True,
                )
                measurement_qc.compose(self.config.postprocessing.copy(), inplace=True)  # type: ignore
                measurement_qc.compose(self.config.measurement.copy(), inplace=True)  # type: ignore

                counts = (
                    self.sampling_backend.run(measurement_qc, shots=num_shots)
                    .result()
                    .get_counts()
                )
                simulation_result.save_timestep_counts(counts, step)
            else:
                circuit.compose(self.config.measurement.copy(), inplace=True)  # type: ignore
                counts = (
                    self.execution_backend.run(circuit, shots=num_shots)  # type: ignore
                    .result()
                    .get_counts()
                )
                simulation_result.save_timestep_counts(counts, step)
            self.logger.info(
                f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
            )

        return simulation_result
