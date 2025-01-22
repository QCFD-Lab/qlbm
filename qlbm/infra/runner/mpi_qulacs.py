"""MPIQulacs-specific variant of the :class:`CircuitRunner`."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

# import is required for MPI execution even if MPI classes are not used
import numpy as np
from mpi4py import (
    MPI,
)
from qiskit import QuantumCircuit as QiskitQC
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector
from qiskit.result import Counts
from qiskit_aer import AerSimulator, QasmSimulator
from qulacs import QuantumCircuit as QulacsQC
from qulacs import QuantumCircuitSimulator, QuantumState

from qlbm.components.base import LBMPrimitive
from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.result import CollisionlessResult
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import get_circuit_properties, qiskit_to_qulacs


class MPIQulacsRunner:
    """MPIQulacs-specific variant of the :class:`CircuitRunner`."""

    def __init__(
        self,
        lattice: CollisionlessLattice,
        output_directory: str,
        output_file_name: str = "step",
        sampling_backend: AerSimulator = QasmSimulator(),
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__()
        self.lattice = lattice
        self.output_directory = output_directory
        self.output_file_name = output_file_name
        self.logger = logger
        self.sampling_backend = sampling_backend

    def run(
        self,
        initial_condition: Statevector | QiskitQC | QuantumState | QulacsQC,
        algorithm: QulacsQC,
        postprocessing: QulacsQC,
        measurement: QiskitQC | LBMPrimitive,
        num_steps: int,
        num_shots: int,
        snapshot_execution: bool = False,
        statevector_sampling: bool = False,
    ) -> CollisionlessResult:
        """Simualtes the provided algorithm configuration.

        Parameters
        ----------
        initial_condition : Statevector | QiskitQC | QuantumState | QulacsQC
            The state to evolve.
        algorithm : QulacsQC
            The algorithm that evolves the state.
        postprocessing : QulacsQC
            The postprocessing circuit.
        measurement : QiskitQC
            The measurement circuit.
        num_steps : int
            The number of steps to simulate.
        num_shots : int
            The number of shots to perform for each measurement circuit.
        snapshot_execution : bool, optional
            Whether to use snapshots, by default False
        statevector_sampling : bool, optional
            Whether to perfoprm sampling, by default False

        Returns
        -------
        CollisionlessResult
            The result of the simulation.
        """
        measurement = CircuitCompiler("QISKIT", "QISKIT").compile(
            measurement, self.sampling_backend
        )

        return self.run_mpi_qulacs(
            initial_condition,
            algorithm,
            postprocessing,
            measurement,
            num_steps,
            num_shots,
            snapshot_execution,
            statevector_sampling,
        )

    def run_mpi_qulacs(
        self,
        timestep_state: QuantumState | QulacsQC,
        algorithm: QulacsQC,
        postprocessing: QulacsQC,
        measurement: QiskitQC,
        num_steps: int,
        num_shots: int,
        snapshot_execution: bool = False,
        statevector_sampling: bool = False,
    ) -> CollisionlessResult:
        """
        Simualtes the provided algorithm configuration.

        Parameters
        ----------
        timestep_state : QuantumState | QulacsQC
            The state to simulate.
        algorithm : QulacsQC
            The algorithm that evolves the state.
        postprocessing : QulacsQC
            The postprocessing circuit.
        measurement : QiskitQC
            The measurement circuit.
        num_steps : int
            The number of steps to simulate.
        num_shots : int
            The number of shots to perform for each measurement circuit.
        snapshot_execution : bool, optional
            Whether to use snapshots, by default False
        statevector_sampling : bool, optional
            Whether to perfoprm sampling, by default False

        Returns
        -------
        CollisionlessResult
            The result of the simulation.
        """
        simulation_result = CollisionlessResult(
            self.lattice, self.output_directory, self.output_file_name
        )

        simulation_result.visualize_geometry()

        comm = MPI.COMM_WORLD
        mpirank = comm.Get_rank()

        runner_start_time = perf_counter_ns()

        if isinstance(timestep_state, QulacsQC):
            initial_state_circuit = timestep_state.copy()
            timestep_state = QuantumState(
                algorithm.get_qubit_count(), use_multi_cpu=True
            )
            timestep_state.set_zero_state()
            initial_condition_backend = QuantumCircuitSimulator(
                initial_state_circuit, timestep_state
            )
            initial_condition_backend.simulate()
            # initial_state_circuit.update_quantum_state(initial_condition)

        self.logger.info(
            f"Simulation start: MPI QULACS on backend Statevector circuit with num_steps={num_steps}, num_shots={num_shots}, snapshots={snapshot_execution}, and statevector_sampling={statevector_sampling}, mpi_rank={mpirank}, device={timestep_state.get_device_name()}"
        )

        if snapshot_execution:
            for step in range(num_steps):
                step_start_time = perf_counter_ns()
                circuit = QulacsQC(algorithm.get_qubit_count())

                # The first step consists of just initial conditions
                if step > 0:
                    circuit.merge_circuit(algorithm)

                self.logger.info(
                    f"Main circuit for step {step} has properties {get_circuit_properties(circuit)}"
                )

                backend = QuantumCircuitSimulator(circuit, timestep_state)
                backend.simulate()

                if statevector_sampling:
                    measurement_qc = QiskitQC(*(measurement.qregs + measurement.cregs))
                    measurement_qc.append(
                        Initialize(Statevector(timestep_state.get_vector())),
                        list(range(measurement_qc.num_qubits)),
                    )
                    # measurement_qc.compose(postprocessing.copy(), inplace=True)
                    measurement_qc.compose(measurement.copy(), inplace=True)

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
        else:
            for step in range(num_steps):
                step_start_time = perf_counter_ns()
                circuit = QulacsQC(algorithm.get_qubit_count())

                for _ in range(step):
                    circuit.merge_circuit(algorithm)

                circuit.merge_circuit(qiskit_to_qulacs(postprocessing.copy()))

                self.logger.info(
                    f"Main circuit has properties {get_circuit_properties(circuit)}"
                )

                backend = QuantumCircuitSimulator(circuit, timestep_state)
                backend.simulate()

                measurement_qc = QiskitQC(*(measurement.qregs + measurement.cregs))
                measurement_qc.append(
                    Initialize(Statevector(self.get_state_vector(timestep_state))),
                    list(range(measurement_qc.num_qubits)),
                )
                measurement_qc.compose(measurement.copy(), inplace=True)

                counts = (
                    self.sampling_backend.run(measurement_qc, shots=num_shots)
                    .result()
                    .get_counts()
                )
                simulation_result.save_timestep_counts(counts, step)
                self.logger.info(
                    f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
                )

        self.logger.info(
            f"Entire simulation took {perf_counter_ns() - runner_start_time} (ns)"
        )
        return simulation_result

    def get_counts(self, qulacs_samples: List[int], num_bits: int) -> Counts:
        """
        Converts qulacs samples to qiskit ``Counts``.

        Parameters
        ----------
        qulacs_samples : List[int]
            The samples generated through qulacs sampling.
        num_bits : int
            The number of bits each sample contains.

        Returns
        -------
        Counts
            The qiskit ``Counts`` representation of the object.
        """
        return {
            format(measurement, f"0{num_bits}b"): qulacs_samples.count(measurement)
            / len(qulacs_samples)
            for measurement in sorted(set(qulacs_samples))
        }

    def load_state_vector(self, state: QuantumState, vector: np.ndarray):
        """
        Loads the given entire state vector into the given state.

        Args:
            state (qulacs.QuantumState): a quantum state
            vector: a state vector to load
        """
        if state.get_device_name() == "multi-cpu":
            mpicomm = MPI.COMM_WORLD
            mpirank = mpicomm.Get_rank()
            mpisize = mpicomm.Get_size()
            vector_len = len(vector)
            idx_start = vector_len // mpisize * mpirank
            idx_end = vector_len // mpisize * (mpirank + 1)
            state.load(vector[idx_start:idx_end])  # type: ignore
        else:
            state.load(vector)  # type: ignore

    def get_state_vector(self, state: QuantumState) -> np.ndarray:
        """
        Gets the entire state vector from the given state.

        Args:
            state (qulacs.QuantumState): a quantum state
        Return:
            vector: a state vector
        """
        if state.get_device_name() == "multi-cpu":
            mpicomm = MPI.COMM_WORLD
            mpisize = mpicomm.Get_size()
            vec_part = state.get_vector()
            len_part = len(vec_part)
            vector_len = len_part * mpisize
            vector = np.zeros(vector_len, dtype=np.complex128)
            mpicomm.Allgather(
                [vec_part, MPI.DOUBLE_COMPLEX], [vector, MPI.DOUBLE_COMPLEX]
            )
            return vector
        else:
            return state.get_vector()
