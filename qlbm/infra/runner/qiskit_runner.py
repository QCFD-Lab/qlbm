"""Qiskit-specific implementation of the :class:`CircuitRunner`."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit as QiskitQC
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer.library import SetStatevector  # type: ignore[import-untyped]
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
    #. The ``measurement`` circuit contains only ``measure`` (and ``barrier``) instructions when ``statevector_sampling`` is enabled.

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
    can be used as ``execution_backend`` in the config interface.
    With ``statevector_sampling`` enabled, counts are sampled from the saved
    statevector directly (postprocessing applied on the ``execution_backend``
    if there is any), so the ``sampling_backend`` is no longer used.
    """

    def __init__(
        self,
        config: SimulationConfig,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
        device: str = "CPU",  # ! TODO reimplement
        save_statevector_to_disk: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(config, lattice, logger, device)

        self.execution_backend = self.config.execution_backend
        self.sampling_backend = self.config.sampling_backend
        self.statevector_to_disk = save_statevector_to_disk
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @override
    def run(
        self,
        num_steps: int,
        num_shots: int,
        output_directory: str,
        output_file_name: str = "step",
        statevector_snapshots: bool = False,
        recompile_each_step: bool = False,  # ! Document
    ) -> QBMResult:
        """
        Runs the simulation for a given number of time steps.

        Parameters
        ----------
        num_steps : int
            The number of time steps to simulate.
        num_shots : int
            The number of shots to sample at each time step.
        output_directory : str
            The directory to which the results are written.
        output_file_name : str, optional
            The prefix of the per-time-step output files, by default ``"step"``.
        statevector_snapshots : bool, optional
            Whether to carry the statevector between time steps instead of
            re-simulating from the initial conditions, by default False.
        recompile_each_step : bool, optional
            Whether to transpile the assembled circuit anew at every time step
            of the snapshot loop, by default False.

        Returns
        -------
        QBMResult
            The result of the simulation.
        """
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
                recompile_each_step,
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
        recompile_each_step: bool = False,  # ! Document
    ) -> QBMResult:
        # A reinitializer that carries the statevector unchanged needs no
        # feedback between time steps, so the whole trajectory is one job.
        if (
            self.reinitializer.reuses_statevector()
            and self.config.statevector_sampling
            and not recompile_each_step
        ):
            return self._run_single_job_snapshot_loop(
                num_steps, num_shots, initial_conditions, simulation_result
            )

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
                if recompile_each_step:
                    circuit.compose(self.config.algorithm_copy, inplace=True)
                    circuit = transpile(
                        circuit,
                        backend=self.execution_backend,
                        optimization_level=0,
                    )
                else:
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
                counts = self._sample(
                    qiskit_execution_result.data(0)["step"], num_shots
                )
            else:
                circuit.compose(self.config.postprocessing.copy(), inplace=True)  # type: ignore
                circuit.compose(self.config.measurement.copy(), inplace=True)  # type: ignore
                qiskit_execution_result = self.execution_backend.run(  # type: ignore
                    circuit, shots=num_shots, seed_simulator=self._aer_seed(step)
                ).result()
                counts = qiskit_execution_result.get_counts()

            simulation_result.save_timestep_counts(counts, step)

            if self.statevector_to_disk:
                simulation_result.save_statevector(
                    qiskit_execution_result.data(0)["step"], step=step
                )

            # Update the initial conditions for the next step
            initial_conditions = self.reinitializer.reinitialize(
                statevector=qiskit_execution_result.data(0)["step"]
                if self.reinitializer.requires_statevector()
                else Statevector([0]),
                counts=counts,
                backend=self.config.execution_backend,
                optimization_level=self.config.optimization_level,
            )

            self.logger.info(
                f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
            )

        return simulation_result

    def _run_single_job_snapshot_loop(
        self,
        num_steps: int,
        num_shots: int,
        initial_conditions: QiskitQC,
        simulation_result: QBMResult,
        snapshots_per_job: int | None = None,
    ) -> QBMResult:
        registers = self.config.measurement.qregs + self.config.measurement.cregs  # type: ignore
        num_qubits: int = self.config.measurement.num_qubits  # type: ignore[union-attr]
        if snapshots_per_job is None:
            # Every snapshot stays in the job result until it returns, so a
            # job holds at most about a gibibyte of them.
            snapshots_per_job = max(1, (1 << 30) // (16 << num_qubits))

        state: Statevector | None = None
        step = 0
        while step <= num_steps:
            circuit = QiskitQC(*registers)
            first_step = step
            if state is None:
                circuit.compose(
                    initial_conditions, inplace=True, qubits=range(circuit.num_qubits)
                )
                circuit.save_statevector(label=f"step_{step}")
                step += 1
            else:
                circuit.append(SetStatevector(state), circuit.qubits)
            while step <= num_steps and step - first_step < snapshots_per_job:
                circuit.compose(self.config.algorithm, inplace=True)
                circuit.save_statevector(label=f"step_{step}")
                step += 1

            self.logger.info(
                f"Main circuit for steps {first_step}-{step - 1} has properties {get_circuit_properties(circuit)}"
            )
            simulation_start_time = perf_counter_ns()
            data = self.execution_backend.run(circuit).result().data(0)  # type: ignore
            self.logger.info(
                f"Simulation of steps {first_step}-{step - 1} took {perf_counter_ns() - simulation_start_time} (ns)"
            )

            for saved_step in range(first_step, step):
                state = data[f"step_{saved_step}"]
                simulation_result.save_timestep_counts(
                    self._sample(state, num_shots), saved_step
                )
                if self.statevector_to_disk:
                    simulation_result.save_statevector(state, step=saved_step)

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

            self.logger.info(
                f"Main circuit for step {step} has properties {get_circuit_properties(circuit)}"
            )

            if self.config.statevector_sampling:
                circuit.save_statevector(label="step")
                result = self.execution_backend.run(circuit).result()  # type: ignore
                counts = self._sample(result.data(0)["step"], num_shots)
            else:
                circuit.compose(self.config.postprocessing.copy(), inplace=True)  # type: ignore
                circuit.compose(self.config.measurement.copy(), inplace=True)  # type: ignore
                counts = (
                    self.execution_backend.run(  # type: ignore
                        circuit, shots=num_shots, seed_simulator=self._aer_seed(step)
                    )
                    .result()
                    .get_counts()
                )

            simulation_result.save_timestep_counts(counts, step)

            self.logger.info(
                f"Simulation of {step} steps took {perf_counter_ns() - step_start_time} (ns)"
            )

        return simulation_result

    def _aer_seed(self, step: int) -> int | None:
        """
        The simulator seed for time step ``step``; ``None`` when the runner is unseeded.

        Parameters
        ----------
        step : int
            The time step, which offsets the seed so steps draw independently.

        Returns
        -------
        int | None
            The seed to pass to Aer.
        """
        return None if self.seed is None else self.seed + step

    def _measure_pairs(self) -> List[Tuple[int, int]]:
        """
        The ``(qubit, classical bit)`` pairs measured by the ``measurement`` circuit.

        Returns
        -------
        List[Tuple[int, int]]
            The pairs, one per ``measure`` instruction.

        Raises
        ------
        ExecutionException
            If the measurement circuit contains gates: sampling the saved
            statevector only reproduces a measure-only circuit.
        """
        measurement: QiskitQC = self.config.measurement  # type: ignore[assignment]
        pairs = []
        for instruction in measurement.data:
            name = instruction.operation.name
            if name == "measure":
                pairs.append(
                    (
                        measurement.find_bit(instruction.qubits[0]).index,
                        measurement.find_bit(instruction.clbits[0]).index,
                    )
                )
            elif name != "barrier":
                raise ExecutionException(
                    f"Statevector sampling needs a measure-only measurement circuit, found '{name}'. Move gates into postprocessing or disable statevector_sampling."
                )
        return pairs

    def _postprocess(self, statevector: Statevector) -> Statevector:
        """
        Apply the ``postprocessing`` circuit to ``statevector``.

        Parameters
        ----------
        statevector : Statevector
            The state after a time step.

        Returns
        -------
        Statevector
            The state to measure; ``statevector`` itself if there is no postprocessing.
        """
        postprocessing: QiskitQC = self.config.postprocessing  # type: ignore[assignment]
        if postprocessing.size() == 0:
            return statevector
        circuit = QiskitQC(
            *(self.config.measurement.qregs + self.config.measurement.cregs)  # type: ignore
        )
        circuit.append(SetStatevector(statevector), circuit.qubits)
        circuit.compose(postprocessing, inplace=True)
        circuit.save_statevector(label="post")
        return self.execution_backend.run(circuit).result().data(0)["post"]  # type: ignore

    def _sample(self, statevector: Statevector, num_shots: int) -> Dict[str, float]:
        """
        Sample the measurement of ``statevector`` after postprocessing.

        Parameters
        ----------
        statevector : Statevector
            The state after a time step.
        num_shots : int
            The number of shots to draw.

        Returns
        -------
        Dict[str, float]
            Counts in Qiskit's format: bitstrings over the classical register,
            classical bit 0 rightmost.
        """
        postprocessed = self._postprocess(statevector)
        num_qubits = postprocessed.num_qubits
        num_clbits: int = self.config.measurement.num_clbits  # type: ignore[union-attr]
        # Most significant classical bit first, so the marginal's flat index
        # is the measured bits in register order.
        pairs = sorted(self._measure_pairs(), key=lambda pair: pair[1], reverse=True)

        # Axis i of the probability tensor is qubit n-1-i; summing out the
        # unmeasured axes and ordering the rest by classical bit gives the
        # marginal in one pass, without index arrays over the full state.
        tensor = (np.abs(np.asarray(postprocessed.data)) ** 2).reshape([2] * num_qubits)
        kept_axes = [num_qubits - 1 - qubit for qubit, _ in pairs]
        dropped_axes = tuple(sorted(set(range(num_qubits)) - set(kept_axes)))
        marginal = tensor.sum(axis=dropped_axes) if dropped_axes else tensor
        remaining_axes = [
            axis for axis in range(num_qubits) if axis not in dropped_axes
        ]
        marginal = marginal.transpose(
            [remaining_axes.index(axis) for axis in kept_axes]
        ).ravel()

        # Place each measured bit at its classical bit within the full register.
        packed = np.arange(marginal.size)
        keys = np.zeros_like(packed)
        for position, (_, clbit) in enumerate(pairs):
            keys |= ((packed >> (len(pairs) - 1 - position)) & 1) << clbit
        probabilities: np.ndarray = np.zeros(1 << num_clbits)
        probabilities[keys] = marginal

        draws: np.ndarray = self.rng.multinomial(
            num_shots, probabilities / probabilities.sum()
        )
        return {
            format(key, f"0{num_clbits}b"): float(count)
            for key, count in enumerate(draws)
            if count
        }
