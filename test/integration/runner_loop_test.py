"""Sampling and time-loop behaviour of the :class:`.QiskitRunner`."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm.components.common import EmptyPrimitive
from qlbm.components.ms import MSQLBM, GridMeasurement, MSInitialConditions
from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.runner import QiskitRunner
from qlbm.infra.runner.simulation_config import SimulationConfig
from qlbm.lattice import MSLattice
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice

OUTPUT_DIR = "test/artifacts"
NUM_STEPS = 2
SHOTS = 20000


def ms_config(sampling_backend=AerSimulator(method="statevector")):
    """A collisionless MS configuration with statevector sampling."""
    lattice = MSLattice("test/resources/symmetric_2d_1_obstacle.json")
    config = SimulationConfig(
        initial_conditions=MSInitialConditions(lattice),
        algorithm=MSQLBM(lattice),
        postprocessing=EmptyPrimitive(lattice),
        measurement=GridMeasurement(lattice),
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=0,
        statevector_sampling=True,
        execution_backend=AerSimulator(method="statevector"),
        sampling_backend=sampling_backend,
    )
    config.validate()
    config.prepare_for_simulation()
    return config, lattice


def test_statevector_sampling_matches_aer_counts():
    """Counts drawn from the saved statevector follow the same distribution, in the same format, as an Aer measurement of it."""
    config, lattice = ms_config()
    runner = QiskitRunner(config, lattice, seed=7)

    circuit = QuantumCircuit(*(config.measurement.qregs + config.measurement.cregs))
    circuit.compose(config.initial_conditions, inplace=True)
    circuit.compose(config.algorithm, inplace=True)
    circuit.save_statevector(label="step")
    statevector = config.execution_backend.run(circuit).result().data(0)["step"]

    measurement = QuantumCircuit(*(config.measurement.qregs + config.measurement.cregs))
    measurement.compose(runner.statevector_to_circuit(statevector), inplace=True)
    measurement.compose(config.postprocessing, inplace=True)
    measurement.compose(config.measurement, inplace=True)
    aer_counts = (
        config.sampling_backend.run(measurement, shots=SHOTS).result().get_counts()
    )

    counts = runner._sample(statevector, SHOTS)

    assert sum(counts.values()) == SHOTS
    assert {len(key) for key in counts} == {len(key) for key in aer_counts}
    tolerance = 4 / np.sqrt(SHOTS)
    for key in set(counts) | set(aer_counts):
        assert abs(counts.get(key, 0) - aer_counts.get(key, 0)) / SHOTS < tolerance


def test_single_job_snapshot_loop_matches_per_step_loop():
    """One Aer job with a snapshot per step yields the statevectors the per-step loop yields."""
    trajectories = []
    for per_step in (False, True):
        config, lattice = ms_config()
        runner = QiskitRunner(config, lattice, save_statevector_to_disk=True, seed=1)
        if per_step:
            runner.reinitializer.reuses_statevector = lambda: False  # type: ignore[method-assign]
        directory = f"{OUTPUT_DIR}/snapshot-loop-{int(per_step)}"
        runner.run(NUM_STEPS, SHOTS, directory, statevector_snapshots=True)
        trajectories.append(
            [
                np.load(f"{directory}/statevectors/step_{step}.npy")
                for step in range(NUM_STEPS + 1)
            ]
        )
    for step, (single, per_step) in enumerate(zip(*trajectories)):
        np.testing.assert_allclose(single, per_step, atol=1e-12, err_msg=f"step {step}")


def test_sampling_backend_is_optional_with_statevector_sampling():
    """With statevector sampling the configuration validates and runs without a sampling backend."""
    config, lattice = ms_config(sampling_backend=None)
    result = QiskitRunner(config, lattice, seed=3).run(
        1, 128, f"{OUTPUT_DIR}/no-sampling-backend", statevector_snapshots=True
    )
    assert result is not None


def test_spacetime_reinitialization_is_not_transpiled_for_aer():
    """The re-synthesised initial conditions run unchanged on Aer; transpiling them would give the same state."""
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {"dim": {"x": 8}, "velocities": "D1Q2"},
            "geometry": [{"shape": "cuboid", "x": [3, 4], "boundary": "bounceback"}],
        },
    )
    backend = AerSimulator(method="statevector")
    reinitializer = lattice.create_reinitializer(CircuitCompiler("QISKIT", "QISKIT"))
    # Two occupied gridpoints as a counts dictionary: velocities first, then the grid.
    counts = {"01" + format(3, "04b")[::-1]: 5, "11" + format(5, "04b")[::-1]: 3}
    circuit = reinitializer.reinitialize(None, counts, backend=backend)

    assert "mcx" in circuit.count_ops()

    def statevector(qc):
        qc = qc.copy()
        qc.save_statevector(label="state")
        return np.asarray(backend.run(qc).result().data(0)["state"])

    np.testing.assert_allclose(
        statevector(circuit),
        statevector(transpile(circuit, backend=backend, optimization_level=0)),
        atol=1e-12,
    )
