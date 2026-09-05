from typing import cast

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qlbm.components import EmptyPrimitive
from qlbm.components.ab import (
    ABBGKQLBM,
    ABBGKInitialConditions,
    ABBGKMeasurement,
)
from qlbm.infra import QiskitRunner, SimulationConfig
from qlbm.infra.reinitialize import ABBGKReinitializer
from qlbm.infra.result import ABBGKResult
from qlbm.tools.exceptions import ExecutionException


def exact_counts(lattice, statevector, num_shots):
    """Turn exact amplitudes into the count strings an ABBGKMeasurement would give."""
    probabilities = np.abs(np.asarray(statevector)) ** 2
    num_bits = (
        lattice.num_grid_qubits
        + lattice.num_velocity_qubits
        + lattice.num_marker_qubits
    )
    counts = {}

    for x in range(lattice.num_gridpoints[0] + 1):
        for y in range(lattice.num_gridpoints[1] + 1):
            for state, index in enumerate(
                lattice.collision_basis_indices()[x, y].tolist()
            ):
                if probabilities[index] <= 0.0:
                    continue

                measured = (
                    state << lattice.num_grid_qubits
                    | y << lattice.num_gridpoints[0].bit_length()
                    | x
                )
                counts[format(measured, f"0{num_bits}b")] = (
                    num_shots * probabilities[index]
                )

    return counts


def test_result_reconstructs_a_collided_flow_field(
    tmp_path, ab_bgk_lattice_4x4, taylor_green_field_4x4
):
    density, velocity_x, velocity_y = taylor_green_field_4x4
    initial = ABBGKInitialConditions(
        ab_bgk_lattice_4x4, density, velocity_x, velocity_y
    )
    evolved = Statevector(initial.statevector).evolve(
        ABBGKQLBM(ab_bgk_lattice_4x4).circuit
    )

    result = ABBGKResult(ab_bgk_lattice_4x4, str(tmp_path))
    from_counts = result.counts_to_flow_field(
        exact_counts(ab_bgk_lattice_4x4, evolved, 2**20)
    )
    from_amplitudes = result.statevector_to_flow_field(evolved)

    for sampled, exact in zip(from_counts, from_amplitudes):
        assert sampled == pytest.approx(exact, abs=1e-9)


def test_result_reconstructs_the_uncollided_initial_state(
    tmp_path, ab_bgk_lattice_4x4, taylor_green_field_4x4
):
    density, velocity_x, velocity_y = taylor_green_field_4x4
    initial = ABBGKInitialConditions(
        ab_bgk_lattice_4x4, density, velocity_x, velocity_y
    )

    result = ABBGKResult(ab_bgk_lattice_4x4, str(tmp_path))
    recovered = result.statevector_to_flow_field(
        Statevector(initial.statevector), collided=False
    )

    assert recovered[0] == pytest.approx(density)
    assert recovered[1] == pytest.approx(velocity_x)
    assert recovered[2] == pytest.approx(velocity_y)


def test_result_records_the_physical_sector_probability(
    tmp_path, ab_bgk_lattice_4x4, taylor_green_field_4x4
):
    density, velocity_x, velocity_y = taylor_green_field_4x4
    evolved = Statevector(
        ABBGKInitialConditions(
            ab_bgk_lattice_4x4, density, velocity_x, velocity_y
        ).statevector
    ).evolve(ABBGKQLBM(ab_bgk_lattice_4x4).circuit)

    result = ABBGKResult(ab_bgk_lattice_4x4, str(tmp_path))
    result.statevector_to_flow_field(evolved)

    assert 0.9 < result.physical_sector_probability <= 1.0


def build_config(lattice, density, velocity_x, velocity_y):
    return SimulationConfig(
        initial_conditions=ABBGKInitialConditions(
            lattice, density, velocity_x, velocity_y
        ),
        algorithm=ABBGKQLBM(lattice),
        postprocessing=EmptyPrimitive(lattice),
        measurement=ABBGKMeasurement(lattice),
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=0,
        statevector_sampling=True,
        execution_backend=AerSimulator(method="statevector"),
        sampling_backend=AerSimulator(method="statevector"),
    )


def test_runner_advances_the_hybrid_loop(
    tmp_path, ab_bgk_lattice_4x4, taylor_green_field_4x4
):
    density, velocity_x, velocity_y = taylor_green_field_4x4
    config = build_config(ab_bgk_lattice_4x4, density, velocity_x, velocity_y)
    config.prepare_for_simulation()

    runner = QiskitRunner(config, ab_bgk_lattice_4x4, save_statevector_to_disk=True)
    result = cast(
        ABBGKResult, runner.run(2, 2**12, str(tmp_path), statevector_snapshots=True)
    )
    reinitializer = cast(ABBGKReinitializer, runner.reinitializer)

    # The un-evolved state of step zero holds the encoded initial conditions.
    assert result.flow_fields[0][1] == pytest.approx(velocity_x, abs=1e-8)
    assert result.flow_fields[0][2] == pytest.approx(velocity_y, abs=1e-8)

    # A Taylor-Green vortex decays, so later steps are strictly slower.
    speeds = [
        np.max(np.sqrt(field[1] ** 2 + field[2] ** 2))
        for field in result.flow_fields.values()
    ]
    assert speeds[0] > speeds[1] > speeds[2]
    assert reinitializer.steps_reinitialized == 3


def test_runner_refuses_to_concatenate_time_steps(
    tmp_path, ab_bgk_lattice_4x4, taylor_green_field_4x4
):
    density, velocity_x, velocity_y = taylor_green_field_4x4
    config = build_config(ab_bgk_lattice_4x4, density, velocity_x, velocity_y)
    config.prepare_for_simulation()

    runner = QiskitRunner(config, ab_bgk_lattice_4x4)

    with pytest.raises(ExecutionException, match="statevector_snapshots=True"):
        runner.run(2, 2**12, str(tmp_path), statevector_snapshots=False)
