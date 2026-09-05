import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qlbm.components.ab import ABBGKQLBM, ABBGKInitialConditions
from qlbm.infra import CircuitCompiler
from qlbm.infra.reinitialize import ABBGKReinitializer
from qlbm.tools.exceptions import CircuitException


def classical_timestep(encoding, populations, solid_mask=None):
    """Advance one tau=1 BGK step with periodic streaming and halfway bounce-back."""
    num_x, num_y, _ = populations.shape
    velocities = encoding.velocities.astype(int)
    opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]

    density, velocity_x, velocity_y = encoding.macroscopic(populations, solid_mask)
    collided = encoding.equilibrium(density, velocity_x, velocity_y)

    if solid_mask is not None:
        collided[solid_mask, :] = 0.0

    streamed = np.zeros_like(collided)

    for x in range(num_x):
        for y in range(num_y):
            if solid_mask is not None and solid_mask[x, y]:
                continue

            for velocity, (shift_x, shift_y) in enumerate(velocities):
                target = ((x + shift_x) % num_x, (y + shift_y) % num_y)

                if solid_mask is not None and solid_mask[target]:
                    streamed[x, y, opposite[velocity]] += collided[x, y, velocity]
                else:
                    streamed[target[0], target[1], velocity] += collided[x, y, velocity]

    return streamed


def simulate(lattice, density, velocity_x, velocity_y, num_steps):
    """Run the hybrid loop and return the flow field of each time step."""
    algorithm = ABBGKQLBM(lattice)
    reinitializer = ABBGKReinitializer(lattice, CircuitCompiler("QISKIT", "QISKIT"))
    history = []

    for _ in range(num_steps):
        initial = ABBGKInitialConditions(lattice, density, velocity_x, velocity_y)
        evolved = Statevector(initial.statevector).evolve(algorithm.circuit)
        density, velocity_x, velocity_y = reinitializer.decode(evolved)
        history.append(
            (density, velocity_x, velocity_y, dict(reinitializer.diagnostics))
        )

    return history


def test_initial_conditions_encode_every_gridpoint(
    ab_bgk_lattice_4x4, taylor_green_field_4x4
):
    density, velocity_x, velocity_y = taylor_green_field_4x4
    initial = ABBGKInitialConditions(
        ab_bgk_lattice_4x4, density, velocity_x, velocity_y
    )

    encoding = ab_bgk_lattice_4x4.encoding
    amplitudes = initial.statevector.real[ab_bgk_lattice_4x4.collision_basis_indices()]

    assert np.linalg.norm(initial.statevector) == pytest.approx(1.0)

    for x in range(4):
        for y in range(4):
            expected = (density[x, y] / initial.density_norm) * encoding.branch_state(
                *encoding.angles(velocity_x[x, y], velocity_y[x, y])
            )

            assert amplitudes[x, y] == pytest.approx(expected, abs=1e-13)


def test_initial_conditions_publish_the_density_norm(
    ab_bgk_lattice_4x4, taylor_green_field_4x4
):
    density, velocity_x, velocity_y = taylor_green_field_4x4
    initial = ABBGKInitialConditions(
        ab_bgk_lattice_4x4, 2.5 * density, velocity_x, velocity_y
    )

    assert initial.density_norm == pytest.approx(np.linalg.norm(2.5 * density))
    assert ab_bgk_lattice_4x4.density_norm == pytest.approx(initial.density_norm)


def test_initial_conditions_reject_mismatched_shapes(ab_bgk_lattice_4x4):
    with pytest.raises(CircuitException, match="must have shape"):
        ABBGKInitialConditions(
            ab_bgk_lattice_4x4, np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
        )


def test_initial_conditions_reject_negative_density(ab_bgk_lattice_4x4):
    with pytest.raises(CircuitException, match="non-negative"):
        ABBGKInitialConditions(
            ab_bgk_lattice_4x4, -np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))
        )


def test_initial_conditions_reject_an_empty_density(ab_bgk_lattice_4x4):
    with pytest.raises(CircuitException, match="identically zero"):
        ABBGKInitialConditions(
            ab_bgk_lattice_4x4, np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))
        )


def test_periodic_timestep_matches_the_classical_solver(
    ab_bgk_lattice_4x4, taylor_green_field_4x4
):
    encoding = ab_bgk_lattice_4x4.encoding
    density, velocity_x, velocity_y = taylor_green_field_4x4

    populations = encoding.equilibrium(density, velocity_x, velocity_y)
    history = simulate(ab_bgk_lattice_4x4, density, velocity_x, velocity_y, 3)

    for quantum_density, quantum_x, quantum_y, _ in history:
        populations = classical_timestep(encoding, populations)
        reference = encoding.macroscopic(populations)

        assert quantum_density == pytest.approx(reference[0], abs=1e-10)
        assert quantum_x == pytest.approx(reference[1], abs=1e-10)
        assert quantum_y == pytest.approx(reference[2], abs=1e-10)


def test_bounce_back_timestep_matches_the_classical_solver(ab_bgk_lattice_4x4_obstacle):
    encoding = ab_bgk_lattice_4x4_obstacle.encoding
    solid = ABBGKReinitializer.solid_mask_from_geometry(ab_bgk_lattice_4x4_obstacle)

    density = np.ones((4, 4))
    velocity_x = np.full((4, 4), 0.04)
    velocity_y = np.zeros((4, 4))
    density[solid] = 0.0
    velocity_x[solid] = 0.0

    populations = encoding.equilibrium(density, velocity_x, velocity_y)
    populations[solid, :] = 0.0

    history = simulate(ab_bgk_lattice_4x4_obstacle, density, velocity_x, velocity_y, 3)

    for quantum_density, quantum_x, quantum_y, _ in history:
        populations = classical_timestep(encoding, populations, solid)
        reference = encoding.macroscopic(populations, solid)

        assert quantum_density == pytest.approx(reference[0], abs=1e-10)
        assert quantum_x == pytest.approx(reference[1], abs=1e-10)
        assert quantum_y == pytest.approx(reference[2], abs=1e-10)


def test_no_probability_leaks_outside_the_physical_sector(
    ab_bgk_lattice_4x4_obstacle,
):
    solid = ABBGKReinitializer.solid_mask_from_geometry(ab_bgk_lattice_4x4_obstacle)
    density = np.ones((4, 4))
    density[solid] = 0.0

    for _, _, _, diagnostics in simulate(
        ab_bgk_lattice_4x4_obstacle,
        density,
        np.full((4, 4), 0.04) * ~solid,
        np.zeros((4, 4)),
        3,
    ):
        assert diagnostics["unused_velocity_probability"] < 1e-12
        assert diagnostics["ancilla_probability"] < 1e-12
        assert diagnostics["solid_population"] < 1e-12
        assert 0.0 < diagnostics["physical_sector_probability"] <= 1.0


def test_solid_mask_covers_the_obstacle(ab_bgk_lattice_4x4_obstacle):
    solid = ABBGKReinitializer.solid_mask_from_geometry(ab_bgk_lattice_4x4_obstacle)

    assert solid.shape == (4, 4)
    assert solid[1:3, 1:3].all()
    assert solid.sum() == 4


def test_reinitializer_requires_snapshots(ab_bgk_lattice_4x4):
    reinitializer = ABBGKReinitializer(
        ab_bgk_lattice_4x4, CircuitCompiler("QISKIT", "QISKIT")
    )

    assert reinitializer.requires_statevector()
    assert reinitializer.requires_statevector_snapshots()
