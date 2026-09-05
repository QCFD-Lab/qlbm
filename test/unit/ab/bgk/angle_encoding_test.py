import numpy as np
import pytest

from qlbm.lattice.bgk import D2Q9AngleEncoding
from qlbm.tools.exceptions import LatticeException


def test_default_parameters_are_normalized(encoding):
    assert encoding.beta.shape == (encoding.NUM_BRANCHES,)
    assert np.all(encoding.beta > 0.0)
    assert np.isclose(np.sum(encoding.beta**2), 1.0)
    assert encoding.max_velocity == pytest.approx(0.1)


def test_beta_must_have_six_entries():
    with pytest.raises(LatticeException, match="branch amplitudes"):
        D2Q9AngleEncoding(beta=np.ones(5) / np.sqrt(5))


def test_beta_must_be_strictly_positive():
    beta = np.zeros(6)
    beta[0] = 1.0

    with pytest.raises(LatticeException, match="strictly positive"):
        D2Q9AngleEncoding(beta=beta)


def test_beta_must_be_normalized():
    with pytest.raises(LatticeException, match="normalized"):
        D2Q9AngleEncoding(beta=np.full(6, 0.5))


def test_velocity_bound_must_be_positive():
    with pytest.raises(LatticeException, match="must be positive"):
        D2Q9AngleEncoding(max_velocity=0.0)


def test_angles_invert_the_velocity_mapping(encoding, velocity_samples):
    for velocity_x, velocity_y in velocity_samples:
        angle_x, angle_y = encoding.angles(velocity_x, velocity_y)

        assert encoding.max_velocity * np.sin(angle_x) == pytest.approx(velocity_x)
        assert encoding.max_velocity * np.sin(angle_y) == pytest.approx(velocity_y)


def test_angles_reject_velocities_beyond_the_bound(encoding):
    with pytest.raises(LatticeException, match="exceeds the angle-encoding bound"):
        encoding.angles(0.2, 0.0)


def test_equilibrium_has_the_prescribed_moments(encoding, velocity_samples):
    for velocity_x, velocity_y in velocity_samples:
        populations = encoding.equilibrium(1.3, velocity_x, velocity_y)
        density, recovered_x, recovered_y = encoding.macroscopic(populations)

        assert density == pytest.approx(1.3)
        assert recovered_x == pytest.approx(velocity_x)
        assert recovered_y == pytest.approx(velocity_y)


def test_equilibrium_is_vectorized(encoding, taylor_green_field_4x4):
    density, velocity_x, velocity_y = taylor_green_field_4x4
    populations = encoding.equilibrium(density, velocity_x, velocity_y)

    assert populations.shape == (4, 4, encoding.NUM_POPULATIONS)
    assert populations[2, 3] == pytest.approx(
        encoding.equilibrium(density[2, 3], velocity_x[2, 3], velocity_y[2, 3])
    )


def test_macroscopic_zeroes_masked_gridpoints(encoding):
    populations = encoding.equilibrium(
        np.ones((2, 2)), np.full((2, 2), 0.05), np.zeros((2, 2))
    )
    mask = np.array([[True, False], [False, False]])
    populations[mask, :] = 0.0

    density, velocity_x, velocity_y = encoding.macroscopic(populations, mask)

    assert density[0, 0] == 0.0
    assert velocity_x[0, 0] == 0.0
    assert velocity_y[0, 0] == 0.0
    assert density[1, 1] == pytest.approx(1.0)


def test_macroscopic_rejects_non_positive_density(encoding):
    with pytest.raises(LatticeException, match="non-positive density"):
        encoding.macroscopic(np.zeros((2, 2, 9)))


def test_features_reproduce_the_equilibrium(encoding, velocity_samples):
    for velocity_x, velocity_y in velocity_samples:
        angle_x, angle_y = encoding.angles(velocity_x, velocity_y)
        features = np.array(
            [
                1.0,
                np.sin(angle_x),
                np.sin(angle_y),
                np.cos(2 * angle_x),
                np.cos(2 * angle_y),
                np.sin(angle_x) * np.sin(angle_y),
            ]
        )

        assert encoding.feature_matrix() @ features == pytest.approx(
            encoding.equilibrium(1.0, velocity_x, velocity_y)
        )


def test_branch_states_are_normalized(encoding, velocity_samples):
    for velocity_x, velocity_y in velocity_samples:
        state = encoding.branch_state(*encoding.angles(velocity_x, velocity_y))

        assert state.shape == (encoding.NUM_STATES,)
        assert np.linalg.norm(state) == pytest.approx(1.0)


def test_branch_states_leave_the_unused_branches_empty(encoding):
    state = encoding.branch_state(*encoding.angles(0.05, -0.02))

    assert state[4 * encoding.NUM_BRANCHES :] == pytest.approx(0.0)


def test_sparse_collision_map_is_exact(encoding, velocity_samples):
    collision_map = encoding.sparse_collision_map()

    for velocity_x, velocity_y in velocity_samples:
        state = encoding.branch_state(*encoding.angles(velocity_x, velocity_y))

        assert collision_map @ state == pytest.approx(
            encoding.equilibrium(1.0, velocity_x, velocity_y)
        )


def test_reachable_subspace_has_rank_thirteen(encoding):
    basis = encoding.reachable_subspace_basis()

    assert basis.shape == (encoding.NUM_STATES, 13)
    assert encoding.reachable_subspace_rank == 13
    assert basis.T @ basis == pytest.approx(np.eye(13))


def test_reachable_subspace_contains_every_branch_state(encoding, velocity_samples):
    basis = encoding.reachable_subspace_basis()

    for velocity_x, velocity_y in velocity_samples:
        state = encoding.branch_state(*encoding.angles(velocity_x, velocity_y))

        assert basis @ (basis.T @ state) == pytest.approx(state)


def test_collision_unitary_is_orthogonal(encoding):
    unitary = encoding.collision_unitary()

    assert unitary.shape == (encoding.NUM_STATES, encoding.NUM_STATES)
    assert unitary.T @ unitary == pytest.approx(np.eye(encoding.NUM_STATES), abs=1e-12)


def test_collision_unitary_is_reproducible(encoding):
    assert encoding.collision_unitary() == pytest.approx(
        D2Q9AngleEncoding().collision_unitary()
    )


def test_collision_unitary_maps_branch_states_onto_the_equilibrium(
    encoding, velocity_samples
):
    unitary = encoding.collision_unitary()

    for velocity_x, velocity_y in velocity_samples:
        state = encoding.branch_state(*encoding.angles(velocity_x, velocity_y))
        collided = unitary @ state

        assert encoding.alpha * collided[encoding.physical_state_indices()] == (
            pytest.approx(encoding.equilibrium(1.0, velocity_x, velocity_y))
        )
        assert collided[encoding.unused_state_indices()] == pytest.approx(0.0)


def test_physical_sector_probability_follows_the_normalization(
    encoding, velocity_samples
):
    unitary = encoding.collision_unitary()

    for velocity_x, velocity_y in velocity_samples:
        state = encoding.branch_state(*encoding.angles(velocity_x, velocity_y))
        equilibrium = encoding.equilibrium(1.0, velocity_x, velocity_y)
        probability = float(
            np.sum((unitary @ state)[encoding.physical_state_indices()] ** 2)
        )

        assert probability == pytest.approx(np.sum(equilibrium**2) / encoding.alpha**2)
        assert 0.0 < probability <= 1.0


def test_populations_are_recovered_from_probabilities(encoding):
    populations = encoding.equilibrium(1.0, 0.05, -0.02)
    density_norm = 7.5
    probabilities = (populations / (encoding.alpha * density_norm)) ** 2

    assert encoding.populations_from_probabilities(probabilities, density_norm) == (
        pytest.approx(populations)
    )


def test_flow_field_is_recovered_from_branch_amplitudes(encoding, velocity_samples):
    density_norm = 3.25

    for velocity_x, velocity_y in velocity_samples:
        density = 1.1
        amplitudes = (density / density_norm) * encoding.branch_state(
            *encoding.angles(velocity_x, velocity_y)
        )

        recovered = encoding.flow_field_from_branch_amplitudes(amplitudes, density_norm)

        assert recovered[0] == pytest.approx(density)
        assert recovered[1] == pytest.approx(velocity_x)
        assert recovered[2] == pytest.approx(velocity_y)


def test_flow_field_from_branch_amplitudes_ignores_empty_gridpoints(encoding):
    density, velocity_x, velocity_y = encoding.flow_field_from_branch_amplitudes(
        np.zeros((2, encoding.NUM_STATES)), 1.0
    )

    assert density == pytest.approx(0.0)
    assert velocity_x == pytest.approx(0.0)
    assert velocity_y == pytest.approx(0.0)


def test_state_index_partitions_are_disjoint(encoding):
    physical = set(encoding.physical_state_indices().tolist())
    unused = set(encoding.unused_state_indices().tolist())

    assert len(physical) == encoding.NUM_POPULATIONS
    assert len(unused) == encoding.NUM_VELOCITY_STATES - encoding.NUM_POPULATIONS
    assert not physical & unused
    assert min(physical) == encoding.NUM_VELOCITY_STATES


def test_a_larger_velocity_bound_costs_normalization(encoding):
    fast = D2Q9AngleEncoding(max_velocity=0.2)

    assert fast.max_velocity == pytest.approx(0.2)
    assert fast.alpha > encoding.alpha
