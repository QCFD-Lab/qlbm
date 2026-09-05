import numpy as np
import pytest

from qlbm.lattice.lattices.ab_bgk_lattice import ABBGKLattice
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import basis_state_offsets


def test_exactly_one_marker_qubit_is_reserved(ab_bgk_lattice_4x4):
    assert ab_bgk_lattice_4x4.num_marker_qubits == 1
    assert len(ab_bgk_lattice_4x4.marker_index()) == 1


def test_collision_index_is_the_velocity_register_plus_the_marker(ab_bgk_lattice_4x4):
    assert ab_bgk_lattice_4x4.collision_index() == (
        ab_bgk_lattice_4x4.velocity_index() + ab_bgk_lattice_4x4.marker_index()
    )
    assert len(ab_bgk_lattice_4x4.collision_index()) == (
        ab_bgk_lattice_4x4.encoding.NUM_COLLISION_QUBITS
    )


def test_density_norm_starts_at_one(ab_bgk_lattice_4x4):
    assert ab_bgk_lattice_4x4.density_norm == pytest.approx(1.0)


def test_multiple_geometries_are_rejected(ab_bgk_lattice_4x4):
    with pytest.raises(LatticeException, match="cannot simulate multiple geometries"):
        ab_bgk_lattice_4x4.set_geometries(
            [
                [
                    {
                        "shape": "cuboid",
                        "x": [1, 1],
                        "y": [1, 1],
                        "boundary": "bounceback",
                    }
                ],
                [
                    {
                        "shape": "cuboid",
                        "x": [2, 2],
                        "y": [2, 2],
                        "boundary": "bounceback",
                    }
                ],
            ]
        )


def test_setting_a_single_geometry_keeps_the_marker_reserved(ab_bgk_lattice_4x4):
    ab_bgk_lattice_4x4.set_geometries(
        [[{"shape": "cuboid", "x": [1, 2], "y": [1, 2], "boundary": "bounceback"}]]
    )

    assert ab_bgk_lattice_4x4.num_marker_qubits == 1
    assert len(ab_bgk_lattice_4x4.shape_list) == 1


def test_non_d2q9_discretizations_are_rejected():
    with pytest.raises(LatticeException, match="only implemented for D2Q9"):
        ABBGKLattice(
            {
                "lattice": {"dim": {"x": 8}, "velocities": "d1q3"},
                "geometry": [],
            }
        )


def test_grid_offsets_encode_every_coordinate(ab_bgk_lattice_4x4):
    offsets = ab_bgk_lattice_4x4.grid_basis_offsets()

    assert offsets.shape == (4, 4)
    assert len(set(offsets.ravel().tolist())) == 16

    for x in range(4):
        for y in range(4):
            assert offsets[x, y] == (
                basis_state_offsets(x, ab_bgk_lattice_4x4.grid_index(0))
                | basis_state_offsets(y, ab_bgk_lattice_4x4.grid_index(1))
            )


def test_collision_offsets_span_the_collision_register(ab_bgk_lattice_4x4):
    offsets = ab_bgk_lattice_4x4.collision_basis_offsets()

    assert offsets.shape == (ab_bgk_lattice_4x4.encoding.NUM_STATES,)
    assert len(set(offsets.tolist())) == ab_bgk_lattice_4x4.encoding.NUM_STATES


def test_basis_index_tables_are_consistent(ab_bgk_lattice_4x4):
    encoding = ab_bgk_lattice_4x4.encoding
    physical = ab_bgk_lattice_4x4.physical_basis_indices()
    unused = ab_bgk_lattice_4x4.unused_basis_indices()
    everything = ab_bgk_lattice_4x4.collision_basis_indices()

    assert physical.shape == (4, 4, encoding.NUM_POPULATIONS)
    assert unused.shape == (
        4,
        4,
        encoding.NUM_VELOCITY_STATES - encoding.NUM_POPULATIONS,
    )
    assert everything.shape == (4, 4, encoding.NUM_STATES)

    assert not set(physical.ravel().tolist()) & set(unused.ravel().tolist())
    assert set(physical.ravel().tolist()) <= set(everything.ravel().tolist())
    assert np.all(everything < (1 << ab_bgk_lattice_4x4.circuit.num_qubits))


def test_ancilla_mask_selects_only_non_interface_states(ab_bgk_lattice_4x4_obstacle):
    mask = ab_bgk_lattice_4x4_obstacle.ancilla_basis_mask()

    assert mask.shape == (1 << ab_bgk_lattice_4x4_obstacle.circuit.num_qubits,)
    assert mask.any()
    assert not mask[ab_bgk_lattice_4x4_obstacle.collision_basis_indices()].any()


def test_obstacle_lattice_keeps_a_single_marker(ab_bgk_lattice_4x4_obstacle):
    assert ab_bgk_lattice_4x4_obstacle.num_marker_qubits == 1
    assert len(ab_bgk_lattice_4x4_obstacle.shape_list) == 1
    assert ab_bgk_lattice_4x4_obstacle.circuit.num_qubits > 4 + 4 + 1


def test_lattice_reports_its_configuration(ab_bgk_lattice_4x4):
    assert "ABBGKLattice" in str(ab_bgk_lattice_4x4)
    assert "abbgklattice" in ab_bgk_lattice_4x4.logger_name()
