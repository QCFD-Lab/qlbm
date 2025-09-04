import pytest

from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretizationProperties


def test_lqlga_lattice_num_registers_d1q2(lattice_d1q2_256):
    assert len(lattice_d1q2_256.registers) == 256  # One empty register
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512


def test_lqlga_lattice_num_registers_multiple_geometries_once_one_geometry_d1q2(
    lattice_d1q2_256,
):
    assert len(lattice_d1q2_256.registers) == 256
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512

    lattice_d1q2_256.set_geometries(
        [[{"shape": "cuboid", "x": [12, 27], "boundary": "bounceback"}]]
    )

    assert len(lattice_d1q2_256.registers) == 256
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512


def test_lqlga_lattice_num_registers_multiple_geometries_once_two_geometries_d1q2(
    lattice_d1q2_256,
):
    assert len(lattice_d1q2_256.registers) == 256
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512

    lattice_d1q2_256.set_geometries(
        [
            [{"shape": "cuboid", "x": [12, 27], "boundary": "bounceback"}],
            [{"shape": "cuboid", "x": [12, 27], "boundary": "specular"}],
        ]
    )

    assert len(lattice_d1q2_256.registers) == 256 + 1
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512 + 1  # One extra marker qubit


def test_lqlga_lattice_num_registers_multiple_geometries_twice_d1q2(
    lattice_d1q2_256,
):
    assert len(lattice_d1q2_256.registers) == 256
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512

    lattice_d1q2_256.set_geometries(
        [
            [{"shape": "cuboid", "x": [12, 27], "boundary": "bounceback"}],
            [{"shape": "cuboid", "x": [12, 27], "boundary": "specular"}],
        ]
    )

    assert len(lattice_d1q2_256.registers) == 256 + 1
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512 + 1  # One extra marker qubit

    lattice_d1q2_256.set_geometries(
        [[{"shape": "cuboid", "x": [12, 27], "boundary": "bounceback"}]]
    )

    assert len(lattice_d1q2_256.registers) == 256
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512  # Marker qubit removed


def test_lqlga_lattice_num_registers_multiple_geometries_once_many_d1q2(
    lattice_d1q2_256,
):
    assert len(lattice_d1q2_256.registers) == 256
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512

    lattice_d1q2_256.set_geometries(
        [
            [{"shape": "cuboid", "x": [12, 27], "boundary": "bounceback"}]
            + (
                [{"shape": "cuboid", "x": [42, 43], "boundary": "specular"}]
                if i % 3 == 0
                else []
            )
            for i in range(67)
        ]
    )

    assert len(lattice_d1q2_256.registers) == 256 + 1
    assert all(reg.size == 2 for reg in lattice_d1q2_256.velocity_register)
    assert lattice_d1q2_256.circuit.num_qubits == 512 + 7  # 7 qubits to encode 67 geometries


def test_lqlga_lattice_num_registers_d2q4(lattice_d2q4_256_8):
    assert len(lattice_d2q4_256_8.registers) == 256 * 8
    assert all(reg.size == 4 for reg in lattice_d2q4_256_8.velocity_register)
    assert lattice_d2q4_256_8.circuit.num_qubits == 256 * 8 * 4


def test_lqlga_grid_index_mapping_edge():
    lattice = LQLGALattice(
        {
            "lattice": {
                "dim": {"x": 64, "y": 8},
                "velocities": "D2Q4",
            },
        },
    )

    assert lattice.gridpoint_index_flat(0) == (0, 0)
    assert lattice.gridpoint_index_flat(1) == (0, 1)
    assert lattice.gridpoint_index_flat(7) == (0, 7)
    assert lattice.gridpoint_index_flat(8) == (1, 0)
    assert lattice.gridpoint_index_flat(9) == (1, 1)
    assert lattice.gridpoint_index_flat(511) == (63, 7)


@pytest.mark.parametrize(
    "lattice_name",
    ["lattice_d2q4_256_8", "lattice_d1q2_256"],
)
def test_lqlga_grid_index_mapping_general(lattice_name, request):
    lattice = request.getfixturevalue(lattice_name)
    assert all(
        lattice.gridpoint_index_tuple(lattice.gridpoint_index_flat(gp)) == gp
        for gp in range(
            lattice.num_total_qubits
            // LatticeDiscretizationProperties.get_num_velocities(
                lattice.discretization
            )
        )
    )
