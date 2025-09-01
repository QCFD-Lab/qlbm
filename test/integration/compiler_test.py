from itertools import product

import pytest
from qiskit import QuantumCircuit as QiskitQC
from qiskit_aer import AerSimulator
from qulacs import QuantumCircuit as QulacsQC

from qlbm.components import CollisionlessStreamingOperator
from qlbm.infra import (
    CircuitCompiler,
)
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import get_time_series


@pytest.fixture
def lattice_asymmetric_medium_3d():
    return CollisionlessLattice("test/resources/asymmetric_3d_no_obstacles.json")


@pytest.fixture
def lattice_symmetric_small_2d():
    return CollisionlessLattice("test/resources/symmetric_2d_no_obstacles.json")


@pytest.mark.parametrize(
    "lattice_fixture,velocity,compiler_platform,backend_method",
    list(
        product(
            ["lattice_symmetric_small_2d", "lattice_asymmetric_medium_3d"],
            list(range(3)),
            ["QISKIT", "TKET"],
            ["statevector", "matrix_product_state"],
        )
    ),
)
def test_qiskit_target_compilation(
    lattice_fixture, velocity, compiler_platform, backend_method, request
):
    backend = AerSimulator(method=backend_method)
    lattice = request.getfixturevalue(lattice_fixture)
    num_velocities = (
        lattice.num_velocities[0] + 1
    )  # +1 because velocities are between 0 and n_vi
    velocities = get_time_series(num_velocities)[velocity]
    op = CollisionlessStreamingOperator(lattice, velocities)
    compiler = CircuitCompiler(compiler_platform, "QISKIT")

    compiled_circuit = compiler.compile(op, backend, 0)

    assert isinstance(compiled_circuit, QiskitQC)

# Qulacs is not currently supported due to qiskit 2.0
# @pytest.mark.parametrize(
#     "lattice_fixture,velocity,compiler_platform",
#     list(
#         product(
#             ["lattice_symmetric_small_2d", "lattice_asymmetric_medium_3d"],
#             list(range(3)),
#             ["QISKIT", "TKET"],
#         )
#     ),
# )
# def test_qulacs_target_compilation(
#     lattice_fixture, velocity, compiler_platform, request
# ):
#     lattice = request.getfixturevalue(lattice_fixture)
#     num_velocities = (
#         lattice.num_velocities[0] + 1
#     )  # +1 because velocities are between 0 and n_vi
#     velocities = get_time_series(num_velocities)[velocity]
#     op = CollisionlessStreamingOperator(lattice, velocities)
#     compiler = CircuitCompiler(compiler_platform, "QULACS")

#     # Qulacs backend is determined automatically
#     compiled_circuit = compiler.compile(op, None, 0)

#     assert isinstance(compiled_circuit, QulacsQC)
