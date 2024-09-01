# import pytest
# from qiskit import AerSimulator
# from qiskit.result import Counts

# from qlbm.infra import CircuitCompiler
# from qlbm.infra.reinitialize import SpaceTimeReinitializer
# from qlbm.lattice import SpaceTimeLattice


# @pytest.fixture
# def spacetime_lattice():
#     return SpaceTimeLattice(1, "test/resources/symmetric_2d_1_obstacle_q4.json")


# @pytest.fixture
# def qiskit_compiler():
#     return CircuitCompiler("QISKIT", "QISKIT")


# def test_spacetime_reinitializer(spacetime_lattice, qiskit_compiler):
#     reinitializer = SpaceTimeReinitializer(
#         lattice=spacetime_lattice,
#         compiler=qiskit_compiler,
#     )
#     reinitializer.reinitialize(
#         Counts(
#             data={
#                 "000011011": 42,
#                 "010010111": 1,
#                 "111100001": 42,
#                 "000000000": 120,
#                 "000111111": 1,
#             },
#         ),
#     )
