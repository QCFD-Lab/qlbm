from qiskit.circuit.library import MCMT, XGate

from qlbm.components.common.primitives import MCSwap
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


def test_mcswap_13ctrl():
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": {"x": 2, "y": 2},
            },
        },
    )

    mcswap = MCSwap(lattice, [0, 2, 3], (5, 6))
    expected_circuit = lattice.circuit.copy()

    expected_circuit.cx(6, 5)
    expected_circuit.compose(
        MCMT(XGate(), 4, 1),
        qubits=[0, 2, 3, 5, 6],
        inplace=True,
    )
    expected_circuit.cx(6, 5)

    assert mcswap.circuit == expected_circuit


def test_mcswap_grid_ctrl():
    lattice = SpaceTimeLattice(
        1,
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": {"x": 2, "y": 2},
            },
        },
    )

    mcswap = MCSwap(
        lattice,
        lattice.grid_index(),
        (lattice.velocity_index(0, 1)[0], lattice.velocity_index(0, 3)[0]),
    )
    expected_circuit = lattice.circuit.copy()

    expected_circuit.cx(11, 9)
    expected_circuit.compose(
        MCMT(XGate(), 9, 1),
        qubits=list(range(8)) + [9, 11],
        inplace=True,
    )
    expected_circuit.cx(11, 9)

    assert mcswap.circuit == expected_circuit
