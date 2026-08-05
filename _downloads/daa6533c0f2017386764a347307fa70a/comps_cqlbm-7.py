from qlbm.components.ab import ABParallelDiscreteUniformInitialConditions
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
    }
)

lattice.set_num_marker_qubits(2)

ABParallelDiscreteUniformInitialConditions(
    lattice,
    [[0, 1], [0, 3], [0], [0, 5]],
    [([0], [0])] * 4,
).draw("mpl")