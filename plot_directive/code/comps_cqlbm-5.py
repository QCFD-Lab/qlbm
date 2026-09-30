from qlbm.components.ab import ABDiscreteUniformInitialConditions
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
    }
)

ABDiscreteUniformInitialConditions(lattice, [1, 3, 4], ([], [])).draw("mpl")