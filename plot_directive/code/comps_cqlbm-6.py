from qlbm.components.ab import ABDiscreteUniformInitialConditions
from qlbm.lattice import OHLattice

lattice = OHLattice(
    {
        "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
    }
)

ABDiscreteUniformInitialConditions(lattice, [0, 1], ([0, 1], [0])).draw("mpl")