from qlbm.components.ab import ABInitialConditions
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
        "geometry": [],
    }
)

ABInitialConditions(lattice).draw("mpl")