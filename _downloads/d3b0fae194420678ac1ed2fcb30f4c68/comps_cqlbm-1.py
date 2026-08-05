from qlbm.components import CQLBM
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
        "geometry": [],
    }
)

CQLBM(lattice).draw("mpl")