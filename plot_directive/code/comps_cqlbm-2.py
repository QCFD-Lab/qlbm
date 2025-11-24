from qlbm.components import CQLBM
from qlbm.lattice import MSLattice

lattice = MSLattice(
    {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    }
)

CQLBM(lattice).draw("mpl")