from qlbm.components.ab import ABStreamingOperator
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "d2q9"},
        "geometry": [],
    }
)

ABStreamingOperator(lattice).draw("mpl")